import torch
import numpy as np
import copy
import warnings
from funcs.ordinary_least_square import OLS
from funcs.mul_w_B_inner_opt import inner_optimization_ALS, inner_loss, opt_multiple_w_gauss_seidel, opt_multiple_B_gauss_seidel
import scipy.io as sio
from scipy import spatial
from numpy.linalg import inv

MAX_ITER = 50 
CONV_CRITERIA = 1e-4 

class META_LSTD:
    def __init__(self, common_path,  beta_in_seq_total, supp_mb, window_length, lag, ar_from_psd, noise_var_meta,lambda_coeff, normalize_factor, if_low_rank_for_regressor, rank_for_regressor, lr_EP, B_bar_init,  number_of_w = 1, channel_dim = None, if_get_query_loss_mse=None,  if_EP_start_from_free_stationary=False,  if_nmse_during_meta_transfer_training=False, if_low_rank_meta_self_determin_num_w=False, total_iter_EP=50000, coeff_EP=0.01, num_mb_task=100, if_do_iter_exp=False, if_adam=True): 
        self.window_length = window_length
        self.lag = lag
        supp_mb = supp_mb-window_length-lag+1 # actual number of supp_mb -- actual number of training pairs -> this is L^tr -- before supp_mb is number of total available channels
        self.supp_mb = supp_mb
        self.curr_lambda = lambda_coeff
        self.beta_in_seq_total = beta_in_seq_total
        self.rank_for_regressor = rank_for_regressor
        self.noise_var_meta = 0.0  # if force this to be 0 regardless of input, we will just use pure meta and then directly use to meta-te .. similar to joint but quite naive..
        self.normalize_factor = normalize_factor
        assert self.noise_var_meta == 0
        assert self.normalize_factor == 1
        assert if_low_rank_for_regressor == True
        self.total_iter_EP = total_iter_EP
        self.lr_EP = lr_EP
        self.coeff_EP = coeff_EP
        self.num_mb_task = num_mb_task
        self.B_bar_init = B_bar_init
        self.if_do_iter_exp = if_do_iter_exp
        if self.curr_lambda[0] == 0 and self.curr_lambda[1] == 0:
            self.total_iter_EP = 0
        else:
            pass
        self.if_adam = if_adam
        if self.if_adam:
            self.num_mb_task =  10 #len(self.beta_in_seq_total) // 50 # 100
            if self.num_mb_task > len(self.beta_in_seq_total):
                self.num_mb_task = len(self.beta_in_seq_total)
            else:
                pass
            self.if_get_query_loss_mse = True
        else:
            self.if_get_query_loss_mse = True
        if if_get_query_loss_mse is not None:
            self.if_get_query_loss_mse = if_get_query_loss_mse # follow the input setting
        else:
            pass
        
        self.number_of_w = number_of_w
        self.channel_dim = channel_dim
        self.if_low_rank_for_regressor = if_low_rank_for_regressor
        self.common_path = common_path
        self.if_EP_start_from_free_stationary = if_EP_start_from_free_stationary
        self.iter_for_mse_and_mte = 1
        self.num_tasks_for_query_loss_val = 20
        self.iter_for_mse_and_mte = 1 #*= self.number_of_w
        self.total_iter_EP *= self.number_of_w
        self.per_factor_total_iter_EP = self.total_iter_EP//self.number_of_w # for each factor!

        self.if_nmse_during_meta_transfer_training = if_nmse_during_meta_transfer_training
        self.if_low_rank_meta_self_determin_num_w = if_low_rank_meta_self_determin_num_w

    def v_bar_update(self):
        total_possible_supp_query_pairs = 1
        ### get Z_tr and Z_te
        X_tr_dict = {} # X_tr_dict[ind_task] -> gives list of all X_l 
        Y_tr_dict = {}
        X_te_dict = {}
        Y_te_dict = {}
        for ind_task in range(len(self.beta_in_seq_total)):
            curr_beta = self.beta_in_seq_total[ind_task]
            for ind_possible_slot in range(total_possible_supp_query_pairs):
                prev_channels_tr = [] # all X_l s
                future_channels_tr = [] # all y_l s
                for ind_sample in range(self.supp_mb): # use whole
                    X_l = [] # S * T
                    supp_start_ind = ind_possible_slot + self.window_length-1 + ind_sample
                    y_l = curr_beta[supp_start_ind+self.lag].unsqueeze(dim=1) # S*1
                    if self.if_nmse_during_meta_transfer_training:
                        norm_of_future_channel = torch.norm(y_l)
                    else:
                        norm_of_future_channel = 1
                    for ind_col in range(self.window_length): # window_length = T 
                        curr_channel = curr_beta[supp_start_ind-ind_col]  # S
                        X_l.append(curr_channel.unsqueeze(dim=1)) # S * 1
                    X_l = torch.cat((X_l), dim=1) # S * T
                    X_l /= norm_of_future_channel
                    y_l /= norm_of_future_channel
                    prev_channels_tr.append(X_l) # l = 1,2,...,L
                    future_channels_tr.append(y_l)

                X_tr_dict[ind_task] = prev_channels_tr
                Y_tr_dict[ind_task] = future_channels_tr
                
                start_ind_for_XY = ind_possible_slot + self.window_length-1
                last_ind_for_XY = ind_possible_slot + self.window_length-1 + self.supp_mb -1
                query_start_ind = 0 + self.window_length-1 # from very first, except for the supp channels
                prev_channels_te = []
                future_channels_te = []
                while query_start_ind <= len(self.beta_in_seq_total[0])-1-self.lag:
                    if start_ind_for_XY <= query_start_ind <= last_ind_for_XY:
                        query_start_ind += 1
                    else:
                        X_l_te = [] # S * T
                        y_l_te = curr_beta[query_start_ind+self.lag].unsqueeze(dim=1) # S*1
                        if self.if_nmse_during_meta_transfer_training:
                            norm_of_future_channel_te = torch.norm(y_l_te)
                        else:
                            norm_of_future_channel_te = 1
                        for ind_col in range(self.window_length): # window_length = T 
                            curr_channel = curr_beta[query_start_ind-ind_col]  # S
                            X_l_te.append(curr_channel.unsqueeze(dim=1)) # S * 1
                        X_l_te = torch.cat((X_l_te), dim=1) # S * T
                        X_l_te /= norm_of_future_channel_te
                        y_l_te /= norm_of_future_channel_te
                        prev_channels_te.append(X_l_te)
                        future_channels_te.append(y_l_te)
                        query_start_ind += 1
                X_te_dict[ind_task] = prev_channels_te
                Y_te_dict[ind_task] = future_channels_te

        ### now do EP
        B_bar_curr = {}
        w_bar_best = {}
        w_bar_curr = {}
        B_bar_best = {}
        for k in range(self.number_of_w):
            B_bar_curr[k] = self.B_bar_init[:, k]
            B_bar_curr[k] = B_bar_curr[k].unsqueeze(dim=1)
            B_bar_best[k] = self.B_bar_init[:, k]
            B_bar_best[k] = B_bar_best[k].unsqueeze(dim=1)
            w_bar_curr[k] = torch.zeros((self.window_length, 1), dtype=torch.cdouble)        
            w_bar_best[k] = torch.zeros((self.window_length, 1), dtype=torch.cdouble)
        if len(self.beta_in_seq_total) == 0:
            B_bar_trim = B_bar_best
            w_bar_trim = w_bar_best
        else:
            curr_mean_mse_per_outer = torch.zeros(self.total_iter_EP)
            if self.if_do_iter_exp:
                mse_over_iteration_over_task = {} # for each ind_iter!
                for ind_task in range(len(self.beta_in_seq_total)):
                    mse_over_iteration_over_task[ind_task] = {}
            else:
                pass
            best_mean_mse = 999999999999
            best_mean_mse_prev = 9999999999999

            prev_m_B_bar = {}
            prev_v_B_bar = {}
            prev_m_w_bar = {}
            prev_v_w_bar = {}
            for k in range(self.number_of_w):
                prev_m_w_bar[k] = 0
                prev_v_w_bar[k] = 0
            for k in range(self.number_of_w):
                prev_m_B_bar[k] = 0
                prev_v_B_bar[k] = 0

            if self.if_get_query_loss_mse:
                eval_results_during_mtr = {}
                mte_mse_per_mtr_iter = []
                mte_iter_value_per_mtr_iter = []
                eval_results_path = self.common_path + 'results_during_mtr.mat'
                B_bar_best_path = self.common_path + 'B_bar_best.pt'
                w_bar_best_path = self.common_path + 'w_bar_best.pt'
            else:
                pass

            if self.if_adam:
                iter_for_each_particle = {}
                for k in range(self.number_of_w):
                    iter_for_each_particle[k] = 0
            else:
                pass
            
            ind_iter = 0
            remaining_iter_for_curr_factor = 0 # no meaning unless strategy 3
            count_for_overfitting = 0
            while ind_iter < (self.total_iter_EP):
                curr_meta_grad_B_bar = {}
                curr_meta_grad_w_bar = {}
                for k in range(self.number_of_w):
                    curr_meta_grad_w_bar[k] = torch.zeros(w_bar_curr[k].shape, dtype=torch.cdouble)
            
                for k in range(self.number_of_w):
                    curr_meta_grad_B_bar[k] = torch.zeros(B_bar_curr[k].shape, dtype=torch.cdouble)
                loss_curr_iter = 0
                if self.num_mb_task <= len(self.beta_in_seq_total):
                    if self.num_tasks_for_query_loss_val + 20 <= len(self.beta_in_seq_total): # sufficiently smaller
                        if_meta_val = True
                        task_mb = torch.randperm(len(self.beta_in_seq_total)-self.num_tasks_for_query_loss_val)
                    else:
                        if_meta_val = False ## use all F for query!
                        task_mb = torch.randperm(len(self.beta_in_seq_total))
                    task_mb = task_mb[:self.num_mb_task]
                else:
                    raise NotImplementedError
                    task_mb = torch.arange(len(self.beta_in_seq_total)-self.num_tasks_for_query_loss_val)

                curr_k = ind_iter // self.per_factor_total_iter_EP
                if ind_iter % self.per_factor_total_iter_EP == 0: # when starting to new factor
                    remaining_iter_for_curr_factor = self.per_factor_total_iter_EP
                    count_for_overfitting = 0
                    if curr_k == 0:
                        pass
                    else:
                        prev_k = curr_k - 1 
                        if self.if_get_query_loss_mse:
                            # update to the best!
                            B_bar_curr[prev_k] = copy.deepcopy(B_bar_best[prev_k])
                            w_bar_curr[prev_k] = copy.deepcopy(w_bar_best[prev_k])
                        else:
                            pass
                        # should be done similarly during meta-testing period!
                        for ind_task in range(len(self.beta_in_seq_total)):
                            curr_B = copy.deepcopy(B_bar_curr)
                            curr_w = copy.deepcopy(w_bar_curr)
                            curr_loss = 9999999999999
                            conv_criteria = CONV_CRITERIA #1e-4
                            curr_single_w = {}
                            curr_single_B = {}
                            curr_single_w_bar = {}
                            curr_single_B_bar = {}
                            curr_single_w[0] = curr_w[prev_k]
                            curr_single_B[0] = curr_B[prev_k]
                            curr_single_w_bar[0] = w_bar_curr[prev_k]
                            curr_single_B_bar[0] = B_bar_curr[prev_k]

                            for iter in range(MAX_ITER): 
                                curr_single_w = opt_multiple_w_gauss_seidel(curr_single_B_bar, curr_single_w_bar, X_tr_dict[ind_task], Y_tr_dict[ind_task], self.curr_lambda[1], curr_single_B, curr_single_w, max_iter=1)
                                curr_single_B = opt_multiple_B_gauss_seidel(curr_single_B_bar, curr_single_w_bar, X_tr_dict[ind_task], Y_tr_dict[ind_task], self.curr_lambda[0], curr_single_w, self.rank_for_regressor, curr_single_B, max_iter=1)
                                loss_tmp = inner_loss(curr_single_B_bar, curr_single_w_bar, X_tr_dict[ind_task], Y_tr_dict[ind_task], self.curr_lambda[0], self.curr_lambda[1], curr_single_B, curr_single_w)
                                if loss_tmp<=conv_criteria:
                                    break 
                                elif (loss_tmp <= curr_loss) and (torch.norm(curr_loss - loss_tmp) <= conv_criteria):
                                    break
                                else:
                                    curr_loss = loss_tmp
                            B_0_k = curr_single_B[0]
                            w_0_k = curr_single_w[0]
                            for ind_sample in range(len(Y_tr_dict[ind_task])):
                                X_l_tr = X_tr_dict[ind_task][ind_sample]
                                Y_tr_dict[ind_task][ind_sample] -= B_0_k@Herm(B_0_k)@X_l_tr@w_0_k
                            for ind_sample in range(len(Y_te_dict[ind_task])):
                                X_l_te = X_te_dict[ind_task][ind_sample]
                                Y_te_dict[ind_task][ind_sample] -= B_0_k@Herm(B_0_k)@X_l_te@w_0_k
                else:
                    pass
                if self.if_adam:
                    iter_for_each_particle[curr_k] += 1
                else:
                    pass
                for ind_task in task_mb:
                    ind_task = int(ind_task)
                    # EP inner
                    curr_B = copy.deepcopy(B_bar_curr)
                    curr_w = copy.deepcopy(w_bar_curr)
                    curr_loss = 9999999999999
                    conv_criteria = CONV_CRITERIA #1e-4
                    curr_single_w = {}
                    curr_single_B = {}
                    curr_single_w_bar = {}
                    curr_single_B_bar = {}
                    curr_single_w[0] = curr_w[curr_k]
                    curr_single_B[0] = curr_B[curr_k]
                    curr_single_w_bar[0] = w_bar_curr[curr_k]
                    curr_single_B_bar[0] = B_bar_curr[curr_k]
                    for iter in range(MAX_ITER): 
                        curr_single_w = opt_multiple_w_gauss_seidel(curr_single_B_bar, curr_single_w_bar, X_tr_dict[ind_task], Y_tr_dict[ind_task], self.curr_lambda[1], curr_single_B, curr_single_w, max_iter=1)
                        curr_single_B = opt_multiple_B_gauss_seidel(curr_single_B_bar, curr_single_w_bar, X_tr_dict[ind_task], Y_tr_dict[ind_task], self.curr_lambda[0], curr_single_w, self.rank_for_regressor, curr_single_B, max_iter=1)
                        loss_tmp = inner_loss(curr_single_B_bar, curr_single_w_bar, X_tr_dict[ind_task], Y_tr_dict[ind_task], self.curr_lambda[0], self.curr_lambda[1], curr_single_B, curr_single_w)
                        if loss_tmp<=conv_criteria:
                            break 
                        elif (loss_tmp <= curr_loss) and (torch.norm(curr_loss - loss_tmp) <= conv_criteria):
                            break
                        else:
                            curr_loss = loss_tmp
                    B_0_k = copy.deepcopy(curr_single_B[0])
                    w_0_k = copy.deepcopy(curr_single_w[0])
                    # EP outer
                    X_te_weighted = [x_te*np.sqrt(self.coeff_EP) for x_te in X_te_dict[ind_task]]
                    Y_te_weighted = [y_te*np.sqrt(self.coeff_EP) for y_te in Y_te_dict[ind_task]]
                    new_X = X_tr_dict[ind_task] + X_te_weighted
                    new_Y = Y_tr_dict[ind_task] + Y_te_weighted
                    ##
                    if self.if_EP_start_from_free_stationary:
                        curr_B = copy.deepcopy(curr_single_B)
                        curr_w = copy.deepcopy(curr_single_w)
                        curr_single_w = {}
                        curr_single_B = {}
                        curr_single_w[0] = curr_w[0]
                        curr_single_B[0] = curr_B[0]
                    else:
                        raise NotImplementedError
                        curr_B = copy.deepcopy(B_bar_curr)
                        curr_w = copy.deepcopy(w_bar_curr)
                        curr_single_w = {}
                        curr_single_B = {}
                        curr_single_w[0] = curr_w[curr_k]
                        curr_single_B[0] = curr_B[curr_k]

                    curr_loss = 9999999999999
                    conv_criteria = CONV_CRITERIA #1e-4
                    for iter in range(MAX_ITER): 
                        curr_single_w = opt_multiple_w_gauss_seidel(curr_single_B_bar, curr_single_w_bar, new_X, new_Y, self.curr_lambda[1], curr_single_B, curr_single_w, max_iter=1)
                        curr_single_B = opt_multiple_B_gauss_seidel(curr_single_B_bar, curr_single_w_bar, new_X, new_Y, self.curr_lambda[0], curr_single_w, self.rank_for_regressor, curr_single_B, max_iter=1)
                        loss_tmp = inner_loss(curr_single_B_bar, curr_single_w_bar, new_X, new_Y, self.curr_lambda[0], self.curr_lambda[1], curr_single_B, curr_single_w)
                        if loss_tmp<=conv_criteria:
                            break 
                        elif (loss_tmp <= curr_loss) and (torch.norm(curr_loss - loss_tmp) <= conv_criteria):
                            break
                        else:
                            curr_loss = loss_tmp
                    B_new_k = copy.deepcopy(curr_single_B[0])
                    w_new_k = copy.deepcopy(curr_single_w[0])
                    curr_meta_grad_w_bar[curr_k] += (2*self.curr_lambda[1]/self.coeff_EP) * (w_0_k - w_new_k)
                    curr_meta_grad_B_bar[curr_k] += (2*self.curr_lambda[0]/self.coeff_EP) * (B_0_k@Herm(B_0_k) - B_new_k@Herm(B_new_k) ) @ B_bar_curr[curr_k]
                    
                for k in range(self.number_of_w):
                    curr_meta_grad_B_bar[k] /= len(task_mb)
                for k in range(self.number_of_w):
                    curr_meta_grad_w_bar[k] /= len(task_mb)
                if self.if_get_query_loss_mse:
                    if ind_iter % self.iter_for_mse_and_mte == 0: # 100..
                        curr_mean_mse = 0
                        for ind_task in range(len(self.beta_in_seq_total)):
                            if if_meta_val:
                                F_meta_val_start_ind = len(self.beta_in_seq_total) - self.num_tasks_for_query_loss_val
                            else:
                                F_meta_val_start_ind = 0 # use all for query loss and also for self rank det.
                            if ind_task >= F_meta_val_start_ind: #len(self.beta_in_seq_total) - self.num_tasks_for_query_loss_val:
                                curr_B = copy.deepcopy(B_bar_curr)
                                curr_w = copy.deepcopy(w_bar_curr)
                                curr_loss = 9999999999999
                                conv_criteria = CONV_CRITERIA #1e-4
                                curr_single_w = {}
                                curr_single_B = {}
                                curr_single_w_bar = {}
                                curr_single_B_bar = {}
                                curr_single_w[0] = curr_w[curr_k]
                                curr_single_B[0] = curr_B[curr_k]
                                curr_single_w_bar[0] = w_bar_curr[curr_k]
                                curr_single_B_bar[0] = B_bar_curr[curr_k]
                                for iter in range(MAX_ITER): 
                                    curr_single_w = opt_multiple_w_gauss_seidel(curr_single_B_bar, curr_single_w_bar, X_tr_dict[ind_task], Y_tr_dict[ind_task], self.curr_lambda[1], curr_single_B, curr_single_w, max_iter=1)
                                    curr_single_B = opt_multiple_B_gauss_seidel(curr_single_B_bar, curr_single_w_bar, X_tr_dict[ind_task], Y_tr_dict[ind_task], self.curr_lambda[0], curr_single_w, self.rank_for_regressor, curr_single_B, max_iter=1)
                                    loss_tmp = inner_loss(curr_single_B_bar, curr_single_w_bar, X_tr_dict[ind_task], Y_tr_dict[ind_task], self.curr_lambda[0], self.curr_lambda[1], curr_single_B, curr_single_w)
                                    if loss_tmp<=conv_criteria:
                                        break 
                                    elif (loss_tmp <= curr_loss) and (torch.norm(curr_loss - loss_tmp) <= conv_criteria):
                                        break
                                    else:
                                        curr_loss = loss_tmp
                                mse_query_curr_task = inner_loss(curr_single_B_bar, curr_single_w_bar, X_te_dict[ind_task], Y_te_dict[ind_task], 0.0, 0.0, curr_single_B, curr_single_w)
                                curr_mean_mse += mse_query_curr_task
                            else:
                                pass
                        if curr_mean_mse < best_mean_mse:
                            B_bar_best = copy.deepcopy(B_bar_curr) #B_bar_curr.clone().detach()
                            w_bar_best = copy.deepcopy(w_bar_curr) #w_bar_curr.clone().detach()
                            best_mean_mse = curr_mean_mse
                            torch.save(B_bar_best, B_bar_best_path)
                            torch.save(w_bar_best, w_bar_best_path)
                            count_for_overfitting = 0
                        else:
                            count_for_overfitting += 1 # means that updating is inefficient..!
                    else:
                        pass
                else:
                    pass
                
                if self.if_adam:
                    adam_grad_w_bar, prev_m_w_bar[curr_k], prev_v_w_bar[curr_k] = Adam(curr_meta_grad_w_bar[curr_k], prev_m_w_bar[curr_k], prev_v_w_bar[curr_k], iter_for_each_particle[curr_k])
                    adam_grad_B_bar, prev_m_B_bar[curr_k], prev_v_B_bar[curr_k] = Adam(curr_meta_grad_B_bar[curr_k], prev_m_B_bar[curr_k], prev_v_B_bar[curr_k], iter_for_each_particle[curr_k])
                    w_bar_curr[curr_k] -= self.lr_EP * adam_grad_w_bar
                    B_bar_curr[curr_k] -= self.lr_EP * adam_grad_B_bar
                else:
                    for k in range(self.number_of_w):
                        w_bar_curr[k] -= self.lr_EP * curr_meta_grad_w_bar[k]
                        B_bar_curr[k] -= self.lr_EP * curr_meta_grad_B_bar[k]

                if count_for_overfitting < 20:
                    ind_iter += 1
                    remaining_iter_for_curr_factor -= 1
                else:
                    ind_iter += remaining_iter_for_curr_factor
                    if best_mean_mse == best_mean_mse_prev: #means no additional factor is needed!
                        if self.if_low_rank_meta_self_determin_num_w:
                            B_bar_trim = {}
                            w_bar_trim = {}
                            for k in range(curr_k): # 0,..., curr_k - 1
                                B_bar_trim[k] = copy.deepcopy(B_bar_best[k])
                                w_bar_trim[k] = copy.deepcopy(w_bar_best[k])
                            break
                        else:
                            pass
                    else:
                        B_bar_trim = B_bar_best
                        w_bar_trim = w_bar_best
                    best_mean_mse_prev = best_mean_mse
        if self.if_do_iter_exp or self.if_get_query_loss_mse:
            if self.if_low_rank_meta_self_determin_num_w:
                return B_bar_trim, w_bar_trim
            else:
                return B_bar_best, w_bar_best
        else:
            return B_bar_curr, w_bar_curr

def Herm(vector):
    return np.transpose(np.conjugate(vector))

def Herm_tensor(vector):
    return torch.transpose(torch.conj(vector),0,1)

def Adam(curr_grad, prev_m, prev_v, iter, beta_1 = 0.9, beta_2 = 0.999, epsilon=1e-8):
    # iter should start from 1!
    curr_m = beta_1*prev_m + (1-beta_1)*curr_grad
    curr_v = beta_2*prev_v + (1-beta_2)*torch.mul(curr_grad, curr_grad)
    curr_m_hat = curr_m/(1-beta_1**iter)
    curr_v_hat = curr_v/(1-beta_2**iter)
    adam_grad = curr_m_hat/(epsilon+torch.sqrt(curr_v_hat))
    return adam_grad, curr_m, curr_v 

