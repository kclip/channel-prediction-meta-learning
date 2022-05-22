import torch
import numpy as np
import copy
from numpy.linalg import inv

class META_NAIVE:
    def __init__(self, beta_in_seq_total, supp_mb, window_length, lag, ar_from_psd, noise_var_meta, lambda_coeff=None, normalize_factor=None, if_low_rank_for_regressor=False, if_nmse_during_meta_transfer_training=False): 
        self.window_length = window_length
        self.lag = lag
        supp_mb = supp_mb-window_length-lag+1 # actual number of supp_mb -- actual number of training pairs -> this is L^tr -- before supp_mb is number of total available channels
        self.supp_mb = supp_mb
        self.lambda_coeff = lambda_coeff
        self.beta_in_seq_total = beta_in_seq_total
        self.noise_var_meta = 0.0  # if force this to be 0 regardless of input, we will just use pure meta and then directly use to meta-te .. similar to joint but quite naive..
        self.normalize_factor = normalize_factor
        self.lambda_dict = {}
        if self.lambda_coeff is None:
            for i in range(13):
                self.lambda_dict[i] = 10**(i-6)
        else:
            self.lambda_dict[0] = self.lambda_coeff # no grid search!! use given lambda!
        assert self.noise_var_meta == 0
        self.if_low_rank_for_regressor = if_low_rank_for_regressor
        self.if_nmse_during_meta_transfer_training = if_nmse_during_meta_transfer_training

    def grid_search(self, fixed_lambda_value, prev_v_bar_for_online_for_closed_form=None):
        if fixed_lambda_value is not None:
            assert fixed_lambda_value == self.lambda_coeff 
            curr_lambda = self.lambda_coeff 
            self.curr_lambda = torch.tensor(curr_lambda, dtype=torch.double, requires_grad=False)
            curr_common_mean = self.v_bar_update(prev_v_bar_for_online_for_closed_form)
            return curr_common_mean.detach(), curr_lambda
        else:
            pred_mse_best = 99999999999
            if prev_v_bar_for_online_for_closed_form is not None:
                raise NotImplementedError  ## this should not be considered for offline case !
            for ind_lambda in self.lambda_dict.keys():
                #print('grid search ind lambda', ind_lambda)
                curr_lambda = self.lambda_dict[ind_lambda]
                self.curr_lambda = torch.tensor(curr_lambda, dtype=torch.double, requires_grad=False)
                # find v_bar
                curr_common_mean = self.v_bar_update(prev_v_bar_for_online_for_closed_form)
                # get the predictive performance of this v_bar ### same dataset for now...
                pred_mse = self.lambda_update(curr_common_mean)
                pred_mse = pred_mse.real
                if pred_mse < pred_mse_best:
                    pred_mse_best = pred_mse
                    best_lambda = curr_lambda
                    best_common_mean = curr_common_mean
                else:
                    pass
            print('grid search best lambda', best_lambda)
            return best_common_mean.detach(), best_lambda
    
    def lambda_update(self, curr_common_mean):
        X_bar_H_X_bar = 0
        X_bar_H_Y_bar = 0 
        count_sample = 0
        total_possible_supp_query_pairs = 10 # for different random split of the dataset (regarding eq 6)
        pred_mse = 0
        for ind_task in range(len(self.beta_in_seq_total)):
            curr_beta = self.beta_in_seq_total[ind_task]
            for ind_possible_slot in range(total_possible_supp_query_pairs):
                ind_possible_slot = ind_possible_slot*10 + 3 # working as random split
                prev_channels = []
                future_channels = []
                for ind_row in range(self.supp_mb):
                    curr_row = []
                    supp_start_ind = ind_possible_slot + self.window_length-1 + ind_row
                    future_channel = Herm(curr_beta[supp_start_ind+self.lag])
                    if self.if_nmse_during_meta_transfer_training:
                        norm_of_future_channel = torch.norm(future_channel)
                    else:
                        norm_of_future_channel = 1
                    for ind_col in range(self.window_length):
                        curr_channel = Herm(curr_beta[supp_start_ind-ind_col])
                        curr_row.append(curr_channel.unsqueeze(dim=0))
                    curr_row = torch.cat((curr_row), dim=1)
                    curr_row /= norm_of_future_channel
                    prev_channels.append(curr_row)
                    future_channels.append(future_channel.unsqueeze(dim=0)/norm_of_future_channel)
                X = torch.cat(prev_channels, dim=0)
                Y = torch.cat(future_channels, dim=0)
                noise_X = torch.randn(X.shape, dtype=torch.cdouble)*np.sqrt(self.noise_var_meta) 
                X += noise_X
                if self.normalize_factor is not None:
                    pass
                else: # unless specified, use normal approach
                    self.normalize_factor = self.supp_mb
                C_inv = torch.inverse((Herm(X)@X/self.normalize_factor) + self.curr_lambda*torch.eye(X.shape[1])) #self.lambda_coeff*np.eye(window_length))
                start_ind_for_XY = ind_possible_slot + self.window_length-1
                last_ind_for_XY = ind_possible_slot + self.window_length-1 + self.supp_mb -1
                query_start_ind = 0 + self.window_length-1 # from very first, except for the supp channels
                query_start_ind_actual = query_start_ind*10 + 3

                while query_start_ind_actual <= len(self.beta_in_seq_total[0])-1-self.lag:# or count_sample == 100:
                    if start_ind_for_XY <= query_start_ind_actual <= last_ind_for_XY:
                        query_start_ind += 1
                        query_start_ind_actual = query_start_ind*10 + 3
                    else:
                        x_tmp = []
                        y_tmp = []
                        future_channel = Herm(curr_beta[query_start_ind_actual+self.lag])
                        if self.if_nmse_during_meta_transfer_training:
                            norm_of_future_channel = torch.norm(future_channel)
                        else:
                            norm_of_future_channel = 1
                        for ind_col in range(self.window_length):
                            curr_channel = Herm(curr_beta[query_start_ind_actual-ind_col]) # row vector
                            x_tmp.append(curr_channel.unsqueeze(dim=0))
                        x = torch.cat((x_tmp), dim=1) # one row
                        noise_x = torch.randn(x.shape, dtype=torch.cdouble)*np.sqrt(self.noise_var_meta)
                        x += noise_x # deprecated part
                        x /= norm_of_future_channel
                        y = future_channel.unsqueeze(dim=0)/norm_of_future_channel
                        x_bar = self.curr_lambda*x@C_inv 
                        y_bar = y - x_bar@((Herm(X)@Y)/(self.curr_lambda*self.normalize_factor))
                        pred_diff = (y_bar - x_bar @ curr_common_mean)
                        pred_mse += pred_diff @ Herm_tensor(pred_diff)
                        count_sample += 1
                        query_start_ind += 1
                        query_start_ind_actual = query_start_ind*10 + 3
        pred_mse /= count_sample
        return pred_mse

    def v_bar_update(self, prev_v_bar_for_online_for_closed_form):
        B = None
        X_bar_H_X_bar = 0
        X_bar_H_Y_bar = 0 
        count_sample = 0
        total_possible_supp_query_pairs = 1 #50 # consider given single split
        X_bar_list = [] 
        Y_bar_list = []
        for ind_task in range(len(self.beta_in_seq_total)):
            #print('ind task', ind_task)
            curr_beta = self.beta_in_seq_total[ind_task]
            for ind_possible_slot in range(total_possible_supp_query_pairs):
                # supp part
                prev_channels = []
                future_channels = []
                for ind_row in range(self.supp_mb):
                    curr_row = []
                    supp_start_ind = ind_possible_slot + self.window_length-1 + ind_row
                    future_channel = Herm(curr_beta[supp_start_ind+self.lag])
                    if self.if_nmse_during_meta_transfer_training:
                        norm_of_future_channel = torch.norm(future_channel)
                    else:
                        norm_of_future_channel = 1
                    for ind_col in range(self.window_length):
                        curr_channel = Herm(curr_beta[supp_start_ind-ind_col])
                        curr_row.append(curr_channel.unsqueeze(dim=0))
                    curr_row = torch.cat((curr_row), dim=1)
                    curr_row /= norm_of_future_channel
                    prev_channels.append(curr_row)
                    future_channels.append(future_channel.unsqueeze(dim=0)/norm_of_future_channel)
                X = torch.cat(prev_channels, dim=0)
                Y = torch.cat(future_channels, dim=0)
                noise_X = torch.randn(X.shape, dtype=torch.cdouble)*np.sqrt(self.noise_var_meta)  # deprecated
                X += noise_X
                if self.normalize_factor is not None:
                    pass
                else: # unless specified, use normal approach
                    self.normalize_factor = self.supp_mb
                C_inv = torch.inverse((Herm(X)@X/self.normalize_factor) + self.curr_lambda*torch.eye(X.shape[1])) #self.lambda_coeff*np.eye(window_length))
                # query part
                start_ind_for_XY = ind_possible_slot + self.window_length-1
                last_ind_for_XY = ind_possible_slot + self.window_length-1 + self.supp_mb -1
                query_start_ind = 0 + self.window_length-1 # from very first, except for the supp channels
                while query_start_ind <= len(self.beta_in_seq_total[0])-1-self.lag:
                    if start_ind_for_XY <= query_start_ind <= last_ind_for_XY:
                        query_start_ind += 1
                    else:
                        # now bar_x and bar_y for all every possible future channels
                        x_tmp = []
                        y_tmp = []
                        future_channel = Herm(curr_beta[query_start_ind+self.lag])
                        if self.if_nmse_during_meta_transfer_training:
                            norm_of_future_channel = torch.norm(future_channel)
                        else:
                            norm_of_future_channel = 1
                        for ind_col in range(self.window_length):
                            curr_channel = Herm(curr_beta[query_start_ind-ind_col]) # row vector
                            x_tmp.append(curr_channel.unsqueeze(dim=0))
                        x = torch.cat((x_tmp), dim=1) # one row
                        # only add noise to this x!!!!! use perfect for y!! -- deprecated
                        noise_x = torch.randn(x.shape, dtype=torch.cdouble)*np.sqrt(self.noise_var_meta)
                        x += noise_x # deprecated
                        x /= norm_of_future_channel
                        y = future_channel.unsqueeze(dim=0)/norm_of_future_channel
                        x_bar = self.curr_lambda*x@C_inv #self.lambda_coeff*C_inv@x 
                        y_bar = y - x_bar@((Herm(X)@Y)/(self.curr_lambda*self.normalize_factor))
                        X_bar_list.append(x_bar)
                        Y_bar_list.append(y_bar)

                        X_bar_H_X_bar += Herm_tensor(x_bar) @ x_bar
                        X_bar_H_Y_bar += Herm_tensor(x_bar) @ y_bar

                        count_sample += 1
                        query_start_ind += 1
        if count_sample == 0: # no meta-training task -- online at first
            self.common_mean = torch.zeros(1)
        else:
            X_bar_H_X_bar /= count_sample # not necessary -- for numerical stability purpose
            X_bar_H_Y_bar /= count_sample
            if prev_v_bar_for_online_for_closed_form is None:
                X_bar = torch.cat(X_bar_list, dim=0) # KL * Ns 
                Y_bar = torch.cat(Y_bar_list, dim=0)
                if X_bar.shape[0] > X_bar.shape[1]:
                    # least squares
                    self.common_mean = (torch.linalg.pinv(X_bar_H_X_bar))@X_bar_H_Y_bar
                else:
                    # least norm
                    self.common_mean = Herm(X_bar)@torch.linalg.pinv(X_bar@Herm(X_bar))@Y_bar
            else: # do ridge regression with prev_v_bar_for_online_for_closed_form -- deprecated
                raise NotImplementedError
                X_bar_H_X_bar_modified = X_bar_H_X_bar + 1 * np.eye(X_bar_H_X_bar.shape[0]) # we currently fix lambda for this as 1
                X_bar_H_Y_bar_modified = X_bar_H_Y_bar + 1 * prev_v_bar_for_online_for_closed_form
                self.common_mean = (torch.inverse(X_bar_H_X_bar_modified))@X_bar_H_Y_bar_modified
        return self.common_mean

def Herm(vector):
    return np.transpose(np.conjugate(vector))

def Herm_tensor(vector):
    return torch.transpose(torch.conj(vector),0,1)

