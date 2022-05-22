import torch
import numpy as np
from numpy.linalg import inv
from funcs.mul_w_B_inner_opt import inner_optimization_ALS

class OLS:
    def __init__(self, beta_in_seq, window_length, lag, ar_from_psd, lambda_coeff=None, W_common_mean=None, noise_var=None, if_low_rank_for_regressor=None, rank_for_regressor=1, number_of_w=1, channel_dim=1, if_nmse_during_OLS=True): 
        assert ar_from_psd == None 
        self.beta_in_seq_complex = beta_in_seq[0]
        self.beta_in_seq_complex_noiseless = beta_in_seq[1] # deprecated now -- same with beta_in_seq_complex
        self.lambda_coeff = lambda_coeff
        self.noise_var = noise_var
        self.channel_dim = channel_dim
        self.number_of_w = number_of_w
        if W_common_mean is None:
            if if_low_rank_for_regressor:
                B_bar_tmp = {}
                w_bar_tmp = {}
                for k in range(self.number_of_w):
                    B_bar_tmp[k] = torch.zeros((self.channel_dim, 1), dtype=torch.cdouble)       
                    B_bar_tmp[k][k] = 1 
                    w_bar_tmp[k] = torch.zeros((window_length, 1), dtype=torch.cdouble)
                self.W_common_mean = [B_bar_tmp, w_bar_tmp]
            else:
                self.W_common_mean = torch.zeros((window_length*self.channel_dim, self.channel_dim), dtype=torch.cdouble)
        else:
            self.W_common_mean = W_common_mean
        self.if_low_rank_for_regressor = if_low_rank_for_regressor
        self.rank_for_regressor = rank_for_regressor
        self.if_nmse_during_OLS = if_nmse_during_OLS
        
    def coefficient_compute(self, window_length, lag, if_ridge, perf_stat=None, normalize_factor=None, x_te=None): #x_te is used for adapting lambda for meta-learning
        curr_beta = self.beta_in_seq_complex
        curr_beta_noiseless = self.beta_in_seq_complex_noiseless
        num_paths = curr_beta.shape[1]
        if self.if_low_rank_for_regressor:
            supp_mb = len(curr_beta)-window_length-lag+1 # number of total samples in the beta
            prev_channels = [] # all X_l s
            future_channels = [] # all y_l s
            for ind_sample in range(supp_mb): # use whole
                X_l = [] # S * T
                rand_start_ind = ind_sample + window_length-1
                y_l = curr_beta_noiseless[rand_start_ind+lag].unsqueeze(dim=1) # S*1
                for ind_col in range(window_length): # window_length = T 
                    curr_channel = curr_beta[rand_start_ind-ind_col]  # S
                    X_l.append(curr_channel.unsqueeze(dim=1)) # S * 1
                X_l = torch.cat((X_l), dim=1) # S * T
                if self.if_nmse_during_OLS:
                    norm_of_future_channel = torch.norm(y_l)
                else:
                    norm_of_future_channel = 1
                X_l /= norm_of_future_channel
                y_l /= norm_of_future_channel
                prev_channels.append(X_l) # l = 1,2,...,L
                future_channels.append(y_l)
            assert len(self.W_common_mean) == 2
            assert len(self.lambda_coeff) == 2
            B_bar = self.W_common_mean[0]
            w_bar = self.W_common_mean[1]
            A_adapted, B_adpated, w_adapted, _ = inner_optimization_ALS(None, None, False,  B_bar, w_bar, prev_channels, future_channels, self.lambda_coeff, self.rank_for_regressor, lag)
            assert A_adapted is None # A = B@Herm(B)
            self.W = []
            for k in range(len(w_adapted)):
                if A_adapted is None:
                    self.W.append(torch.kron(Trans(w_adapted[k]), B_adpated[k]@Herm(B_adpated[k])))
                else:
                    self.W.append(torch.kron(Trans(w_adapted[k]), A_adapted[k]))
        else:
            supp_mb = len(curr_beta)-window_length-lag+1 # number of total samples in the beta
            prev_channels = []
            future_channels = []
            if normalize_factor is not None:
                #normalize_factor = 1
                pass
            else: # unless specified, use normal approach
                normalize_factor = supp_mb
            for ind_row in range(supp_mb): # use whole
                curr_row = []
                rand_start_ind = ind_row + window_length-1
                future_channel = Herm(curr_beta_noiseless[rand_start_ind+lag]) # only noiseless when args.if_joint_noise_free and in that case, only for meta-training set 
                if self.if_nmse_during_OLS:
                    norm_of_future_channel = torch.norm(future_channel)
                else:
                    norm_of_future_channel = 1
                for ind_col in range(window_length):
                    curr_channel = Herm(curr_beta[rand_start_ind-ind_col]) # input should be noisy
                    curr_row.append(curr_channel.unsqueeze(dim=0))
                curr_row = torch.cat((curr_row), dim=1)
                curr_row /= norm_of_future_channel
                prev_channels.append(curr_row)
                future_channels.append(future_channel.unsqueeze(dim=0)/norm_of_future_channel)
                
            X = torch.cat(prev_channels, dim=0)
            Y = torch.cat(future_channels, dim=0)
            if if_ridge:
                H_herm_H = (Herm(X)@X)/normalize_factor 
                H_herm_h = (Herm(X)@Y)/normalize_factor
            else:
                if (supp_mb < X.shape[1]):
                    H_H_herm = (X@Herm(X)) 
                else:
                    H_herm_H = (Herm(X)@X)/normalize_factor # instead Herm(X)@X it would be better to directly average 
                    H_herm_h = (Herm(X)@Y)/normalize_factor # instead Herm(X)@Y it would be better to directly average 
        
        if self.if_low_rank_for_regressor:
            pass # already computed W
        else:
            # no need to decompose into A and B, directly optimize over W
            if if_ridge:
                H_herm_H += self.lambda_coeff * np.eye(H_herm_H.shape[0])
            else:
                pass
            if if_ridge:
                H_herm_h += self.lambda_coeff * self.W_common_mean 
            else:
                pass
            if perf_stat is not None:
                self.W = Herm(torch.from_numpy(inv(H_herm_H)) @ H_herm_h)
            else:
                if if_ridge:
                    self.W = Herm(torch.from_numpy(inv(H_herm_H)) @ H_herm_h)
                else:
                    if (supp_mb < X.shape[1]): ## need to consider gen inv also for the underdetermined case!!! # maybe if we change X.shape[1] to window_length, we may not have to worry about this!!
                        try:
                            self.W = Herm(Herm(X) @inv(H_H_herm) @ Y)
                        except np.linalg.LinAlgError:
                            self.W = Herm(Herm(X) @torch.linalg.pinv(H_H_herm) @ Y)
                    else:
                        try:
                            self.W = Herm(torch.from_numpy(inv(H_herm_H)) @ H_herm_h)
                        except np.linalg.LinAlgError:
                            self.W = Herm(torch.linalg.pinv(H_herm_H) @ H_herm_h)
            self.W = [self.W]
    def prediction(self, input_seq, window_length, lag):
        assert input_seq.shape[0] == window_length
        input_seq_complex = input_seq
        num_complex_paths = input_seq_complex.shape[1]
        pred_channel = 0
        for W in self.W:
            for ind_window in range(window_length):
                ind_window_reverse = window_length-1-ind_window
                pred_channel += W[:, ind_window*num_complex_paths:ind_window*num_complex_paths+num_complex_paths] @ input_seq_complex[ind_window_reverse].unsqueeze(dim=1).numpy()        
        return pred_channel

def Herm(vector):
    return np.transpose(np.conjugate(vector))

def Trans(vector):
    return np.transpose(vector)

def Herm_tensor(vector):
    return torch.transpose(torch.conj(vector),0,1)