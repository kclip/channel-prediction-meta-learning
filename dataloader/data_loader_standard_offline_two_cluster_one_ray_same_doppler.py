import torch
import numpy as np
import scipy.io as sio
import h5py
import mat73
from dataloader.saved_paths_channels import saved_path_channels

class Doppler_Jake_dataloader_standard:
    def __init__(self, dataset_key, num_supp, num_query, if_jakes_rounded_toy, if_meta_tr, num_meta_tr_tasks, num_meta_te_tasks, noise_var, noise_var_meta, if_simple_multivariate_extension, total_num_antennas,meta_tr_tasks_randperm=None): # num_supp working as training data, num_query working as test data for conven. learning
        self.num_supp = num_supp 
        self.num_query= num_query
        self.noise_var = noise_var
        self.noise_var_meta = noise_var_meta

        if if_jakes_rounded_toy:
            raise NotImplementedError
        else:              

            saved_path_meta_tr = '..' + saved_path_channels[dataset_key]
            total_channels_meta_tr = mat73.loadmat(saved_path_meta_tr)
            
            saved_path_meta_te = saved_path_meta_tr
            total_channels_meta_te = mat73.loadmat(saved_path_meta_te)

        if if_meta_tr:
            random_seed = 9999
        else:
            random_seed = 99999
        torch.manual_seed(random_seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)

        if if_meta_tr:
            self.meta_tr_dict = {}
            self.meta_tr_dict_noiseless = {}
            if meta_tr_tasks_randperm is None:
                raise NotImplementedError
            else:
                pass
            ind_task_actual = 0
            for ind_task in meta_tr_tasks_randperm[:num_meta_tr_tasks]:
                if if_jakes_rounded_toy:
                    raise NotImplementedError
                else:
                    supp_query_samples_total = torch.from_numpy(total_channels_meta_tr['meta_te_dataset'][ind_task][0])
                    if len(supp_query_samples_total.shape) == 1:
                        supp_query_samples_total = supp_query_samples_total.unsqueeze(dim=0)
                    else:
                        pass
                    supp_query_samples_total = torch.transpose(supp_query_samples_total, 0,1)
                ## add noise!
                curr_task_noise = torch.randn(supp_query_samples_total.shape, dtype=torch.cdouble) # CN(0,1)
                curr_task_noise *= np.sqrt(self.noise_var_meta) # if var = 0 then noise is considerd during meta-learning or joint learning just use perfect samples with meta-te noisy samples jointly
                supp_query_samples_total_noisy = supp_query_samples_total + curr_task_noise
                self.meta_tr_dict[ind_task_actual] = supp_query_samples_total_noisy
                self.meta_tr_dict_noiseless[ind_task_actual] = supp_query_samples_total
                ind_task_actual += 1
        else:
            self.meta_te_dict = {}
            self.meta_te_dict_without_noise = {} # for only computing MSE performance
            self.gt_R = {}
            for ind_task in range(num_meta_te_tasks):
                if if_jakes_rounded_toy:
                    raise NotImplementedError
                else:
                    supp_query_samples_total = torch.from_numpy(total_channels_meta_te['meta_te_dataset'][ind_task+500][0])# 1000
                    if len(supp_query_samples_total.shape) == 1:
                        supp_query_samples_total = supp_query_samples_total.unsqueeze(dim=0)
                    else:
                        pass
                    supp_query_samples_total = torch.transpose(supp_query_samples_total, 0,1)
                    self.ar_from_psd = None
                curr_task_noise = torch.randn(supp_query_samples_total.shape, dtype=torch.cdouble) # CN(0,1)
                curr_task_noise *= np.sqrt(self.noise_var)
                supp_query_samples_total_noisy = supp_query_samples_total + curr_task_noise
                self.meta_te_dict[ind_task] = supp_query_samples_total_noisy
                self.meta_te_dict_without_noise[ind_task] = supp_query_samples_total
                self.gt_R[ind_task] = self.ar_from_psd

    def get_supp_samples_total(self, if_meta_tr, ind_task, num_supp=None):
        if num_supp is None:
            num_supp_curr = self.num_supp # for meta-tr
        else:
            num_supp_curr = num_supp # for varying exp. over num_supp
        if if_meta_tr:
            return self.meta_tr_dict[ind_task][:num_supp_curr]
        else: # should not use query set (test channels)
            assert num_supp_curr + self.num_query <= self.meta_te_dict[ind_task].shape[0] # total samples # unless we are using more pilots than available!
            return self.meta_te_dict[ind_task][:num_supp_curr]


    def get_all_possible_querys(self, window_length, lag, ind_mte_task): # since we are not using in-batch compute. it is fine to use single mb to get the test result # mb only works for averaging grads
        query_samples_total = self.meta_te_dict[ind_mte_task][-self.num_query:] # use last self.num_query channels for test
        query_samples_total_without_noise = self.meta_te_dict_without_noise[ind_mte_task][-self.num_query:]
        all_possible_querys = []
        total_queries = len(query_samples_total)-window_length-lag+1
        for curr_ind in range(total_queries):
            curr_input_mb = query_samples_total[curr_ind:curr_ind+window_length]
            curr_output_mb = query_samples_total[curr_ind+window_length-1+lag] # noisy target which may not be used at all...
            curr_output_mb_without_noise = query_samples_total_without_noise[curr_ind+window_length-1+lag]
            all_possible_querys.append([curr_input_mb, curr_output_mb, curr_output_mb_without_noise])
        return all_possible_querys
