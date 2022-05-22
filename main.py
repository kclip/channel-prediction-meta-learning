import torch
import numpy
import numpy as np
import scipy.io as sio
from train_and_test.tr_te import one_mc_trial
import argparse
import time
import os
from torch.utils.tensorboard import SummaryWriter

if_fix_random_seed = True
random_seed = 1

def parse_args():
    parser = argparse.ArgumentParser(description='offline scalar channel prediction')
    parser.add_argument('--window_length', type=int, default=5, help='window length for channel prediction (size of covariate vector (N))')
    parser.add_argument('--lag', type=int, default=3, help='prediction lag (delta)') 
    parser.add_argument('--if_ridge', dest='if_ridge', action='store_true', default=False, help='whether to consider ridge regression or ordinary least squares')
    parser.add_argument('--linear_ridge_mode', type=int, default=2, help='0: meta learning by closed form, 1: joint learning, 2: conventional learning, 6: meta-learning by implicit gradient theorem, 7: meta-learning by EP, 11:transfer learning')
    parser.add_argument('--ridge_lambda_coeff', type=float, default=1, help='coefficient for the regularizor (lamdba)') 
    parser.add_argument('--meta_training_samples_per_task', type=int, default=None, help='number of samples for meta-training dataset (L = L^tr + L^te)') 
    parser.add_argument('--num_meta_tr_tasks', type=int, default=None, help='number of meta-training tasks (determined automatically)') 
    parser.add_argument('--num_meta_te_tasks', type=int, default=None, help='number of meta-testing tasks (determined automatically)') 
    parser.add_argument('--fading_mode', type=int, default=4, help='4: offline spatial consistency') 
    parser.add_argument('--if_jakes_rounded_toy', dest='if_jakes_rounded_toy', action='store_true', default=False) # deprecated
    parser.add_argument('--if_mtr_fix_supp', dest='if_mtr_fix_supp', action='store_true', default=False,  help='whether to use fixed supp size (deprecated, we now control this in the main file)') # deprecated
    parser.add_argument('--meta_training_fixed_supp', type=int, default=8, help='fixed supp size for meta-training if args.if_mtr_fix_supp = True (deprecated now)')
    parser.add_argument('--if_not_wiener_filter_as_ols', dest='if_wiener_filter_as_ols', action='store_false', default=True, help ='consider ordinary least squares for regressor (only considering this way)')
    parser.add_argument('--normalize_factor', type=int, default=1, help='normalize factor for ridge regression during meta-testing, we are not considering normalization now. if 1 no normalize')
    parser.add_argument('--normalize_factor_meta_ridge', type=int, default=1, help='normalize factor for ridge regression during meta-training, we are not considering normalization now. if 1 no normalize')
    parser.add_argument('--if_simple_multivariate_extension', dest='if_simple_multivariate_extension', action='store_true', default=False) # deprecated 
    parser.add_argument('--multivariate_expansion_dim_from_doppler_scalar', type=int, default=None, help='simple extension for vector case')
    parser.add_argument('--noise_var_for_exp_over_supp', type=float, default=1e-4, help='channel estimation noise variance for experiment over supp')
    parser.add_argument('--fixed_lambda_value', type=float, default=1, help='if None, use grid search, else: fix with this lambda always')
    # vector case
    parser.add_argument('--lr_EP', type=float, default=0.01, help='step size for EP')
    parser.add_argument('--if_low_rank_for_regressor', dest='if_low_rank_for_regressor', action='store_true', default=False, help='impose low rank constraint for regressor')
    parser.add_argument('--rank_for_regressor', type=int, default=1, help='always one since we are doing sequential update for every low-rank schemes')
    parser.add_argument('--lambda_b_bar', type=float, default=1, help='lambda for b')
    parser.add_argument('--lambda_w_bar', type=float, default=1, help='lambda for v')
    parser.add_argument('--if_not_EP_start_from_free_stationary', dest='if_EP_start_from_free_stationary', action='store_false', default=True, help='whether to start next EP from previous point or random point')    
    parser.add_argument('--if_do_not_get_query_loss_mse', dest='if_get_query_loss_mse', action='store_false', default=None, help='None: follow default setting (if adam, true other false)')        
    parser.add_argument('--curr_path_name', type=str, default=None, help='name for each scheme for saving')
    parser.add_argument('--if_nmse_during_meta_transfer_training', dest='if_nmse_during_meta_transfer_training', action='store_true', default=False, help='nmse during transfer and meta-learning')    
    parser.add_argument('--if_nmse_during_OLS', dest='if_nmse_during_OLS', action='store_true', default=False, help='nmse during conventional learning')    
    parser.add_argument('--if_low_rank_meta_self_determin_num_w', dest='if_low_rank_meta_self_determin_num_w', action='store_true', default=False, help='proposed rank selection scheme for meta-learning')    
    parser.add_argument('--if_considering_normalize_for_supp_size_always_for_low_rank', dest='if_considering_normalize_for_supp_size_always_for_low_rank', action='store_true', default=False, help='normalizing with number of training samples for LSTD ')    
    parser.add_argument('--if_aic_to_determine_K', dest='if_aic_to_determine_K', action='store_true', default=False, help='use AIC for rank estimation')    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    start_time = time.time()
    args = parse_args()
    

    torch.set_default_dtype(torch.double)
    torch.set_default_tensor_type(torch.DoubleTensor)

    if args.linear_ridge_mode == 0:
        assert args.meta_training_samples_per_task == None # we are going to set as curr supp (L^new) + fixed query (100)
    else:
        pass
    
    if args.if_low_rank_for_regressor and args.if_ridge:
        pass
    elif args.if_ridge:
        pass
    else:
        pass

    if args.num_meta_tr_tasks is None:
        args.num_meta_tr_tasks = 500 #480#280
    else:
        pass
    if args.num_meta_te_tasks is None:
        args.num_meta_te_tasks = 100 #10*2 
    else:
        pass
    ## we do not consider normalization at all for all cases
    assert args.normalize_factor_meta_ridge == 1
    assert args.normalize_factor == 1
    assert args.rank_for_regressor == 1 # for now -- mul w case

    if if_fix_random_seed:
        torch.manual_seed(random_seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        numpy.random.seed(random_seed)
    else:
        pass

    args.window_length = 5 
    main_mode = 'vector'
    args.if_nmse_during_meta_transfer_training = True
    args.if_nmse_during_OLS = True
    args.dataset_key_common = 'final_vector_channel_'
    args.num_meta_te_tasks = 200 
    args.if_considering_normalize_for_supp_size_always_for_low_rank = True
    
    print('Called with args:')
    print(args)


    DS_ratio_list = [1] 
    available_velocity_list = [1, 20]
    L_new_for_mte = [1,2,5,10,20]
    num_total_antennas_list = [8]
    num_cluster_list = [19] 
    args.number_of_w = None 
    num_taps = None
    meta_training_tasks_list = [500]
    mc_num = 1
    available_supp_list = [L_new + args.window_length+args.lag-1 for L_new in L_new_for_mte]


    curr_dir = '../../../saved_results/' + main_mode + '/' + str(args.dataset_key_common) + '/num_task_' + str(meta_training_tasks_list[0]) + '/num_supp_' + str(available_supp_list[0]) + '/num_w_' + str(args.number_of_w) + '/' + args.curr_path_name + '/nmse_' + str(args.if_nmse_during_meta_transfer_training) + str(args.if_nmse_during_OLS) + '/win_len_' + str(args.window_length) + '/' + 'normalize_factor_' + str(args.normalize_factor) + '/'
    
    args.common_path = curr_dir + 'during_meta_training/'
    if os.path.isdir(curr_dir):
        pass
    else:
        os.makedirs(curr_dir)
    if os.path.isdir(args.common_path):
        pass
    else:
        os.makedirs(args.common_path)

    eval_results_path = curr_dir + 'test_result.mat'

    noise_var_list = [args.noise_var_for_exp_over_supp] 
    lambda_coeff_list = [args.fixed_lambda_value]
    mse_curr_net_per_supp_wiener_tr = torch.zeros(1, len(DS_ratio_list), len(num_total_antennas_list), len(meta_training_tasks_list), len(num_cluster_list), len(lambda_coeff_list), len(noise_var_list), mc_num, len(available_supp_list), len(available_velocity_list))
    mse_curr_net_per_supp_wiener = torch.zeros(1, len(DS_ratio_list), len(num_total_antennas_list), len(meta_training_tasks_list), len(num_cluster_list), len(lambda_coeff_list), len(noise_var_list), mc_num, len(available_supp_list), len(available_velocity_list))
    nmse_curr_net_per_supp_wiener = torch.zeros(1, len(DS_ratio_list), len(num_total_antennas_list), len(meta_training_tasks_list), len(num_cluster_list), len(lambda_coeff_list), len(noise_var_list), mc_num, len(available_supp_list), len(available_velocity_list))
    args.ridge_meta = None # once we compute this, do not need to repeat this again!
    args.common_mean_joint = None # for mode 3
    args.prev_v_bar_for_online_for_closed_form = None # this is for online -- for offline we do not consider continual closed form case # if this is None, original ridge meta

    
    for ind_w in range(1):
        for ind_DS in range(len(DS_ratio_list)):
            print('DS ratio', DS_ratio_list[ind_DS]) #[256, 512, 1024, 2048, 4096]
            num_taps = int(DS_ratio_list[ind_DS] * 2) # two taps contain 90% power at original DS (ratio=1)
            args.number_of_w = DS_ratio_list[ind_DS]*2*num_total_antennas_list[0] # full number of w possible
            for ind_num_antenna in range(len(num_total_antennas_list)):
                args.total_num_antennas = num_total_antennas_list[ind_num_antenna] * num_taps
                print('num antennas', num_total_antennas_list[ind_num_antenna])
                for ind_meta_training_tasks in range(len(meta_training_tasks_list)):
                    args.num_meta_tr_tasks = meta_training_tasks_list[ind_meta_training_tasks]
                    print('number of meta-training tasks', args.num_meta_tr_tasks)
                    for ind_cluster in range(len(num_cluster_list)):
                        for ind_lambda in range(len(lambda_coeff_list)):
                            args.ridge_lambda_coeff = lambda_coeff_list[ind_lambda]
                            ind_snr = 0
                            for noise_var in noise_var_list:
                                args.noise_var = noise_var
                                args.noise_var_meta = noise_var
                                for ind_velocity in range(len(available_velocity_list)):
                                    velocity_ratio = available_velocity_list[ind_velocity]
                                    if velocity_ratio == 1:
                                        print('slow-varying environment')
                                    elif velocity_ratio == 20:
                                        print('fast-varying environment')
                                    else:
                                        pass
                                    ind_supp = 0
                                    for num_supp in available_supp_list:
                                        if args.if_considering_normalize_for_supp_size_always_for_low_rank:
                                            args.lambda_b_bar = num_supp - args.window_length-args.lag+1
                                            args.lambda_w_bar = num_supp - args.window_length-args.lag+1
                                            args.ridge_lambda_coeff = num_supp - args.window_length-args.lag+1
                                            args.fixed_lambda_value = num_supp - args.window_length-args.lag+1
                                        else:
                                            pass                                  
                                        args.num_samples_for_test = 100 - num_supp                                    
                                        print('number of training samples (L^new): ', num_supp - (args.window_length+args.lag -1))
                                        for ind_mc in range(mc_num): # mc for different realization of training (adaptation) set. especially for small number of dataset
                                            args.dataset_key = args.dataset_key_common + 'num_antennas_' + str(num_total_antennas_list[ind_num_antenna]) + 'num_clusters_' + str(num_cluster_list[ind_cluster]) + 'num_taps_' + str(num_taps) + 'vel_mul_' + str(velocity_ratio) + 'DS_ratio_' + str(DS_ratio_list[ind_DS])
                                            args.Jake_dataloader = None
                                            args.ridge_meta = None
                                            args.Jake_dataloader_meta = None
                                            args.v_bar_curr = None # need to start from scratch for common mean
                                            velocity_kmph_actual = None
                                            args.prev_v_bar_for_online_for_closed_form = None # this is for online -- for offline we do not consider continual closed form case # if this is None, original ridge meta
                                            snr_curr, mse_wiener, mse_wiener_tr, nmse_wiener = one_mc_trial(args, curr_dir, num_supp, args.num_samples_for_test, velocity_kmph_actual, ind_mc)
                                            mse_wiener /= args.total_num_antennas
                                            print('nmse', nmse_wiener)
                                            mse_curr_net_per_supp_wiener_tr[ind_w, ind_DS, ind_num_antenna, ind_meta_training_tasks, ind_cluster, ind_lambda, ind_snr, ind_mc, ind_supp, ind_velocity] = mse_wiener_tr
                                            mse_curr_net_per_supp_wiener[ind_w, ind_DS, ind_num_antenna, ind_meta_training_tasks, ind_cluster, ind_lambda, ind_snr, ind_mc, ind_supp, ind_velocity] = mse_wiener              
                                            nmse_curr_net_per_supp_wiener[ind_w, ind_DS, ind_num_antenna, ind_meta_training_tasks, ind_cluster, ind_lambda, ind_snr, ind_mc, ind_supp, ind_velocity] = nmse_wiener              
                                        ind_supp += 1
                                        eval_results = {}
                                        eval_results['mse_wiener'] = mse_curr_net_per_supp_wiener.data.numpy() # mean, first dim, second dim
                                        eval_results['mse_wiener_tr'] = mse_curr_net_per_supp_wiener_tr.data.numpy()
                                        eval_results['nmse_wiener'] = nmse_curr_net_per_supp_wiener.data.numpy()
                                        sio.savemat(eval_results_path, eval_results)
                                ind_snr += 1
    print('eval_results_path', eval_results_path)


