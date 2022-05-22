import torch
import numpy as np
from dataloader.data_loader_standard_offline_two_cluster_one_ray_same_doppler import Doppler_Jake_dataloader_standard
from funcs.meta_linear_filter import META_NAIVE
from funcs.meta_lstd_linear_filter import META_LSTD
from funcs.transfer_linear_filter import OLS_transfer
from funcs.ordinary_least_square import OLS
from funcs.aic import aic_for_rank
import copy

NUMBER_OF_SRS_PER_FRAME_FOR_EXTRA_FRAMES = 100

def one_mc_trial(args, curr_dir, num_supp, num_query, velocity_kmph, ind_mc):
    loss = torch.nn.MSELoss(reduction='sum')
    if args.linear_ridge_mode == 2:
        args.num_meta_tr_tasks = 0
    mse_wiener_tr = -999
    if args.Jake_dataloader is None:
        if args.fading_mode == 4: 
            Jake_dataloader = Doppler_Jake_dataloader_standard(dataset_key = args.dataset_key, num_supp=num_supp, num_query=num_query, if_jakes_rounded_toy=args.if_jakes_rounded_toy, 
                                                                        if_meta_tr=False, num_meta_tr_tasks=args.num_meta_tr_tasks, num_meta_te_tasks=args.num_meta_te_tasks,  noise_var = args.noise_var, noise_var_meta = args.noise_var_meta, if_simple_multivariate_extension=args.if_simple_multivariate_extension,total_num_antennas=args.total_num_antennas)
        else:
            raise NotImplementedError
        args.Jake_dataloader = Jake_dataloader # since only supp is different for offline case -- we can assign is directly via get_supp_samples_total
        # for online, we need to renew always.
    else:
        Jake_dataloader = args.Jake_dataloader
    if args.Jake_dataloader_meta is None:
        meta_training_total_samples_per_tasks = NUMBER_OF_SRS_PER_FRAME_FOR_EXTRA_FRAMES
        if args.fading_mode == 4:
            random_seed = ind_mc
            torch.manual_seed(random_seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(random_seed)
            meta_tr_tasks_randperm = torch.randperm(500)
            Jake_dataloader_meta =  Doppler_Jake_dataloader_standard(dataset_key = args.dataset_key, num_supp=meta_training_total_samples_per_tasks, num_query= 0, if_jakes_rounded_toy=args.if_jakes_rounded_toy, if_meta_tr=True, num_meta_tr_tasks=args.num_meta_tr_tasks, num_meta_te_tasks=args.num_meta_te_tasks, noise_var = args.noise_var, noise_var_meta = args.noise_var_meta, if_simple_multivariate_extension=args.if_simple_multivariate_extension, total_num_antennas=args.total_num_antennas, meta_tr_tasks_randperm=meta_tr_tasks_randperm)
        else:
            raise NotImplementedError
        args.Jake_dataloader_meta = Jake_dataloader_meta # it cannot directly controlled via get_supp_samples_total when num_supp changes -- since actual dataset is different. we need to redefine per num_supp.
    else:
        Jake_dataloader_meta = args.Jake_dataloader_meta


    ## aic for rank
    if args.linear_ridge_mode == 2:
        assert args.num_meta_tr_tasks == 0

    if args.if_aic_to_determine_K:
        if args.num_meta_tr_tasks == 0:
            pass
        else:
            assert args.linear_ridge_mode == 11 or args.linear_ridge_mode == 0
            beta_total_for_pca = []
            for ind_task_mtr in range(args.num_meta_tr_tasks): 
                curr_mtr_data = Jake_dataloader_meta.get_supp_samples_total(if_meta_tr=True, ind_task=ind_task_mtr)  # num_samples * S (dim)
                beta_total_for_pca.append(curr_mtr_data) 
            total_X= torch.cat(beta_total_for_pca, dim=0) # many_num_samples(dim) * S
            args.number_of_w = aic_for_rank(total_X)

    # initialize B_bar
    if args.if_low_rank_for_regressor:
        B_bar_init = torch.zeros(args.total_num_antennas, args.number_of_w,dtype=torch.cdouble)
        for ind_w in range(args.number_of_w):
            B_bar_init[ind_w, ind_w] = 1
    else:
        B_bar_init = None


    snr_curr = -999
    # WF
    if args.linear_ridge_mode == 0: # 0: meta, 2: conventional learning, 11: transfer learning
        #### meta-learning ####
        beta_in_seq_total = []
        for ind_task_mtr in range(args.num_meta_tr_tasks):
            curr_mtr_data = Jake_dataloader_meta.get_supp_samples_total(if_meta_tr=True, ind_task=ind_task_mtr)
            beta_in_seq_total.append(curr_mtr_data)
        
        if args.ridge_meta is None:
            if args.if_mtr_fix_supp:
                num_supp_ridge_meta = args.meta_training_fixed_supp # not the number of training samples (L^tr) -- this is number of total availble samples = L^tr + N+delta-1
            else:
                num_supp_ridge_meta = num_supp
            if args.if_low_rank_for_regressor: # if genie B is None, then just do full rank!!
                # low rank & full structured without knowledge of B
                lambda_coeff = [args.lambda_b_bar, args.lambda_w_bar]
                ridge_meta = META_LSTD(common_path=args.common_path, beta_in_seq_total=beta_in_seq_total, supp_mb=num_supp_ridge_meta, window_length=args.window_length, lag=args.lag, ar_from_psd=None, noise_var_meta= args.noise_var,  lambda_coeff=lambda_coeff, normalize_factor=args.normalize_factor_meta_ridge, if_low_rank_for_regressor=args.if_low_rank_for_regressor, rank_for_regressor=args.rank_for_regressor, lr_EP=args.lr_EP, B_bar_init=B_bar_init, number_of_w=args.number_of_w, channel_dim=args.total_num_antennas, if_get_query_loss_mse=args.if_get_query_loss_mse, if_EP_start_from_free_stationary=args.if_EP_start_from_free_stationary, if_nmse_during_meta_transfer_training=args.if_nmse_during_meta_transfer_training, if_low_rank_meta_self_determin_num_w=args.if_low_rank_meta_self_determin_num_w) #lambda_1, lambda_2
                    
                B_bar, w_bar = ridge_meta.v_bar_update()
                adapted_common_mean = [B_bar, w_bar]
                adapted_lambda = None
            else:
                # low rank with genie B and full rank both covered here
                lambda_coeff=args.ridge_lambda_coeff
                ridge_meta = META_NAIVE(beta_in_seq_total=beta_in_seq_total, supp_mb=num_supp_ridge_meta, window_length=args.window_length, lag=args.lag, ar_from_psd=None, noise_var_meta= args.noise_var, lambda_coeff=lambda_coeff, normalize_factor=args.normalize_factor_meta_ridge, if_low_rank_for_regressor=args.if_low_rank_for_regressor, if_nmse_during_meta_transfer_training=args.if_nmse_during_meta_transfer_training) # use noise_var_meta as same as that will be used at meta-te
                adapted_common_mean, adapted_lambda = ridge_meta.grid_search(args.fixed_lambda_value, args.prev_v_bar_for_online_for_closed_form)
        else:
            raise NotImplementedError # now we are only considering adapting lambda for each case!
        mse_wiener = 0
        nmse_wiener = 0
        mse_wiener_tr = 0
        for ind_task_mte in range(args.num_meta_te_tasks):   
            curr_training_seq = Jake_dataloader.get_supp_samples_total(if_meta_tr=False, ind_task=ind_task_mte, num_supp=num_supp)  
            ar_from_psd = None
            perf_stat = None
            assert args.if_ridge == True
            if args.if_ridge:
                if adapted_lambda is not None:
                    lambda_coeff = adapted_lambda # may be adapted
                else:
                    pass # use fixed one
            else:
                lambda_coeff = None

            if args.if_wiener_filter_as_ols:
                WF = OLS(beta_in_seq = (curr_training_seq, curr_training_seq), window_length=args.window_length, lag=args.lag, ar_from_psd=ar_from_psd, lambda_coeff = lambda_coeff, W_common_mean=adapted_common_mean, noise_var=args.noise_var, if_low_rank_for_regressor=args.if_low_rank_for_regressor, number_of_w=args.number_of_w, channel_dim=args.total_num_antennas,  if_nmse_during_OLS=args.if_nmse_during_OLS)
            else:
                raise NotImplementedError
            WF.coefficient_compute(window_length=args.window_length, lag=args.lag, if_ridge=args.if_ridge, perf_stat=perf_stat, normalize_factor=args.normalize_factor)          
            mse_wiener_curr_mte_task = 0
            nmse_wiener_curr_mte_task = 0
            all_possible_test = Jake_dataloader.get_all_possible_querys(window_length=args.window_length, lag=args.lag, ind_mte_task=ind_task_mte)
            for test_ch in all_possible_test:
                pred_ch = WF.prediction(input_seq = test_ch[0], window_length=args.window_length, lag=args.lag)
                pred_ch_in_real = torch.flatten(torch.view_as_real(pred_ch))
                true_ch_in_real = torch.flatten(torch.view_as_real(test_ch[2]))
                mse_wiener_curr_mte_task += loss(pred_ch_in_real, true_ch_in_real)#/(torch.norm(true_ch_in_real)**2) # compare to perfect channel
                nmse_wiener_curr_mte_task += loss(pred_ch_in_real, true_ch_in_real)/(torch.norm(true_ch_in_real)**2) # compare to perfect channel
            mse_wiener_curr_mte_task /= len(all_possible_test)
            nmse_wiener_curr_mte_task /= len(all_possible_test)
            mse_wiener += mse_wiener_curr_mte_task
            nmse_wiener += nmse_wiener_curr_mte_task
        mse_wiener /= args.num_meta_te_tasks
        nmse_wiener /= args.num_meta_te_tasks
    elif args.linear_ridge_mode == 2:
        #### conventional learning ####
        mse_wiener = 0
        mse_wiener_tr = 0
        nmse_wiener = 0
        for ind_task_mte in range(args.num_meta_te_tasks):   
            curr_training_seq = Jake_dataloader.get_supp_samples_total(if_meta_tr=False, ind_task=ind_task_mte, num_supp=num_supp)  
            if args.if_aic_to_determine_K:
                args.number_of_w = aic_for_rank(curr_training_seq)
            else:
                pass            
            ar_from_psd = None
            perf_stat = None
            if args.if_ridge:
                if args.if_low_rank_for_regressor:
                    lambda_coeff = [args.lambda_b_bar, args.lambda_w_bar]
                else:
                    lambda_coeff = args.ridge_lambda_coeff
            else:
                if args.if_low_rank_for_regressor:
                    lambda_coeff = [0.0, 0.0]
                else:
                    lambda_coeff = None
            if args.if_wiener_filter_as_ols:
                WF = OLS(beta_in_seq = (curr_training_seq, curr_training_seq), window_length=args.window_length, lag=args.lag, ar_from_psd=ar_from_psd, lambda_coeff = lambda_coeff, W_common_mean=None, noise_var=args.noise_var, if_low_rank_for_regressor=args.if_low_rank_for_regressor, number_of_w=args.number_of_w, channel_dim=args.total_num_antennas, if_nmse_during_OLS=args.if_nmse_during_OLS)
            else:
                raise NotImplementedError
            WF.coefficient_compute(window_length=args.window_length, lag=args.lag, if_ridge=args.if_ridge, perf_stat=perf_stat, normalize_factor=args.normalize_factor)

            mse_wiener_curr_mte_task = 0
            nmse_wiener_curr_mte_task = 0
            all_possible_test = Jake_dataloader.get_all_possible_querys(window_length=args.window_length, lag=args.lag, ind_mte_task=ind_task_mte)
            for test_ch in all_possible_test:
                pred_ch = WF.prediction(input_seq = test_ch[0], window_length=args.window_length, lag=args.lag)
                pred_ch_in_real = torch.flatten(torch.view_as_real(pred_ch))
                true_ch_in_real = torch.flatten(torch.view_as_real(test_ch[2]))
                mse_wiener_curr_mte_task += loss(pred_ch_in_real, true_ch_in_real)#/(torch.norm(true_ch_in_real)**2) # compare to perfect channel
                nmse_wiener_curr_mte_task += loss(pred_ch_in_real, true_ch_in_real)/(torch.norm(true_ch_in_real)**2) # compare to perfect channel
            mse_wiener_curr_mte_task /= len(all_possible_test)
            nmse_wiener_curr_mte_task /= len(all_possible_test)
            mse_wiener += mse_wiener_curr_mte_task
            nmse_wiener += nmse_wiener_curr_mte_task
        mse_wiener /= args.num_meta_te_tasks
        nmse_wiener /= args.num_meta_te_tasks
    elif args.linear_ridge_mode == 11:
        #### tranfer learning ####
        # concatenate all available data
        beta_in_seq_total = []
        for ind_task_mtr in range(args.num_meta_tr_tasks):
            curr_mtr_data = Jake_dataloader_meta.get_supp_samples_total(if_meta_tr=True, ind_task=ind_task_mtr)
            beta_in_seq_total.append(curr_mtr_data)
        ## during transfer learning -- no ridge regression!
        if args.if_low_rank_for_regressor:
            lambda_coeff = [0.0, 0.0]
        else:
            lambda_coeff = None            
        adapted_common_mean = None
        if args.if_wiener_filter_as_ols:
            WF_joint = OLS_transfer(beta_in_seq_total = beta_in_seq_total, window_length=args.window_length, lag=args.lag, ar_from_psd=None, lambda_coeff = lambda_coeff, W_common_mean=adapted_common_mean, noise_var=args.noise_var, if_low_rank_for_regressor=args.if_low_rank_for_regressor, number_of_w=args.number_of_w, channel_dim=args.total_num_antennas,  if_nmse_during_meta_transfer_training=args.if_nmse_during_meta_transfer_training, B_bar_init=B_bar_init)
        else:
            raise NotImplementedError
        
        if args.num_meta_tr_tasks == 0:
            if args.if_low_rank_for_regressor:
                lambda_coeff = [args.lambda_b_bar, args.lambda_w_bar]
                B_tranfer = WF_joint.W_common_mean[0]
                w_tranfer = WF_joint.W_common_mean[1]
                adapted_common_mean = [B_tranfer, w_tranfer]
                adapted_lambda = None
            else:
                lambda_coeff=args.ridge_lambda_coeff
                W_tranfer = WF_joint.W_common_mean # since it is single element list
                adapted_common_mean = W_tranfer
                adapted_lambda = None
        else:
            WF_joint.coefficient_compute(window_length=args.window_length, lag=args.lag, if_ridge=False, normalize_factor=args.normalize_factor)
            if args.if_low_rank_for_regressor:
                lambda_coeff = [args.lambda_b_bar, args.lambda_w_bar]
                B_tranfer = WF_joint.B_adpated
                w_tranfer = WF_joint.w_adapted
                adapted_common_mean = [B_tranfer, w_tranfer]
                adapted_lambda = None
            else:
                lambda_coeff=args.ridge_lambda_coeff
                W_tranfer = torch.transpose(torch.conj(WF_joint.W[0]),0,1) # since it is single element list
                adapted_common_mean = W_tranfer
                adapted_lambda = None

        mse_wiener = 0
        mse_wiener_tr = 0
        nmse_wiener = 0
        for ind_task_mte in range(args.num_meta_te_tasks):   
            curr_training_seq = Jake_dataloader.get_supp_samples_total(if_meta_tr=False, ind_task=ind_task_mte, num_supp=num_supp)  
            ar_from_psd = None
            perf_stat = None
            assert args.if_ridge == True
            if args.if_ridge:
                if adapted_lambda is not None:
                    lambda_coeff = adapted_lambda # may be adapted
                else:
                    pass # use fixed one
            else:
                lambda_coeff = None

            if args.if_wiener_filter_as_ols:
                WF = OLS(beta_in_seq = (curr_training_seq, curr_training_seq), window_length=args.window_length, lag=args.lag, ar_from_psd=ar_from_psd, lambda_coeff = lambda_coeff, W_common_mean=adapted_common_mean, noise_var=args.noise_var, if_low_rank_for_regressor=args.if_low_rank_for_regressor, number_of_w=args.number_of_w, channel_dim=args.total_num_antennas,if_nmse_during_OLS=args.if_nmse_during_OLS)
            else:
                raise NotImplementedError
            WF.coefficient_compute(window_length=args.window_length, lag=args.lag, if_ridge=args.if_ridge, perf_stat=perf_stat, normalize_factor=args.normalize_factor)         
            mse_wiener_curr_mte_task = 0
            nmse_wiener_curr_mte_task = 0
            all_possible_test = Jake_dataloader.get_all_possible_querys(window_length=args.window_length, lag=args.lag, ind_mte_task=ind_task_mte)
            for test_ch in all_possible_test:
                pred_ch = WF.prediction(input_seq = test_ch[0], window_length=args.window_length, lag=args.lag)
                pred_ch_in_real = torch.flatten(torch.view_as_real(pred_ch))
                true_ch_in_real = torch.flatten(torch.view_as_real(test_ch[2]))
                mse_wiener_curr_mte_task += loss(pred_ch_in_real, true_ch_in_real)#/(torch.norm(true_ch_in_real)**2) # compare to perfect channel
                nmse_wiener_curr_mte_task += loss(pred_ch_in_real, true_ch_in_real)/(torch.norm(true_ch_in_real)**2) # compare to perfect channel
            mse_wiener_curr_mte_task /= len(all_possible_test)
            nmse_wiener_curr_mte_task /= len(all_possible_test)
            mse_wiener += mse_wiener_curr_mte_task
            nmse_wiener += nmse_wiener_curr_mte_task
        mse_wiener /= args.num_meta_te_tasks
        nmse_wiener /= args.num_meta_te_tasks
    else:
        raise NotImplementedError

    return snr_curr, mse_wiener, mse_wiener_tr, nmse_wiener
