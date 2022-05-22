clc;
clear all;
tx_pol = 1;
rx_pol = 1; 
init.fc = 6e9;  % 6GHz
f_SRS = 1/0.005; % 200Hz
lightspeed = 299792458;

f_doppler_min = f_SRS/200; % 100Hz
f_doppler_max = f_SRS/20; % 200Hz
v_min = (lightspeed*f_doppler_min)/init.fc;
v_max = (lightspeed*f_doppler_max)/init.fc;
num_FFT = 256;
for DS_ratio = [1]
    if DS_ratio == 0
        number_of_taps_to_save = 1;
    else
        number_of_taps_to_save = DS_ratio * 2;
    end    
    rng('default')
    disp('DS_ratio')
    disp(DS_ratio)
    disp('num taps to save')
    disp(number_of_taps_to_save)
    for num_tx_hor = [2]%[1,2,4]
        rng('default')
        for num_tx_ver = [2]%[1,2,4]
            rng('default')
            for num_rx_hor = [2]%[1,2]
                rng('default')
                for num_rx_ver =  [1]%[1,2]
                    rng('default')
                    for num_cluster_for_ch_gen = [19]
                        rng('default')
                        for val_mul = [1,20] 
                            disp('val mul')
                            disp(val_mul)
                            rng('default')
                            TOTAL_META_DATASET = 700;
                            META_TR_TASK_NUM = 500;
                            %% generate long term a priori
                            %% generate velocity a priori
                            %% so that consistent over num_antennas!
                            number_of_cluster_for_channel_generation = num_cluster_for_ch_gen;

                            num_antennas = [num_tx_hor,num_tx_ver,num_rx_hor,num_rx_ver,tx_pol,rx_pol];  %[num_tx_hor,num_tx_ver,num_rx_hor,num_rx_ver,2,1];%[2,2,2,1,2,1]; %[2,1,2,1,2,1]; 
                            disp('num_anteanns:');
                            disp(num_antennas);
                            %% custom 
                            %% simulation set-up
                            RS_pos = 13; % OFDM symbol index for RS
                            %% basic init. setting for all tasks 
                            init.DS = NaN;
                            init.mu = 1; 
                            init.num_RB = NaN; 

                            init.num_Tx_antenna_horizontal = num_antennas(1);            
                            init.num_Tx_antenna_vertical = num_antennas(2);
                            init.num_Rx_antenna_horizontal = num_antennas(3);
                            init.num_Rx_antenna_vertical = num_antennas(4);
                            init.Tx_pol = num_antennas(5);
                            init.Rx_pol = num_antennas(6);

                            init.ch_type = 'CDL_C';
                            init.total_rs_num = init.num_RB*12;
                            init.num_FFT = num_FFT; 
                            init.InitialTime = 0; 

                            meta_te_dataset = cell(TOTAL_META_DATASET,1);
                            rand_coup_AOD = NaN;
                            rand_coup_ZOD = NaN;
                            rand_coup_AOA = NaN;
                            rand_coup_ZOA = NaN;
                            mu_desired_AoD = NaN;
                            mu_desired_AoA = NaN;
                            mu_desired_ZoD = NaN;
                            mu_desired_ZoA = NaN;
                            prev_CDL = NaN;
                            Initial_Phase_for_all_rays = NaN;
                            for ind_meta_dataset = 1:TOTAL_META_DATASET % this is frame in the paper notation!
                                if ind_meta_dataset <= META_TR_TASK_NUM
                                    TOTAL_FRAME = 50; % TOTAL_FRAME*2 = number of total dataset (should be larger than training size + testing size)
                                else
                                    TOTAL_FRAME = 50; 
                                end
                                init.user_speed = rand*(v_max*val_mul-v_min*val_mul) + v_min*val_mul; %10 ~ 30 m/sx`
                                init.theta_v = rand*360 - 180;
                                init.phi_v = rand*360 - 180; 
                                if ind_meta_dataset > 1
                                    init.InitialTime = para.CurrentTime; % time keeps going!
                                end

                                para = module_Parameter_MIMO(init);
                                num_Tx_antenna = para.num_Tx_antenna;
                                num_Rx_antenna = para.num_Rx_antenna;
                                % for each task, at the very first, we need to generate long-term
                                para.c_ASD = 10; %degrees
                                para.c_ASA = 22;
                                para.c_ZSD = NaN; % not used!
                                para.c_ZSA = 7;
                                para.XPR_dB = 8; % use mean
                                d_2d = normrnd(100,10); % accounts for ue position change per frame
                                % spatial consistency
                                para.CDL = para.CDL(1:number_of_cluster_for_channel_generation,:);
                                original_total_clusters = size(para.CDL,1);
                                [fixed_spatial_for_scalar, DS_curr] = spatial_consistency_curr_frame(original_total_clusters, para.fc, d_2d);
                                para.DS = DS_curr * DS_ratio;
                                para.CDL = fixed_spatial_for_scalar;
                                % features
                                very_first = true;
                                XPR = NaN;
                                Rx_patterns = NaN;
                                Tx_patterns = NaN;
                                rhat_rx = NaN;
                                wl = NaN;
                                a_MS = NaN;
                                a_BS = NaN;
                                pathDelays = NaN;
                                pathPowers = NaN;
                                Num_ray = -1;%-1; % ray        
                                nTap = number_of_cluster_for_channel_generation;%1; % cluster
                                % currently XPR is fixed!
                                ind_data = 1;
                                multi_tap_channel = zeros(num_Tx_antenna*num_Rx_antenna*number_of_taps_to_save, 2*TOTAL_FRAME);
                                for ind_frame = 1:TOTAL_FRAME
                                    for ind_subframe = 1:10
                                        for ind_slot =  1:2^para.mu
                                            for ind_OFDM_symbol = 1:14
                                                if ind_OFDM_symbol == 1 | ind_OFDM_symbol == 8
                                                    symbol_duration = para.symbol_duration(1);
                                                else
                                                    symbol_duration = para.symbol_duration(2);
                                                end
                                                % generate channel 
                                                if (ind_OFDM_symbol == RS_pos) && (ind_slot == 1) && (rem(ind_subframe,5) == 1)
                                                    % cdl channel
                                                    [H_actual_tap, Initial_Phase_for_all_rays, XPR, Rx_patterns, Tx_patterns, rhat_rx, wl, a_MS, a_BS, Num_ray, nTap, rand_coup_AOD, rand_coup_AOA, rand_coup_ZOD, rand_coup_ZOA, pathDelays, pathPowers, user_speed, mu_desired_AoD, mu_desired_AoA, mu_desired_ZoD, mu_desired_ZoA] =  CDL_generation(para, very_first, Initial_Phase_for_all_rays, XPR, Rx_patterns, Tx_patterns, rhat_rx, wl, a_MS, a_BS, Num_ray, nTap, rand_coup_AOD, rand_coup_AOA, rand_coup_ZOD, rand_coup_ZOA, pathDelays, pathPowers, mu_desired_AoD, mu_desired_AoA, mu_desired_ZoD, mu_desired_ZoA, d_2d);
                                                    para.user_speed = user_speed;
                                                    if very_first == true
                                                        very_first = false;
                                                    else
                                                        ;
                                                    end
                                                    curr_channel = zeros(num_Tx_antenna*num_Rx_antenna,para.L); %zeros(para.L,num_Tx_antenna*num_Rx_antenna);
                                                    for ind_tx_antenna = 1:num_Tx_antenna
                                                        curr_channel((ind_tx_antenna-1)*num_Rx_antenna+1:(ind_tx_antenna-1)*num_Rx_antenna+num_Rx_antenna,:) = H_actual_tap(:, ind_tx_antenna, :);
                                                    end 
                                                    multi_tap_channel(:,ind_data) = reshape(curr_channel(:,1:number_of_taps_to_save), [num_Tx_antenna*num_Rx_antenna*number_of_taps_to_save,1]);
                                                    for ind_tap = 1:number_of_taps_to_save 
                                                        if num_Tx_antenna*num_Rx_antenna == 1
                                                            ;
                                                        else
                                                            ;
                                                        end
                                                        
                                                    end
                                                    ind_data = ind_data + 1;
                                                else
                                                    ;
                                                end
                                                para.CurrentTime = para.CurrentTime + symbol_duration;
                                            end
                                        end
                                    end
                                end
                                meta_te_dataset{ind_meta_dataset} = multi_tap_channel;
                                prev_CDL = para.CDL;
                            end
                            if Num_ray == -1
                                num_ray_for_saving_path = 20;
                            else
                                num_ray_for_saving_path = Num_ray;
                            end

                            rank_for_path = num_ray_for_saving_path * nTap;

                            currFolder = strcat('../../3gpp_channel_generated_data/cluster_', string(nTap), '/num_taps_to_save_', string(number_of_taps_to_save) + '/');
                            if ~exist(currFolder, 'dir')
                               mkdir(currFolder)
                            end
                            file_name = strcat(currFolder, 'num_Tx_antennas_', string(num_Tx_antenna),'num_tx_hor_', string(num_tx_hor), 'num_tx_ver', string(num_tx_ver), '_num_Rx_antennas_', string(num_Rx_antenna), 'num_rx_hor', string(num_rx_hor), 'num_rx_ver', string(num_rx_ver), 'tx_pol', string(tx_pol), 'rx_pol', string(rx_pol), 'val_mul', string(val_mul) , 'DS_ratio', string(DS_ratio), '.mat'); % for online 
                            save(file_name, 'meta_te_dataset','-v7.3');
                        end
                    end
                end
            end
        end
    end  
end



