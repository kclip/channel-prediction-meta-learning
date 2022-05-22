function [H_actual_tap, Initial_Phase_for_all_rays, XPR, Rx_patterns, Tx_patterns, rhat_rx, wl, a_MS, a_BS, Num_ray, nTap, rand_coup_AOD, rand_coup_AOA, rand_coup_ZOD, rand_coup_ZOA, pathDelays, pathPowers, user_speed, mu_desired_AoD, mu_desired_AoA, mu_desired_ZoD, mu_desired_ZoA] =  CDL_generation(para, very_first, Initial_Phase_for_all_rays, XPR, Rx_patterns, Tx_patterns, rhat_rx, wl, a_MS, a_BS, Num_ray, nTap, rand_coup_AOD, rand_coup_AOA, rand_coup_ZOD, rand_coup_ZOA, pathDelays, pathPowers, mu_desired_AoD, mu_desired_AoA, mu_desired_ZoD, mu_desired_ZoA, d_2d)
    % longterm info: Initial_Phase_for_all_rays, Rx_patterns, Tx_patterns, rhat_rx, vbar, wl, a_MS, a_BS, Num_ray
    % This code stems from my previous project: G. Park et al., "5G K-Simulator: Link Level Simulator for 5G," 2018 IEEE DySPAN, 2018, pp. 1-2, doi: 10.1109/DySPAN.2018.8610463.
    % CDL channel in TR 38.901. 7.7.1
    if very_first == true  
        %% parameters
        Nt = para.num_Tx_antenna;
        Nr = para.num_Rx_antenna;
        channel_type = para.Channel.type;
        user_speed = para.user_speed;
        theta_v = para.theta_v;
        phi_v = para.phi_v;
        %t = para.t;
        if Num_ray == -1
            Num_ray = 20;
        else
            ; % use custom num_ray
        end
        
        freq = para.fc;
        DS = para.DS;     % Wanted delay spread in ns;
        % subject to the frequency f and the scenario type
        % sceType (TR 38.901 Table 7.7.3-1~2)
        % 100ns for calibration
        j = sqrt(-1);
        % Antenna configuration
        % Transmit array type (ULA or URA)
        TxArrayType = para.TxArrayType; % for calibration
        % Receive array type (ULA or URA)
        RxArrayType = para.RxArrayType; % for calibration
        % Transmit antenna spacing in wavelengths (0.1-100)
        TxAnt = para.Tx_d_lambda;
        % Receive antenna spacing in wavelengths (0.1-100)
        RxAnt = para.Rx_d_lambda; % smaller spacing, resolvability decreases -> makes effectively fewer paths. check this!
        % Number of transmit antenna elements per row for URA i.e. # of columns
        % N in BS; M * Wt * P = Nt in the input
        Wt = para.N1; % For calibration
        % Number of receive antenna elements per row for URA i.e. # of columns
        % N in UE; M * Wr * P = Nr in the input
        Wr = para.M1;
        % ratio AS_desired/AS_model parameter
        AS_ratio = para.AS_ratio;
        %% CDL Channel Type and related parameters
        % TR 38.901 Table 7.5-3
        offset_ang = [0.0447, -0.0447, 0.1413, -0.1413, 0.2492, -0.2492, 0.3715, -0.3715, 0.5129, -0.5129, ...
            0.6797, -0.6797, 0.8844, -0.8844, 1.1481, -1.1481, 1.5195, -1.5195, 2.1551, -2.1551]';
        % CDL channel type 
        % TR 38.901 Table 7.7.1-1~5
        CDL = para.CDL;
        
        if nTap == -1
            nTap = size(CDL, 1); % # of cluster
        else
            ; %use custom nTap
        end
        pathDelays = CDL(:,1)*DS;
        pathPowers_dB = CDL(:,2);
        pathPowers = 10.^( pathPowers_dB/10);
        
        for actual_taps_to_save = 1:para.L
            if para.T_actual_sampling * actual_taps_to_save > pathDelays(end)
                break
            else
                ;
            end
        end
        %disp(actual_taps_to_save)
        
        % R1-1700144
        if strcmp(channel_type,'CDL_A') % R1-1700144
            mu_model_AoD = -2.05;
            mu_model_AoA = -164.4;
            mu_model_ZoD = 97.27;
            mu_model_ZoA = 87.68;
        elseif strcmp(channel_type,'CDL_B') % R1-1700144
            mu_model_AoD = -4.96;
            mu_model_AoA = 176.6;
            mu_model_ZoD = 105.64;
            mu_model_ZoA = 71.76;
        elseif strcmp(channel_type,'CDL_C') % calculated by using R1-1700144
            mu_model_AoD = -19.6808;
            mu_model_AoA = 149.8943;
            mu_model_ZoD = 99.3310;
            mu_model_ZoA = 73.4974;
        elseif strcmp(channel_type,'CDL_D') % calculated by using R1-1700144
            mu_model_AoD = 1.9930;
            mu_model_AoA = 178.1941;
            mu_model_ZoD = 98.0897;
            mu_model_ZoA = 81.5213;
        elseif strcmp(channel_type,'CDL_E') % calculated by using R1-1700144
            mu_model_AoD = 2.8396;
            mu_model_AoA = 179.2480;
            mu_model_ZoD = 99.7957;
            mu_model_ZoA = 80.5322;
        end
                
       
        % mu_desired angle
        % Default: For calibration in R1-1701823
        if isnan(mu_desired_AoD)
            mu_desired_AoD = 120*(rand(1,1) - 0.5);
            mu_desired_AoA = 360*(rand(1,1) - 0.5);
            mu_desired_ZoD = 45*(rand(1,1) + 2);
            mu_desired_ZoA = 45*(rand(1,1) + 1);
        else
            ;
        end
       

        

        AODs_temp = CDL(:,3);
        AOAs_temp = CDL(:,4);
        ZODs_temp = CDL(:,5);
        ZOAs_temp = CDL(:,6);

        AODs_temp = AS_ratio*(AODs_temp - mu_model_AoD) + mu_desired_AoD;
        ZODs_temp = AS_ratio*(ZODs_temp - mu_model_ZoD) + mu_desired_ZoD;
        AOAs_temp = AS_ratio*(AOAs_temp - mu_model_AoA) + mu_desired_AoA;
        ZOAs_temp = AS_ratio*(ZOAs_temp - mu_model_ZoA) + mu_desired_ZoA;
        
        

        Pi_UT_alpha = 360*rand;
        
        
        AODs = zeros(nTap*Num_ray,1);
        ZODs = zeros(nTap*Num_ray,1);
        AOAs = zeros(nTap*Num_ray,1);
        ZOAs = zeros(nTap*Num_ray,1);

        % random coupling
        if isnan(rand_coup_AOD)
            disp('generating coupling')
            rand_coup_AOD = randperm(Num_ray);
            rand_coup_ZOD = randperm(Num_ray);
            rand_coup_AOA = randperm(Num_ray);
            rand_coup_ZOA = randperm(Num_ray);
        else
            ;
        end
        
        mu_lgZSD = max(-0.5, -3.1*(d_2d/1000)+0.2 );
        ZOD_mu_offset = -10^(-1.5*log10(max(10,d_2d))+3.3);
        for ind_tap = 1: nTap
            AODs(1+Num_ray*(ind_tap-1):Num_ray*ind_tap,:) = repelem(AODs_temp(ind_tap),Num_ray).' + offset_ang(rand_coup_AOD)*para.c_ASD;
            AOAs(1+Num_ray*(ind_tap-1):Num_ray*ind_tap,:) = repelem(AOAs_temp(ind_tap),Num_ray).' + offset_ang(rand_coup_AOA)*para.c_ASA;
            ZOAs(1+Num_ray*(ind_tap-1):Num_ray*ind_tap,:) = repelem(ZOAs_temp(ind_tap),Num_ray).' + offset_ang(rand_coup_ZOA)*para.c_ZSA;
            % spatial consistency
            ZODs(1+Num_ray*(ind_tap-1):Num_ray*ind_tap,:) = repelem(ZODs_temp(ind_tap),Num_ray).' + ZOD_mu_offset + offset_ang(rand_coup_ZOD)*(3/8)*10^mu_lgZSD;
        end
        
        % Confine azimuth angles in [-180,180].
        AOAs = Conf_AzimuthAngles(AOAs);
        AODs = Conf_AzimuthAngles(AODs);
        
        % Confine zenith angles in [0,360] and map [180,360] to [180,0].
        ZOAs = Conf_ZenithAngles(ZOAs);
        ZODs = Conf_ZenithAngles(ZODs);
        
        % For LOS environment, adjust subpath AoDs and AoAs such that the AoD
        % and AoA of the LOS component are aligned in line
        if strcmp(channel_type,'CDL_D') || strcmp(channel_type,'CDL_E')
            % Calculate the correct azimuth AoA for LOS case, which differ from azimuth AoD by 180 degrees
            % Channel_Info(1,4) denotes the azimuth AoD for LOS component
            correctAzAOA = 180; % AoA = 0 for both of CDL_D and CDL_E.
            % Calculate the difference between the generated azimuth AoA and the correct azimuth AoA
            AzAOA = AOAs(1) - correctAzAOA;
            AOAs = AOAs - AzAOA;
            azAOA_temp = AOAs;
            azAOA_temp(azAOA_temp < 0) = azAOA_temp(azAOA_temp < 0) + 360;
            AOAs = azAOA_temp;
            
            % Calculate the correct elevation AoA for LOS component, which is the additive inverse of the corresponding elevation AoD
            correctElAOA = -ZODs(1);
            % Calculate the difference between the generated elevation AoA and the correct elevation AoA
            ElAOA =  ZOAs(1) - correctElAOA;
            ZOAs = ZOAs - ElAOA;
        end
        
        rhat_rx = sph_to_car(AOAs,ZOAs);
        % TR 38.901 Equation 7.5-22/7.5-28/7.5-29, exponentional term which is a
        % function of the Doppler due to user velocity
        vbar = user_speed * sph_to_car(phi_v,theta_v);    % UT velocity vector. (eq. 7.5-25)
        
        wl = para.light_speed/freq; % m
        
        n_BS = (1:1:para.N1*para.N2)';
        n_MS = (1:1:para.M1*para.M2)';
        % LCS of AoDs and ZoDs with BS downtilt
        [ZoDs_LCS,AoDs_LCS] = GCS_to_LCS(0,para.Tx_downtilt,0,ZODs,AODs);
        [ZoAs_LCS,AoAs_LCS] = GCS_to_LCS(Pi_UT_alpha,0,0,ZOAs,AOAs);
        
        a_BS = zeros(para.N1*para.N2, length(AODs));
        a_MS = zeros(para.M1*para.M2, length(AOAs));
        
        switch TxArrayType
            case 'ULA'
                for i = 1:para.N1*para.N2
                    for k = 1:length(AODs)
                        a_BS(i,k) = exp(j.*(n_BS(i)-1).*2.*pi.*TxAnt.*sind(AoDs_LCS(k)));
                    end
                end
            case 'URA'
                for i = 1:para.N1*para.N2
                    for k = 1:length(AODs)
                        a_BS(i,k) = exp(j.*2.*pi.*TxAnt.*(mod(i-1,Wt).*cosd(ZoDs_LCS(k))+...
                            fix((n_BS(i)-1)/Wt).*sind(ZoDs_LCS(k)).*sind(AoDs_LCS(k))));
                    end
                end
        end
        
        switch RxArrayType
            case 'ULA'
                for i = 1:para.M1*para.M2
                    for k = 1:length(AOAs)
                        a_MS(i,k) = exp(j.*(n_MS(i)-1).*2.*pi.*RxAnt.*sind(AoAs_LCS(k)));
                    end
                end
            case 'URA'
                for i = 1:para.M1*para.M2
                    for k = 1:length(AOAs)
                        a_MS(i,k) = exp(j.*2.*pi.*RxAnt.*(mod(i-1,Wr).*cosd(ZoAs_LCS(k))+...
                            fix((n_MS(i)-1)/Wr).*sind(ZoAs_LCS(k)).*sind(AoAs_LCS(k))));
                    end
                end
        end
        
        switch para.Tx_pattern_type
            case 'Omni-directional'
                Tx_patterns = ones(1,Num_ray*nTap);
            case 'Pattern'
                Tx_patterns = BS_antenna_patterns(ZoDs_LCS,AoDs_LCS);
        end
        
        switch para.Rx_pattern_type
            case 'Omni-directional'
                Rx_patterns = ones(1,Num_ray*nTap);
            case 'Pattern'
                Rx_patterns = UE_antenna_patterns(ZoAs_LCS,AoAs_LCS);
        end
        
        % TR 38.901 7.7.1. step 3
        XPR = 10^(para.XPR_dB/10);
        
        % TR 38.901 7.7.1. step 4 (7.5. step 10, 11)
        t = para.CurrentTime; % should be in the form of m/W, but this is automatically done via OFDM symbols indices....
        dopplerTerm = exp(kron(t,(1i*2*pi*rhat_rx*vbar'/wl)));
        % sinc.. baseband equiv. channel
        W = 1/para.T_actual_sampling; 
        if isnan(Initial_Phase_for_all_rays)
            Initial_Phase_for_all_rays = zeros(nTap, Num_ray, para.Tx_pol, para.Rx_pol);
        else
            ;
        end
        % this part generates a term at current time (e.g., m/W)
        H = zeros(Nr, Nt, nTap);
        for ind_tap = 1:nTap
            b = 1;
            for ind_ray = 1:Num_ray
                % only generate this for the very first time
                Initial_Phase =  2*pi*(rand(para.Tx_pol,para.Rx_pol)-1/2);
                Initial_Phase_for_all_rays(ind_tap, ind_ray, :, :) = Initial_Phase;
                if para.Tx_pol == 1 && para.Rx_pol == 1
                    H(:, :, ind_tap) =  H(:, :, ind_tap)+  [sqrt(Rx_patterns(ind_ray + Num_ray*(ind_tap-1)))*cosd(0)]*...
                            [  exp(j*Initial_Phase(1))   ]...
                            *[sqrt(Tx_patterns(ind_ray + Num_ray*(ind_tap-1)))*cosd(0)  ].'*dopplerTerm(ind_ray+Num_ray*(ind_tap-1),b)*a_MS(:,ind_ray + Num_ray*(ind_tap-1))*a_BS(:,ind_ray + Num_ray*(ind_tap-1))';
                elseif para.Tx_pol == 2 && para.Rx_pol == 1
                    H1 =  [sqrt(Rx_patterns(ind_ray + Num_ray*(ind_tap-1)))*cosd(0) ]*...
                        [  exp(j*Initial_Phase(1)) , ...
                        sqrt(1/XPR)*exp(j*Initial_Phase(2))    ]...
                        *[sqrt(Tx_patterns(ind_ray + Num_ray*(ind_tap-1)))*cosd(para.pol_slant_angle(1)) , sqrt(Tx_patterns(ind_ray + Num_ray*(ind_tap-1)))*sind(para.pol_slant_angle(1)) ].'*dopplerTerm(ind_ray+Num_ray*(ind_tap-1),b)*a_MS(:,ind_ray + Num_ray*(ind_tap-1))*a_BS(:,ind_ray + Num_ray*(ind_tap-1))';

                    H3 =   [sqrt(Rx_patterns(ind_ray + Num_ray*(ind_tap-1)))*cosd(0) ]*...
                        [  exp(j*Initial_Phase(1)), ...
                        sqrt(1/XPR)*exp(j*Initial_Phase(2))    ]...
                        *[sqrt(Tx_patterns(ind_ray + Num_ray*(ind_tap-1)))*cosd(para.pol_slant_angle(2)) , sqrt(Tx_patterns(ind_ray + Num_ray*(ind_tap-1)))*sind(para.pol_slant_angle(2)) ].'*dopplerTerm(ind_ray+Num_ray*(ind_tap-1),b)*a_MS(:,ind_ray + Num_ray*(ind_tap-1))*a_BS(:,ind_ray + Num_ray*(ind_tap-1))';

                    H(:, :, ind_tap) = H(:, :, ind_tap) + [H1 , H3 ];
                elseif para.Tx_pol == 2 && para.Rx_pol == 2
                    H1 =  [sqrt(Rx_patterns(ind_ray + Num_ray*(ind_tap-1)))*cosd(para.pol_slant_angle(1)),  sqrt(Rx_patterns(ind_ray + Num_ray*(ind_tap-1)))*sind(para. pol_slant_angle(1))]*...
                        [  exp(j*Initial_Phase(1)) ,  sqrt(1/XPR)*exp(j*Initial_Phase(2)); ...
                        sqrt(1/XPR)*exp(j*Initial_Phase(3)) , exp(j*Initial_Phase(4))    ]...
                        *[sqrt(Tx_patterns(ind_ray + Num_ray*(ind_tap-1)))*cosd(para.pol_slant_angle(1)) , sqrt(Tx_patterns(ind_ray + Num_ray*(ind_tap-1)))*sind(para.pol_slant_angle(1)) ].'*dopplerTerm(ind_ray+Num_ray*(ind_tap-1),b)*a_MS(:,ind_ray + Num_ray*(ind_tap-1))*a_BS(:,ind_ray + Num_ray*(ind_tap-1))';
                
                    H2 =  [sqrt(Rx_patterns(ind_ray + Num_ray*(ind_tap-1)))*cosd(para.pol_slant_angle(2)),  sqrt(Rx_patterns(ind_ray + Num_ray*(ind_tap-1)))*sind(para.pol_slant_angle(2))]*...
                        [  exp(j*Initial_Phase(1)) ,  sqrt(1/XPR)*exp(j*Initial_Phase(2));...
                        sqrt(1/XPR)*exp(j*Initial_Phase(3)) , exp(j*Initial_Phase(4))  ]...
                        *[sqrt(Tx_patterns(ind_ray + Num_ray*(ind_tap-1)))*cosd(para.pol_slant_angle(1)) , sqrt(Tx_patterns(ind_ray + Num_ray*(ind_tap-1)))*sind(para.pol_slant_angle(1)) ].'*dopplerTerm(ind_ray+Num_ray*(ind_tap-1),b)*a_MS(:,ind_ray + Num_ray*(ind_tap-1))*a_BS(:,ind_ray + Num_ray*(ind_tap-1))';
                    
                    H3 =   [sqrt(Rx_patterns(ind_ray + Num_ray*(ind_tap-1)))*cosd(para.pol_slant_angle(1)),  sqrt(Rx_patterns(ind_ray + Num_ray*(ind_tap-1)))*sind(para.pol_slant_angle(1))]*...
                        [  exp(j*Initial_Phase(1)) ,  sqrt(1/XPR)*exp(j*Initial_Phase(2)); ...
                        sqrt(1/XPR)*exp(j*Initial_Phase(3)) , exp(j*Initial_Phase(4))    ]...
                        *[sqrt(Tx_patterns(ind_ray + Num_ray*(ind_tap-1)))*cosd(para.pol_slant_angle(2)) , sqrt(Tx_patterns(ind_ray + Num_ray*(ind_tap-1)))*sind(para.pol_slant_angle(2)) ].'*dopplerTerm(ind_ray+Num_ray*(ind_tap-1),b)*a_MS(:,ind_ray + Num_ray*(ind_tap-1))*a_BS(:,ind_ray + Num_ray*(ind_tap-1))';
                    
                    H4 =  [sqrt(Rx_patterns(ind_ray + Num_ray*(ind_tap-1)))*cosd(para.pol_slant_angle(2)),  sqrt(Rx_patterns(ind_ray + Num_ray*(ind_tap-1)))*sind(para.pol_slant_angle(2))]*...
                        [  exp(j*Initial_Phase(1)) ,  sqrt(1/XPR)*exp(j*Initial_Phase(2)); ...
                        sqrt(1/XPR)*exp(j*Initial_Phase(3)) , exp(j*Initial_Phase(4))  ]...
                        *[sqrt(Tx_patterns(ind_ray + Num_ray*(ind_tap-1)))*cosd(para.pol_slant_angle(2)) , sqrt(Tx_patterns(ind_ray + Num_ray*(ind_tap-1)))*sind(para.pol_slant_angle(2))].'*dopplerTerm(ind_ray+Num_ray*(ind_tap-1),b)*a_MS(:,ind_ray + Num_ray*(ind_tap-1))*a_BS(:,ind_ray + Num_ray*(ind_tap-1))';
                    H(:, :, ind_tap) = H(:, :, ind_tap) + [H1 , H3 ; H2 , H4];
                else
                    adsfasdf
                end
            end
            H(:, :, ind_tap) = sqrt(pathPowers(ind_tap)/Num_ray)*H(:, :, ind_tap);
        end
        L = para.L;
        H_actual_tap = zeros(Nr, Nt, L); % h_l
        for ind_actual_tap = 1:L
            for ind_cluster = 1:nTap
                a = H(:,:,ind_cluster); % a is function of current time! TODO % time is in the form of m/W
                curr_cluster_delay = pathDelays(ind_cluster);
                a_b_curr_ray = a;
                H_actual_tap(:, :, ind_actual_tap) = H_actual_tap(:, :, ind_actual_tap)  + a_b_curr_ray * sinc((ind_actual_tap-1)-curr_cluster_delay*W); % (2.34)
            end
        end
    else
        % load longterm info
        j = sqrt(-1);
        Nt = para.num_Tx_antenna;
        Nr = para.num_Rx_antenna;
        user_speed = para.user_speed;
        theta_v = para.theta_v;
        phi_v = para.phi_v;
        t = para.CurrentTime; % if curr_time is same, we shouldd get the same result! debug with this!!
        DS = para.DS;     % Wanted delay spread in ns;
        W = 1/para.T_actual_sampling;
        vbar = user_speed * sph_to_car(phi_v,theta_v);    % UT velocity vector. (eq. 7.5-25)
        dopplerTerm = exp(kron(t,(1i*2*pi*rhat_rx*vbar'/wl)));
        H = zeros(Nr, Nt, nTap);
        for ind_tap = 1:nTap
            b = 1;
            for ind_ray = 1:Num_ray
                Initial_Phase = Initial_Phase_for_all_rays(ind_tap, ind_ray, :, :);
                if para.Tx_pol == 1 && para.Rx_pol == 1
                    H(:, :, ind_tap) =  H(:, :, ind_tap)+  [sqrt(Rx_patterns(ind_ray + Num_ray*(ind_tap-1)))*cosd(0)]*...
                            [  exp(j*Initial_Phase(1))   ]...
                            *[sqrt(Tx_patterns(ind_ray + Num_ray*(ind_tap-1)))*cosd(0)  ].'*dopplerTerm(ind_ray+Num_ray*(ind_tap-1),b)*a_MS(:,ind_ray + Num_ray*(ind_tap-1))*a_BS(:,ind_ray + Num_ray*(ind_tap-1))';
                elseif para.Tx_pol == 2 && para.Rx_pol == 1
                    H1 =  [sqrt(Rx_patterns(ind_ray + Num_ray*(ind_tap-1)))*cosd(0) ]*...
                        [  exp(j*Initial_Phase(1)) , ...
                        sqrt(1/XPR)*exp(j*Initial_Phase(2))    ]...
                        *[sqrt(Tx_patterns(ind_ray + Num_ray*(ind_tap-1)))*cosd(para.pol_slant_angle(1)) , sqrt(Tx_patterns(ind_ray + Num_ray*(ind_tap-1)))*sind(para.pol_slant_angle(1)) ].'*dopplerTerm(ind_ray+Num_ray*(ind_tap-1),b)*a_MS(:,ind_ray + Num_ray*(ind_tap-1))*a_BS(:,ind_ray + Num_ray*(ind_tap-1))';

                    H3 =   [sqrt(Rx_patterns(ind_ray + Num_ray*(ind_tap-1)))*cosd(0) ]*...
                        [  exp(j*Initial_Phase(1)), ...
                        sqrt(1/XPR)*exp(j*Initial_Phase(2))    ]...
                        *[sqrt(Tx_patterns(ind_ray + Num_ray*(ind_tap-1)))*cosd(para.pol_slant_angle(2)) , sqrt(Tx_patterns(ind_ray + Num_ray*(ind_tap-1)))*sind(para.pol_slant_angle(2)) ].'*dopplerTerm(ind_ray+Num_ray*(ind_tap-1),b)*a_MS(:,ind_ray + Num_ray*(ind_tap-1))*a_BS(:,ind_ray + Num_ray*(ind_tap-1))';

                    H(:, :, ind_tap) = H(:, :, ind_tap) + [H1 , H3 ];
                
                elseif para.Tx_pol == 2 && para.Rx_pol == 2
                    H1 =  [sqrt(Rx_patterns(ind_ray + Num_ray*(ind_tap-1)))*cosd(para.pol_slant_angle(1)),  sqrt(Rx_patterns(ind_ray + Num_ray*(ind_tap-1)))*sind(para. pol_slant_angle(1))]*...
                        [  exp(j*Initial_Phase(1)) ,  sqrt(1/XPR)*exp(j*Initial_Phase(2)); ...
                        sqrt(1/XPR)*exp(j*Initial_Phase(3)) , exp(j*Initial_Phase(4))    ]...
                        *[sqrt(Tx_patterns(ind_ray + Num_ray*(ind_tap-1)))*cosd(para.pol_slant_angle(1)) , sqrt(Tx_patterns(ind_ray + Num_ray*(ind_tap-1)))*sind(para.pol_slant_angle(1)) ].'*dopplerTerm(ind_ray+Num_ray*(ind_tap-1),b)*a_MS(:,ind_ray + Num_ray*(ind_tap-1))*a_BS(:,ind_ray + Num_ray*(ind_tap-1))';
                
                    H2 =  [sqrt(Rx_patterns(ind_ray + Num_ray*(ind_tap-1)))*cosd(para.pol_slant_angle(2)),  sqrt(Rx_patterns(ind_ray + Num_ray*(ind_tap-1)))*sind(para.pol_slant_angle(2))]*...
                        [  exp(j*Initial_Phase(1)) ,  sqrt(1/XPR)*exp(j*Initial_Phase(2));...
                        sqrt(1/XPR)*exp(j*Initial_Phase(3)) , exp(j*Initial_Phase(4))  ]...
                        *[sqrt(Tx_patterns(ind_ray + Num_ray*(ind_tap-1)))*cosd(para.pol_slant_angle(1)) , sqrt(Tx_patterns(ind_ray + Num_ray*(ind_tap-1)))*sind(para.pol_slant_angle(1)) ].'*dopplerTerm(ind_ray+Num_ray*(ind_tap-1),b)*a_MS(:,ind_ray + Num_ray*(ind_tap-1))*a_BS(:,ind_ray + Num_ray*(ind_tap-1))';
                    
                    H3 =   [sqrt(Rx_patterns(ind_ray + Num_ray*(ind_tap-1)))*cosd(para.pol_slant_angle(1)),  sqrt(Rx_patterns(ind_ray + Num_ray*(ind_tap-1)))*sind(para.pol_slant_angle(1))]*...
                        [  exp(j*Initial_Phase(1)) ,  sqrt(1/XPR)*exp(j*Initial_Phase(2)); ...
                        sqrt(1/XPR)*exp(j*Initial_Phase(3)) , exp(j*Initial_Phase(4))    ]...
                        *[sqrt(Tx_patterns(ind_ray + Num_ray*(ind_tap-1)))*cosd(para.pol_slant_angle(2)) , sqrt(Tx_patterns(ind_ray + Num_ray*(ind_tap-1)))*sind(para.pol_slant_angle(2)) ].'*dopplerTerm(ind_ray+Num_ray*(ind_tap-1),b)*a_MS(:,ind_ray + Num_ray*(ind_tap-1))*a_BS(:,ind_ray + Num_ray*(ind_tap-1))';
                    
                    H4 =  [sqrt(Rx_patterns(ind_ray + Num_ray*(ind_tap-1)))*cosd(para.pol_slant_angle(2)),  sqrt(Rx_patterns(ind_ray + Num_ray*(ind_tap-1)))*sind(para.pol_slant_angle(2))]*...
                        [  exp(j*Initial_Phase(1)) ,  sqrt(1/XPR)*exp(j*Initial_Phase(2)); ...
                        sqrt(1/XPR)*exp(j*Initial_Phase(3)) , exp(j*Initial_Phase(4))  ]...
                        *[sqrt(Tx_patterns(ind_ray + Num_ray*(ind_tap-1)))*cosd(para.pol_slant_angle(2)) , sqrt(Tx_patterns(ind_ray + Num_ray*(ind_tap-1)))*sind(para.pol_slant_angle(2))].'*dopplerTerm(ind_ray+Num_ray*(ind_tap-1),b)*a_MS(:,ind_ray + Num_ray*(ind_tap-1))*a_BS(:,ind_ray + Num_ray*(ind_tap-1))';
                    H(:, :, ind_tap) = H(:, :, ind_tap) + [H1 , H3 ; H2 , H4];
                else
                    sfdasdf
                end
            end
            H(:, :, ind_tap) = sqrt(pathPowers(ind_tap)/Num_ray)*H(:, :, ind_tap);
        end

        L = para.L;%para.num_cp(2);% use shorter cp..
        H_actual_tap = zeros(Nr, Nt, L); % h_l
        for ind_actual_tap = 1:L
            for ind_cluster = 1:nTap
                a = H(:,:,ind_cluster); % a is function of current time! TODO % time is in the form of m/W
                curr_cluster_delay = pathDelays(ind_cluster);
                a_b_curr_ray = a;
                % summation over all possible paths w.r.t. current tap
                H_actual_tap(:, :, ind_actual_tap) = H_actual_tap(:, :, ind_actual_tap)  + a_b_curr_ray * sinc((ind_actual_tap-1)-curr_cluster_delay*W); % (2.34)
            end
        end

    end
    
    function phi = Conf_AzimuthAngles(phi)  % Azimuth angle
        phi =  (mod(phi + 180,360) - 180);
    end

    function theta = Conf_ZenithAngles(theta)   % Zenith angle
        theta = mod(theta,360);
        theta(theta>180) = 360 - theta(theta>180);
    end

    function out = sph_to_car(phi,theta)    % Spherical to Cartesian
        sintheta = sind(theta);
        x = sintheta.*cosd(phi);
        y = sintheta.*sind(phi);
        z = cosd(theta);
        out = [x, y, z];
    end

    function A_pattern = BS_antenna_patterns(ZoDs,AoDs) % BS antenna pattern based on TR 38.802, 36.873
        SLA_V = 30;
        theta_3dB = 65;
        phi_3dB = 65;
        A_max = 30;
        G_max = 8;
        
        A_dB_theta = -min(12*((ZoDs - 90)/theta_3dB).^2, SLA_V);
        A_dB_phi = - min(12*(AoDs/phi_3dB).^2,A_max);
        
        A_dB = -min(-(A_dB_theta+A_dB_phi),A_max);
        A_pattern = 10.^((G_max+A_dB)/10);
    end

    function A_pattern = UE_antenna_patterns(ZoAs,AoAs) % UE antenna pattern based on TR 38.802, 36.873
        SLA_V = 25;
        theta_3dB = 90;
        phi_3dB = 90;
        A_max = 25;
        G_max = 5;
        
        A_dB_theta = -min(12*((ZoAs - 90)/theta_3dB).^2, SLA_V);
        A_dB_phi = - min(12*(AoAs/phi_3dB).^2,A_max);
        
        A_dB = -min(-(A_dB_theta+A_dB_phi),A_max);
        A_pattern = sqrt(10^(G_max/10)*10.^(A_dB/10));
    end

    function [theta_LCS, phi_LCS] = GCS_to_LCS(alpha, beta, gamma, theta_GCS , phi_GCS) % GCS to LCS. 38.802, 7.1.
        theta_LCS = acosd(cosd(beta).*cosd(gamma).*cosd(theta_GCS) + (sind(beta)*cosd(gamma).*cosd(phi_GCS - alpha) - sind(gamma).*sind(phi_GCS-alpha)).*sind(theta_GCS));
        phi_LCS = angle((cosd(beta).*sind(theta_GCS).*cosd(phi_GCS-alpha) - sind(beta).*cosd(theta_GCS))+ 1i*(cosd(beta).*sind(gamma).*cosd(theta_GCS) + (sind(beta).*sind(gamma).*cosd(phi_GCS-alpha) + cosd(gamma).*sind(phi_GCS-alpha)).*sind(theta_GCS)));
        phi_LCS = phi_LCS/pi*180;
    end

    function a_b_curr_ray = BB_equiv(a, fc, curr_cluster_delay)
        a_b_curr_ray = a;
    end
end