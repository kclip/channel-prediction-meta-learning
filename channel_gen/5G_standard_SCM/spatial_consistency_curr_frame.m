function [S, DS]  = spatial_consistency_curr_frame(total_num_clusters, fc, d_2d)
    S = zeros(total_num_clusters, 6);
    %% NLOS UMI-street canyon
    
    %% delay
    mu_lgDS = -0.24*log10(1+fc) - 6.83;
    sigma_lgDS =  0.16*log10(1+fc) +0.28;
    
    %% angles
    mu_lgASA = -0.08*log10(1+fc) + 1.81;
    sigma_lgASA = 0.05*log10(1+fc) + 0.3;

    
    mu_lgASD = -0.23*log10(1+fc) + 1.53;
    sigma_lgASD = 0.11*log10(1+fc) + 0.33;

    
    mu_lgZSA = -0.04*log10(1+fc) + 0.92;
    sigma_lgZSA = -0.07*log10(1+fc) + 0.41;

    
    mu_lgZSD = max(-0.5, -3.1*(d_2d/1000)+0.2 ); % ue height < BS height
    sigma_lgZSD = 0.35;

    
    
    %% generation of LSP (correlations accounted) % 3.3.1, IST-4-027756 WINNER II D1.1.2 V1.2 WINNER II Channel Models
    C_MM = [sigma_lgDS^2, 0, 0.4*sigma_lgDS*sigma_lgASA, -0.5*sigma_lgDS*sigma_lgZSD, 0; ...
        0, sigma_lgASD^2, 0, 0.5*sigma_lgASD*sigma_lgZSD, 0.5*sigma_lgASD*sigma_lgZSA; ...
        0.4*sigma_lgASA*sigma_lgDS, 0, sigma_lgASA^2, 0, 0.2*sigma_lgASA*sigma_lgZSA;
        -0.5*sigma_lgZSD*sigma_lgDS, 0.5*sigma_lgZSD*sigma_lgASD, 0, sigma_lgZSD^2, 0;
        0, 0.5*sigma_lgZSA*sigma_lgASD, 0.2*sigma_lgZSA*sigma_lgASA, 0, sigma_lgZSA^2];
    
    sqrt_C_MM = chol(C_MM, 'lower');
    mu_MM = [mu_lgDS, mu_lgASD, mu_lgASA, mu_lgZSD, mu_lgZSA].';
    
    
    DS = 0;
    ASD = 0;
    ASA = 0;
    ZSD = 0;
    ZSA = 0;
    
    while 1
        if (DS< 45e-9) | (ASD < 1) | (ASA < 1) | (ZSD < 1) | (ZSA < 1)
            gaussian_rvs = normrnd(0, 1, 5,1);
            s_tilde = mu_MM + sqrt_C_MM * gaussian_rvs;
            s_actual = 10.^s_tilde;
            DS = s_actual(1);
            ASD = min(s_actual(2), 104);
            ASA = min(s_actual(3), 104);
            ZSD = min(s_actual(4), 52);
            ZSA = min(s_actual(5), 52);
        else
            break
        end
    end
    
    
    
    tmp = zeros(total_num_clusters,1);
    for ind_cluster = 1:total_num_clusters
        tmp(ind_cluster,1) = rand(1,1)*2*10^(mu_lgDS + sigma_lgDS) / DS;
    end
    min_of_tmp = min(tmp);
    tmp = tmp - min_of_tmp;
    tmp = sort(tmp);

    
    power_tmp = zeros(total_num_clusters,1);
    xi = 3;
    for ind_cluster = 1:total_num_clusters
        % path delay
        S(ind_cluster, 1) = tmp(ind_cluster);
        % AOD
        phi_AOD = (1-2*rand(1,1))*2*10^(mu_lgASD+sigma_lgASD);
        S(ind_cluster, 3) = phi_AOD;
        % AOA
        phi_AOA = (1-2*rand(1,1))*2*10^(mu_lgASA+sigma_lgASA);
        S(ind_cluster, 4) = phi_AOA;
        % ZOD
        phi_ZOD = (1-2*rand(1,1))*2*10^(mu_lgZSD+sigma_lgZSD);
        S(ind_cluster, 5) = phi_ZOD;
        % ZOA
        phi_ZOA = (1-2*rand(1,1))*2*10^(mu_lgZSA+sigma_lgZSA);
        S(ind_cluster, 6) = phi_ZOA;
        % PDP
        curr_tau = tmp(ind_cluster) +min_of_tmp;
        z_n = normrnd(0,xi);
        tmp_1 = exp(-curr_tau);
        tmp_2 = exp(-sqrt(2)*abs(phi_AOD)/ASD);
        tmp_3 = exp(-sqrt(2)*abs(phi_AOA)/ASA);
        tmp_4 = exp(-sqrt(2)*abs(phi_ZOD)/ZSD);
        tmp_5 = exp(-sqrt(2)*abs(phi_ZOA)/ZSA);
        tmp_6 = 10^(-z_n/10);
        power_tmp(ind_cluster,1) =tmp_1 * tmp_2 * tmp_3 * tmp_4 * tmp_5 * tmp_6;        
    end

    
    sum_power = sum(power_tmp);
    for ind_cluster = 1:total_num_clusters
        S(ind_cluster, 2) = 10*log10(power_tmp(ind_cluster)/sum_power);
    end
    
    
    
    