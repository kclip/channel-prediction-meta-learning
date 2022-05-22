import torch

saved_path_channels = {}

curr_common_key = 'final_vector_channel_'
curr_common_path = '/3gpp_channel_generated_data/'
for num_antennas in [1,2,4,8,16,32,64,128]:
    rx_pol = 1
    if num_antennas == 1:
        tx_pol = 1
        num_tx_hor = 1
        num_tx_ver = 1
        num_rx_hor = 1
        num_rx_ver = 1
    elif num_antennas == 2: ##
        tx_pol = 1
        num_tx_hor = 2
        num_tx_ver = 1
        num_rx_hor = 1
        num_rx_ver = 1
    elif num_antennas == 4:
        tx_pol = 1
        num_tx_hor = 2
        num_tx_ver = 2
        num_rx_hor = 1
        num_rx_ver = 1
    elif num_antennas == 8: ##
        tx_pol = 1
        num_tx_hor = 2
        num_tx_ver = 2
        num_rx_hor = 2
        num_rx_ver = 1
    elif num_antennas == 16:
        tx_pol = 2
        num_tx_hor = 2
        num_tx_ver = 2
        num_rx_hor = 2
        num_rx_ver = 1
    elif num_antennas == 32:
        tx_pol = 2
        num_tx_hor = 4
        num_tx_ver = 2
        num_rx_hor = 2
        num_rx_ver = 1
    elif num_antennas == 64: ##
        tx_pol = 2
        num_tx_hor = 4
        num_tx_ver = 4
        num_rx_hor = 2
        num_rx_ver = 1
        # for self-rank det. exp
        #tx_pol = 1
        #num_tx_hor = 4
        #num_tx_ver = 2
        #num_rx_hor = 4
        #num_rx_ver = 2
    elif num_antennas == 128:
        tx_pol = 2
        num_tx_hor = 4
        num_tx_ver = 4
        num_rx_hor = 2
        num_rx_ver = 2
    else:
        raise NotImplementedError
    num_tot_tx_antennas = num_tx_hor*num_tx_ver*tx_pol
    num_tot_rx_antennas = num_rx_hor*num_rx_ver*rx_pol 
    assert num_tot_tx_antennas*num_tot_rx_antennas == num_antennas
    for DS_ratio in [0,0.5,1,2,3,4,5,6,7,8,9]:
        for num_clusters in [1,2,3,4,5,10,15,19,20]:
            for num_taps in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]:
                for vel_mul in [0,0.01,0.1,0.5,1,2,10,20,40]:# #[0,0.1, 0.5, 1, 2, 10, 20]:
                    saved_path_channels[curr_common_key + 'num_antennas_' + str(num_antennas) + 'num_clusters_' + str(num_clusters) + 'num_taps_' + str(num_taps) + 'vel_mul_' + str(vel_mul) + 'DS_ratio_' + str(DS_ratio)] = curr_common_path + 'cluster_' + str(num_clusters) + '/num_taps_to_save_' + str(num_taps) + '/num_Tx_antennas_' +  str(num_tot_tx_antennas) + 'num_tx_hor_' + str(num_tx_hor) +  'num_tx_ver' + str(num_tx_ver) + '_num_Rx_antennas_' + str(num_tot_rx_antennas) + 'num_rx_hor' + str(num_rx_hor) + 'num_rx_ver' + str(num_rx_ver)  + 'tx_pol' + str(tx_pol)  + 'rx_pol' + str(rx_pol) + 'val_mul' + str(vel_mul) + 'DS_ratio' + str(DS_ratio) + '.mat'
