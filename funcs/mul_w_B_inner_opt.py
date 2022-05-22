import torch
import numpy as np
import copy
import warnings
import scipy
from numpy.linalg import inv

MAX_ITER = 50
MAX_ITER_FOR_B = 10
CONV_CRITERIA = 1e-4

def inner_optimization_ALS(X_te_list, Y_te_list, if_get_outer_loss, B_bar, w_bar, X_list, Y_list, curr_lambda, rank_for_regressor,lag = None, pure_X_tr = None, pure_Y_tr = None,  prev_B=None, prev_w=None, max_iter= MAX_ITER, conv_criteria=CONV_CRITERIA):
    X_list_tr = copy.deepcopy(X_list)
    Y_list_tr = copy.deepcopy(Y_list)

    K = len(w_bar)
    lambda_1 = curr_lambda[0]
    lambda_2 = curr_lambda[1]

    if prev_B is None:
        B = copy.deepcopy(B_bar)
    else:
        B = copy.deepcopy(prev_B) # from first EP
    if prev_w is None:
        w = copy.deepcopy(w_bar)
    else:
        w = copy.deepcopy(prev_w)
    # do parallel update as deep ensemble
    for k in range(K):
        curr_loss = 9999999999999
        curr_single_w = {}
        curr_single_B = {}
        curr_single_w_bar = {}
        curr_single_B_bar = {}
        curr_single_w[0] = w[k]
        curr_single_B[0] = B[k]
        curr_single_w_bar[0] = w_bar[k]
        curr_single_B_bar[0] = B_bar[k]
        for iter in range(max_iter): 
            # for each opt w and opt B -- only once since only one factor!
            curr_single_w = opt_multiple_w_gauss_seidel(curr_single_B_bar, curr_single_w_bar, X_list_tr, Y_list_tr, lambda_2, curr_single_B, curr_single_w, max_iter=1)
            curr_single_B = opt_multiple_B_gauss_seidel(curr_single_B_bar, curr_single_w_bar, X_list_tr, Y_list_tr, lambda_1, curr_single_w, rank_for_regressor, curr_single_B, max_iter=1)
            loss_tmp = inner_loss(curr_single_B_bar, curr_single_w_bar, X_list_tr, Y_list_tr, lambda_1, lambda_2, curr_single_B, curr_single_w)
            if loss_tmp<=conv_criteria:
                break 
            elif (loss_tmp <= curr_loss) and (torch.norm(curr_loss - loss_tmp) <= conv_criteria):
                break
            else:
                curr_loss = loss_tmp
        w[k] = curr_single_w[0]
        B[k] = curr_single_B[0]
        for l in range(len(Y_list_tr)):
            Y_list_tr[l] -= B[k]@Herm(B[k])@X_list_tr[l]@w[k]

    if if_get_outer_loss:
        query_loss = inner_loss(B_bar, w_bar, X_te_list, Y_te_list, 0.0, 0.0, B, w)
    else:
        query_loss = None
    return None, B, w, query_loss


def inner_loss(B_bar, w_bar, X_list, Y_list, lambda_1, lambda_2, B, w, A = None):
    K = len(w)
    loss_1 = 0
    for ind_sample in range(len(X_list)):
        X_l = X_list[ind_sample]
        y_l = Y_list[ind_sample]
        pred_l = 0
        for k in range(K):
            if A is None:
                pred_l += B[k]@Herm(B[k])@X_l@w[k]
            else:
                pred_l += A[k]@X_l@w[k]
        loss_1 += torch.norm(y_l-pred_l)**2
    loss_2 = 0
    if lambda_1 == 0:
        pass
    else:
        for k in range(K):
            if A is None:
                loss_2 += lambda_1 * torch.norm(Herm(B_bar[k])- Herm(B_bar[k])@B[k]@Herm(B[k]) )**2
            else:
                loss_2 += lambda_1 * torch.norm(Herm(B_bar[k])- Herm(B_bar[k])@A[k] )**2
    loss_3 = 0
    if lambda_2 == 0:
        pass
    else:
        for k in range(K):
            loss_3 += lambda_2 * torch.norm(w[k] - w_bar[k])**2 # it was w!!
    return loss_1 + loss_2 + loss_3

def change_to_matrix(Z_l_list):
    tmp = torch.cat(Z_l_list, dim=1)
    return Herm(tmp)

def opt_multiple_B_gauss_seidel(B_bar, w_bar, X_list, Y_list, lambda_1, w, rank_for_regressor, prev_B, max_iter=MAX_ITER_FOR_B, conv_criteria=CONV_CRITERIA):
    Y = change_to_matrix(Y_list)
    K = len(w_bar)
    # init B
    B = {}
    assert rank_for_regressor == 1
    if prev_B is None:
        raise NotImplementedError
    else:
        B = prev_B
    X_tilde = {}
    for k in range(K):
        X_tilde_list_k = [x @ w[k] for x in X_list]
        X_tilde[k] = change_to_matrix(X_tilde_list_k)
    curr_loss = 999999
    for ind_iter in range(max_iter):
        Y_check = {}
        X_check = {}
        for k in range(K):
            curr_sum_S = 0
            for k_prime in range(K):
                if k == k_prime:
                    pass
                else:
                    curr_sum_S += X_tilde[k_prime]@B[k_prime]@Herm(B[k_prime])   
            Y_check[k] = torch.cat( [Y - curr_sum_S, np.sqrt(lambda_1) * Herm(B_bar[k])], dim=0)
            X_check[k] = torch.cat( [X_tilde[k], np.sqrt(lambda_1) * Herm(B_bar[k])], dim=0)
            C = Herm(Y_check[k])@X_check[k] + Herm(X_check[k])@Y_check[k] - Herm(X_check[k])@X_check[k]
            L, Q = torch.linalg.eigh(-C) 
            B[k] = Q[:, 0:rank_for_regressor] # Gauss-Seidel
        loss_tmp = inner_loss(B_bar, w_bar, X_list, Y_list, lambda_1, 0, B, w) #lambda_2 = 0 since we are fixing w here
        if loss_tmp<=conv_criteria:
            break 
        elif (loss_tmp <= curr_loss) and (torch.norm(curr_loss - loss_tmp) <= conv_criteria):
            break
        else:
            curr_loss = loss_tmp
    return B

def generate_P_Q_R_for_opt_w(X_list, Y_list, B, K, A=None):
    R = {} # for each pair of k and k', get the value
    for k in range(K):
        for k_prime in range(K):
            R[(k, k_prime)] = 0 # considering symmetricity will reduce the computation time!
            for ind_sample in range(len(X_list)):
                X_l = X_list[ind_sample]
                if A is None:
                    R[(k, k_prime)] += Herm(X_l)@B[k]@Herm(B[k])@B[k_prime]@Herm(B[k_prime])@X_l
                else:
                    R[(k, k_prime)] += Herm(X_l)@A[k]@A[k_prime]@X_l
    P = {}
    Q = {}
    for k in range(K):
        P[k] = 0
        Q[k] = 0
        for ind_sample in range(len(X_list)):
            X_l = X_list[ind_sample]
            y_l = Y_list[ind_sample]
            if A is None:
                P[k] += Herm(X_l)@B[k]@Herm(B[k])@X_l
                Q[k] += Herm(X_l)@B[k]@Herm(B[k])@y_l
            else:
                P[k] += Herm(X_l)@A[k]@X_l
                Q[k] += Herm(X_l)@A[k]@y_l
    return R, P, Q

def opt_multiple_w_gauss_seidel(B_bar, w_bar, X_list, Y_list, lambda_2, B, prev_w, max_iter=1, conv_criteria=CONV_CRITERIA):
    K = len(w_bar) # total number of different predictors
    R, P, Q = generate_P_Q_R_for_opt_w(X_list, Y_list, B, K)
    # init w
    if prev_w is None:
        raise NotImplementedError
        w = copy.deepcopy(w_bar)
    else:
        w = prev_w
    curr_loss = 999999
    for ind_iter in range(max_iter):
        for k in range(K): # coordinate descent
            R_sum_of_other_w = torch.zeros(w_bar[k].shape, dtype=torch.cdouble)
            for k_prime in range(K):
                if k == k_prime:
                    pass
                else:
                    R_sum_of_other_w += R[(k,k_prime)]@w[k_prime]
            tmp_1 = lambda_2*torch.eye(P[k].shape[0]) + P[k]
            tmp_2 = lambda_2*w_bar[k] + Q[k] - R_sum_of_other_w
            w[k] = torch.linalg.pinv(tmp_1)@tmp_2 # Guass-Seidel
        loss_tmp = inner_loss(B_bar, w_bar, X_list, Y_list, 0, lambda_2, B, w) #lambda_1 = 0 since we are fixing B here
        if loss_tmp<=conv_criteria:
            break 
        elif (loss_tmp <= curr_loss) and (torch.norm(curr_loss - loss_tmp) <= conv_criteria):
            break
        else:
            curr_loss = loss_tmp
    return w

def Herm(vector):
    return np.transpose(np.conjugate(vector))

def Herm_tensor(vector):
    return torch.transpose(torch.conj(vector),0,1)
