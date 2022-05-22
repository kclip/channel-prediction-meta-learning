import torch 

def aic_for_rank(total_x):
    p = total_x.shape[1]
    if p == 1:
        rank = 1
    else:
        N = total_x.shape[0]
        R = 0
        for ind_x in range(N):
            x = total_x[ind_x].unsqueeze(dim=1)
            R += x @ Herm_tensor(x)
        R /= N 
        L, _ = torch.linalg.eigh(R) # L: ascending order
        L, _ = torch.sort(L, descending=True)
        for ind_L in range(len(L)):
            if L[ind_L] < 0: # numerical unstableness
                L[ind_L] = -L[ind_L]
        if N < p: 
            AIC = []
            for k in range(N): # 0, ..., N-1
                sigma_square = 0
                tmp_2 = 0
                for i in range(N-k):
                    i += k
                    sigma_square += L[i]
                    tmp_2 += torch.log(L[i])
                sigma_square /= (N-k)
                tmp_1 = (N-k)*torch.log( sigma_square) 
                tmp_3 = (k*(2*N-k))/(p-1)
                tmp = tmp_1 -tmp_2 + tmp_3
                AIC.append(tmp)
        else:
            AIC = []
            for k in range(p): # 0, ..., p-1
                numer = 1
                denom = 0
                for i in range(p-k):
                    i += k
                    numer *= L[i]**(1/(p-k))
                    denom += L[i]
                denom /= (p-k)
                ratio = (numer/denom)**((p-k)*N)
                tmp = -2 * torch.log(ratio) + 2*k*(2*p - k)
                AIC.append(tmp)
        min_AIC = min(AIC)
        rank = AIC.index(min_AIC)
        if rank == 0:
            rank = 1 # we know there is at least one signal (path)
        else:
            pass
    return rank

def Herm_tensor(vector):
    return torch.transpose(torch.conj(vector),0,1)