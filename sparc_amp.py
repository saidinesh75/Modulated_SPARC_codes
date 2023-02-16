import numpy as np
from eta_modulated import eta_modulated
def row_map(r,Lr):
    q,r = r //Lr 
    return q

def col_map(c,Lc):
    q,r = c//Lc
    return q
    
def generate_Q(tau,phi,n,N):
    c = np.size(tau)
    r = np.size(phi)

    Q = np.zeros((n,N))
    for i in range(n):
        for j in range(N):
            Q[i,j] = tau[N//c]/phi(n//r) 
    return Q

def sparc_amp(y,beta, A,W,c,code_params, decode_params,rng,delim,cols):
    P,R,L,M,n = map( code_params.get, ['P','R','L','M','n'] )
    K = code_params['K'] if code_params['modulated'] else 1
    N = int(L*M)
    t_max, rtol= map(decode_params.get(),['t_max','rtol'])

    atol = 2*np.finfo(np.float).resolution # abs tolerance 4 early stopping

    beta_hat = np.zeros(L*M) if (K==1 or K==2) else np.zeros(L*M, dtype=complex)

    Lr = W.shape[0]               # Num of row blocks
    Mr = n // Lr                  # Entries per row block
    Lc    = W.shape[-1]               # Num of column blocks
    Mc    = (L*M) // Lc                 # Entries per column block

    # gamma = np.dot(W, np.ones(Lc))/Lc # Residual var - noise var (length Lr)  -> this is gamma at t=0
    nmse  = np.ones((t_max, Lc))      # NMSE of each column block

    z = np.zeros((n,cols))
    phi_init = np.ones((Lr,cols))
    beta_hat = np.zeros((N,cols))
    beta_T = np.zeros((N,cols))
    v_init = np.zeros((n,cols))
    # gamma = np.zeros(Lr)
    
    for t in range(t_max):
        beta_c_coeffs = np.zeros(Lc,cols)
        for p in range(Lc):
            beta_c = beta_hat[p*Mc: (p+1)*Mc,:]
            beta_c_coeffs[p,:] = (1 - (np.linalg.norm(beta_c,axis=0)**2/(L/Lc)) )  
        gamma = (1/Lc)*np.dot(W,beta_c_coeffs)

        if t==0:
            phi_t = phi_init  # needs to be changed to include "cols"
            v_tilda = v_init  
        else:
            phi_t = (np.linalg.norm(z[np.arange(0,n,Mr),:],axis=0)**2) / (n/R)
            v_tilda = np.divide(gamma,np.repeat(phi_t,Mr))
            psi_prev = np.copy(psi)
       
        z = y - np.matmul(A,beta_hat) + np.multiply(v_tilda,z)

        temp1 = np.transpose(W)/phi_t 
        tau_t = ((R/L)*np.log(K*M))/( (1/R)*np.matmul(temp1,np.ones(Lr)) )
        tau_tilda = np.repeat(tau_t,Mc)


        Q = generate_Q(tau_t,phi_t,n,N)  # generating Q matrix

        test_stat_1 = np.multiply(Q,A)
        test_stat_2 = np.matmul(np.transpose(test_stat_1),z)
        
        beta_hat = eta_modulated(beta_hat + test_stat_2,code_params,c,tau_tilda,delim,cols)

        if W.ndim == 0:
            psi       = 1 - (np.abs(beta_hat)**2).sum()/L  #magnitude of the symbols in psk =1
            nmse[t+1] = (np.abs(beta_hat-beta)**2).sum()/L
        else:
            psi       = 1 - (np.abs(beta_hat)**2).reshape(Lc,-1).sum(axis=1)/(L/Lc)
            nmse[t+1] = (np.abs(beta_hat-beta)**2).reshape(Lc,-1).sum(axis=1)/(L/Lc)

        if t>0 and np.allclose(psi, psi_prev, rtol, atol=atol):
            nmse[t:] = nmse[t]
            break
    t_final = t+1

    return beta_hat, t_final,nmse,psi























    # def eta(beta_hat,code_params,c,tau_tilda,delim,cols):
#     beta_th = np.zeros(np.shape(beta_hat))
#     P,R,L,M,dist = map(code_params.get,['P','R','L','M','dist'])
#     K = code_params['K'] if code_params['modulated'] else 1

#     for i in range(cols):
#         s1 = beta_hat[:,i]
#         exp_param = np.zeros(K,M)

#         for j in range(L):
#             beta_section = s1[ int(delim[0,j]):int(delim[1,j]+1)]
#             beta_th_section = np.zeros(int(M))
#             for k in range(K):
#                 new_exp_param = np.real(((beta_section.conj())*c[k]))/tau_tilda[j]
#                 max_exp_param = np.max(new_exp_param)
#                 max_minus_term = 0
#                 if max_exp_param>308:
#                     max_minus_term = max_exp_param - 308
#                     new_exp_param = exp_param - max_minus_term
#                 exp_param[k,:] = new_exp_param 
                
#                 denom = (np.sum(np.exp(exp_param),axis=0))  # each row corresponds to multiplication with each ck and the total np.sum() gives the double summation of the denominator

#             for k in range(int(M)):
#                 num_exp = exp_param[:,k]
#                 num = np.sum(np.multiply( c, np.exp(num_exp).reshape((np.size(num_exp),)) ))
#                 beta_th_section[k] = num/denom
#             beta_th   [int(delim[0,j]):int(delim[1,j]+1), i] = beta_th_section          
#     return beta_th