import numpy as np
from eta_modulated_new import eta_modulated_new

def generate_Q(tau,phi,n,N):
    c = np.size(tau)
    r = np.size(phi)

    Mr = n/r
    Mc = N/c

    # Q = np.zeros((n,N))
    # for i in range(n):
    #     for j in range(N):
    #         Q[i,j] = tau[int(j//Mc)]/phi[int(i//Mr)]

    Q_hat = np.zeros((r,c))
    for i in range(r):
        for j in range(c):
            Q_hat[i,j] = tau[j]/phi[i]

    Q = np.repeat(Q_hat,Mc, axis = 1)
    Q = np.repeat(Q,Mr, axis=0)                 
    return Q

def amp_demod(y_cols,beta_cols, A,W,c,code_params, decode_params,rng,delim,cols):
    P,L,M,n = map(code_params.get, ['P','L','M','n'] )
    K = code_params['K'] if code_params['modulated'] else 1
    L=int(L)
    M=int(M)
    N = int(L*M)
    t_max, rtol= map(decode_params.get,['t_max','rtol'])
    atol = 2*np.finfo(np.float).resolution # abs tolerance 4 early stopping
    
    bit_len = int(round(L*np.log2(K*M)))
    logM = int(round(np.log2(M)))
    sec_size = int(round(np.log2(K*M)))

    R = bit_len/n  # Rate

    Lr = W.shape[0]               # Num of row blocks
    Mr = n // Lr                  # Entries per row block
    Lc    = W.shape[-1]               # Num of column blocks
    Mc    = (L*M) // Lc                 # Entries per column block
    W_params = {'Lr':Lr,
                'Lc':Lc,
                'Mr':Mr,
                'Mc':Mc}

    # gamma = np.dot(W, np.ones(Lc))/Lc # Residual var - noise var (length Lr)  -> this is gamma at t=0
    nmse  = np.ones((t_max, Lc))      # NMSE of each column block

    # Codebook for length M
    codeboook = np.zeros([int(M),int(K*M)],dtype=complex) if K >2 else np.zeros([int(M),int(K*M)])
    for m in range(int(M)):
        for k in range(int(K)):
            codeboook[m, (m*K)+k ]=c[k] 
            
    beta_final = np.zeros([L*M,cols]) if (K==1 or K==2) else np.zeros([L*M,cols], dtype=complex)

    for i in range(cols):
        y = y_cols[:,i]
        beta = beta_cols[:,i]

        ## Initializations
        beta_hat = np.zeros(L*M) if (K==1 or K==2) else np.zeros(L*M, dtype=complex)
        z = np.zeros((n),dtype=complex) if K>2 else np.zeros((n))
        beta_T = np.zeros(N,dtype=complex) if K>2 else np.zeros(N)
        v_init = np.zeros((n),dtype=complex) if K>2 else np.zeros((n))
        nmse  = np.ones((t_max, Lc))
        phi_t = np.zeros((Lr),dtype=complex) if K>2 else np.zeros((Lr))  # might not be needed

        for t in range(t_max):
            beta_c_coeffs = np.zeros(Lc,dtype=complex) if K>2 else np.zeros(Lc)

            if t==0:
                gamma_t = W
                v_tilda = v_init  # should be a single constant or have the same value(if we are using it as hadamard multiplication)
            else:
                for p in range(Lc):
                    beta_c = beta_hat[p*Mc: (p+1)*Mc]
                    beta_c_coeffs[p] = (1 - ( np.linalg.norm(beta_c)**2/(L/Lc) ) )  
                gamma = (1/Lc)*np.dot(W,beta_c_coeffs)
                v = np.divide(gamma,phi_t)
                v_tilda = np.repeat(v,Mr)
                psi_prev = np.copy(psi)

            # Residual step
            z = y - np.matmul(A,beta_hat) + np.multiply(v_tilda,z)

            # phi calculation
            for x in range(Lr):
                phi_t[x] = (np.linalg.norm(z[x*Mr : (x+1)*Mr],axis=0)**2) / (n/Lr)

            ## tau calculation
            temp1 = np.divide(np.transpose(W),phi_t) 
            temp2 = (1/Lr)*np.matmul(temp1,np.ones(Lr))
            temp3 = np.reciprocal(temp2)
            tau_t = ((R/2)/np.log2(K*M)) * temp3
            tau_tilda = np.repeat(tau_t,Mc)

            # generating Q matrix
            # Q = generate_Q(tau_t,phi_t,n,N)  

            # Test statistic
            # test_stat_1 = np.multiply(Q,A)
            test_stat_1 = A
            test_stat_2 = np.matmul(np.transpose(test_stat_1).conj(),z)
            test_stat_3 = beta_hat + test_stat_2
            beta_hat = eta_modulated_new(test_stat_3,code_params,c,tau_t,delim,W_params)
            np.savetxt("/home/saidinesh/Modulated_SPARCs/debug_csv_files/beta_hat.csv", beta_hat, delimiter=",", fmt="%.4e")

            if W.ndim == 0:
                psi       = 1 - (np.abs(beta_hat)**2).sum()/L  #magnitude of the symbols in psk =1
                nmse[t] = (np.abs(beta_hat-beta)**2).sum()/L
            else:
                psi       = 1 - (np.abs(beta_hat)**2).reshape(Lc,-1).sum(axis=1)/(L/Lc)
                nmse[t] = (np.abs(beta_hat-beta)**2).reshape(Lc,-1).sum(axis=1)/(L/Lc)

            if t>0 and np.allclose(psi, psi_prev, rtol, atol=atol):
                nmse[t:] = nmse[t]
                break
        t_final = t+1

        ## MAP estimate for final iteration
        for l in range(L):
            beta_section = beta_hat[ int(delim[0,l]):int(delim[1,l]+1)]
            index = np.argmax( np.real(np.matmul(np.conj(beta_section),codeboook)) )
            beta_T[ int(delim[0,l]):int(delim[1,l]+1)] = codeboook[:,index]

        beta_final[:,i] = beta_T    

    return beta_final,t_final,nmse,psi