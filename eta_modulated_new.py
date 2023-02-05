import numpy as np

def eta_modulated_new(beta_hat,code_params,c,tau_tilda,delim):
    K = code_params['K'] if code_params['modulated'] else 1
    beta_th = np.zeros(np.shape(beta_hat),dtype=complex) if K>2 else np.zeros(np.shape(beta_hat))
    P,R,L,M,dist = map(code_params.get,['P','R','L','M','dist'])
    s1 = beta_hat
        
    for j in range(L):
        beta_section = s1[ int(delim[0,j]):int(delim[1,j]+1)]
        beta_th_section = np.zeros(int(M),dtype=complex) if K>2 else np.zeros(int(M))
        exp_param = np.zeros([K,int(M)]) if K>2 else np.zeros([K,int(M)])
        for k in range(K):
            new_exp_param = np.real(((beta_section.conj())*c[k]))/tau_tilda[j]
            max_exp_param = np.max(new_exp_param)
            max_minus_term = 0
            if max_exp_param>308:
                max_minus_term = max_exp_param - 308
                new_exp_param = new_exp_param - max_minus_term
            exp_param[k,:] = new_exp_param 
        if K>2:    
            denom = np.sum(np.exp(exp_param),dtype=complex)  # each row corresponds to multiplication with each ck and the total np.sum() gives the double summation of the denominator
        else:
            denom = np.sum(np.exp(exp_param))

        for k in range(int(M)):
            num_exp = exp_param[:,k]
            if K>2: 
                num = np.sum( np.multiply( c, np.exp(num_exp).reshape((np.size(num_exp),))), dtype=complex )
            else:
                num = np.sum( np.multiply( c, np.exp(num_exp).reshape((np.size(num_exp),))))
            beta_th_section[k] = num/denom
        beta_th[int(delim[0,j]):int(delim[1,j]+1)] = beta_th_section 
                    
    return beta_th