import numpy as np

def eta_modulated(beta_hat,code_params,c,tau_tilda,delim,cols):
    beta_th = np.zeros(np.shape(beta_hat))
    P,R,L,M,dist = map(code_params.get,['P','R','L','M','dist'])
    K = code_params['K'] if code_params['modulated'] else 1

    for i in range(cols):
        s1 = beta_hat[:,i]
        exp_param = np.zeros(K,M)

        for j in range(L):
            beta_section = s1[ int(delim[0,j]):int(delim[1,j]+1)]
            beta_th_section = np.zeros(int(M))
            for k in range(K):
                new_exp_param = np.real(((beta_section.conj())*c[k]))/tau_tilda[j]
                max_exp_param = np.max(new_exp_param)
                max_minus_term = 0
                if max_exp_param>308:
                    max_minus_term = max_exp_param - 308
                    new_exp_param = exp_param - max_minus_term
                exp_param[k,:] = new_exp_param 
                
                denom = (np.sum(np.exp(exp_param),axis=0))  # each row corresponds to multiplication with each ck and the total np.sum() gives the double summation of the denominator

            for k in range(int(M)):
                num_exp = exp_param[:,k]
                num = np.sum(np.multiply( c, np.exp(num_exp).reshape((np.size(num_exp),)) ))
                beta_th_section[k] = num/denom
            beta_th   [int(delim[0,j]):int(delim[1,j]+1), i] = beta_th_section          
    return beta_th