import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # BE QUIET!!!! (info and warnings are not printed)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np 
import matplotlib.pyplot as plt
dir_name = "/home/dinesh/Modulated_SPARC_codes/Mod_sparcs_Figures/"
plt.rcParams["savefig.directory"] = os.chdir(os.path.dirname(dir_name))

import math
import sys
import numpy.linalg as la
import numpy.matlib
# import tensorflow.compat.v1 as tf
from scipy.io import loadmat
from sklearn.preprocessing import PolynomialFeatures 

rng = np.random.RandomState(seed=None)
import pickle
from generate_msg_mod_modified import generate_msg_mod_modified

def is_power_of_2(x):
    return (x > 0) and ((x & (x - 1)) == 0)  # '&' id bitwise AND operation.

def subblock_calculator(N,K):
    subblock_sizes = np.zeros(K)
    subblock_sizes[0] = np.power(2,np.floor(np.log2(N)))

    for i in range(1,K):
        r = N - np.sum(subblock_sizes)
        val = np.max(subblock_sizes)
        loc = np.argmax(subblock_sizes)
        if r<val/2:
            subblock_sizes[loc] = val/2
            subblock_sizes[i] = val/2
        else:
            subblock_sizes[i] = np.power(2,np.floor(np.log2(r)))

    subblock_sizes[::-1].sort() 

    delim = np.zeros([2,K])
    delim[0,0] = 0
    delim[1,0] = subblock_sizes[0] - 1
    for i in range(1,K):
        delim[0,i] = delim[1,i-1] + 1
        delim[1,i] = delim[1,i-1] + subblock_sizes[i]

    return subblock_sizes,delim

def awgn_channel(in_array, awgn_var, cols,K,rand_seed=None):
    '''
    Adds Gaussian noise to input array

    Real input_array:
        Add Gaussian noise of mean 0 variance awgn_var.

    Complex input_array:
        Add complex Gaussian noise. Indenpendent Gaussian noise of mean 0
        variance awgn_var/2 to each dimension.
    '''
    y = np.zeros(np.shape(in_array), dtype="complex128") if K>2 else np.zeros(np.shape(in_array))
    for c in range(cols):
        input_array = in_array[:,c]
        assert input_array.ndim == 1, 'input array must be one-dimensional'
        assert awgn_var >= 0

        rng = np.random.RandomState(rand_seed)
        n   = input_array.size

        if K<=2:
            y[:,c] =  input_array + np.sqrt(awgn_var/2)*rng.randn(n)

        elif K>2:
            noise = np.sqrt(awgn_var/2)*(rng.randn(n)+1j* rng.randn(n))
            y[:,c] =  input_array + noise

        else:
            raise Exception("Unknown input type '{}'".format(input_array.dtype))

    return y   

def eta_mod(beta_hat,code_params,c,tau_hat,delim):
    K = code_params['K'] if code_params['modulated'] else 1
    beta_th = np.zeros(np.shape(beta_hat),dtype=complex) if K>2 else np.zeros(np.shape(beta_hat))
    P,L,M,dist = map(code_params.get,['P','L','M','dist'])
    s1 = beta_hat

    for j in range(L):
        beta_section = s1[ int(delim[0,j]):int(delim[1,j]+1)]
        beta_th_section = np.zeros(int(M),dtype=complex) if K>2 else np.zeros(int(M))
        exp_param = np.zeros([K,int(M)],dtype=complex) if K>2 else np.zeros([K,int(M)])
        for k in range(K):
            new_exp_param = np.real(((beta_section.conj())*c[k]))/tau_hat
            '''
            max_exp_param = np.max(new_exp_param)
            max_minus_term = 0
            if max_exp_param>308:
                max_minus_term = max_exp_param - 308
                new_exp_param = new_exp_param - max_minus_term
            '''
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

def psk_constel(K):
    '''
    K-PSK constellation symbols
    '''
    assert type(K)==int and K>1 and is_power_of_2(K)

    if K == 2:
        c = np.array([1, -1])
    elif K == 4:
        c = np.array([1+0j, 0+1j, -1+0j, 0-1j])
    else:
        theta = 2*np.pi*np.arange(K)/K
        c     = np.cos(theta) + 1J*np.sin(theta)

    return c

''' def madhu_replica
def eta_madhu_replica(s,tau_hat,K,delim,c):
    exp_param = np.divide(np.real(np.matmul(np.conj(s),c)),tau_hat)

    for ii in range(L):
        exp_param_max = np.max(np.max(exp_param[int(delim[0:ii]):int(delim[1,ii]),:]))
        minus_term=0

        if exp_param_max>308:
            minus_term = exp_param_max-709
    return
'''

##...............Loading Dictionary matrix.................
data=loadmat("/home/dinesh/Modulated_SPARC_codes/MUB_2_6.mat")
A = np.array(data['B'],dtype = complex)
n,NN = np.shape(A)

##...............Setting up simulation parameters..............
ITR = 1e4
EbNo_dB = np.arange(0,11)
EbNo_linear = np.power(10,np.divide(EbNo_dB,10))
L_vec = np.array([4])
K = 4
AMP_ITR = 16

#................Initializing the output variables.............
RatePerRealDim = np.zeros(L_vec[0])
sec_err_rate = np.zeros([np.size(L_vec),np.size(EbNo_dB)])
blk_err_rate = np.zeros([np.size(L_vec),np.size(EbNo_dB)])

for L_index in range(L_vec[0]):
    L = L_vec[L_index]
    P = L/n             # EQUATION 1, makes Q=1(scalar) in equation 8
    W = P               # EQUATION 4, R=1,C=1

    #...............Initializing parameters for BLER calculation...........
    subblock_sizes,delim = subblock_calculator(NN,L)
    N = int(delim[-1,-1] + 1)
    M = int(subblock_sizes[0])
    A = np.sqrt(P/L)*A[:,:N]
    n_bits = np.sum(np.log2(subblock_sizes)) + L*np.log2(K)
    RatePerRealDim[int(L-1)] = n_bits/(2*n)
    Eb = n*P/n_bits
    c = psk_constel(K)
    if L_index == 0:
        codeboook = np.zeros([int(M),int(K*M)],dtype=complex) if K >2 else np.zeros([int(M),int(K*M)])
        for m in range(int(M)):
            for k in range(int(K)):
                codeboook[m, (m*K)+k ]=c[k]

    awgn_var = np.divide(Eb,EbNo_linear)
    # sigma = np.divide(awgn_var,2)

    #..............Introducing phase to subblocks...................
    # Phase_start_val = 0
    
    #..............Start loop for each EbNo value...................
    for sd in range(np.size(awgn_var)):
        blk_err = 0
        sec_err = 0
        itr = 0
        while blk_err<100 and itr<ITR:
            #................Transmitter signal..............
            itr = itr+1
            code_params   = {'P': P,    # Average codeword symbol power constraint
                    'n': n,     # Rate
                    'L': L,    # Number of sections
                    'M': M,      # Columns per section
                    'dist':0,
                    'modulated':True,
                    'power_allocated':True,
                    'spatially_coupled':False,
                    'dist':0,
                    'K':K,
                    }

            beta,c = generate_msg_mod_modified(code_params,rng,cols=1)     # np.shape(beta) = (4096,1)
            np.savetxt("/home/dinesh/Modulated_SPARC_codes/debug_csv_files/beta.csv", beta, delimiter=",", fmt="%.2f")
            #...............Received Signal.................
            x = np.matmul(A,beta)    #np.shape(x) = (64,1)
            y = awgn_channel(x,awgn_var[sd],1,K,rand_seed=None)   #np.shape(y)=(64,1)     

            #...............Modulated AMP decoding......................
            beta_hat = np.zeros(L*M) if (K==1 or K==2) else np.zeros(L*M, dtype=complex)   #shape = (4096,)
            z = np.zeros((n),dtype=complex) if K>2 else np.zeros((n))               #shape = (64,)
            beta_T = np.zeros(N,dtype=complex) if K>2 else np.zeros(N)              #shape = (4096,)
            v=0

            beta = beta.reshape(-1)
            y = y.reshape(-1)

            for ii in range(AMP_ITR):
                ymul = np.matmul(A,beta_hat)
                z = y - ymul + v*z
                gamma_hat = W*(1 - (np.linalg.norm(beta_hat)**2/L) )
                if ii>0:
                    v = gamma_hat/phi_hat

                phi_hat = np.linalg.norm(z)**2/n
                tau_hat = phi_hat/2
                test_stat2 = np.matmul(np.transpose(A).conj(),z)
                s = beta_hat + test_stat2

                if ii == AMP_ITR:
                    for l in range(L):
                        beta_section = beta_hat[ int(delim[0,l]):int(delim[1,l]+1)]
                        index = np.argmax( np.real(np.matmul(np.conj(beta_section),codeboook)) )
                        beta_T[ int(delim[0,l]):int(delim[1,l]+1)] = codeboook[:,index]
                    break

                beta_hat = eta_mod(s,code_params,c,tau_hat,delim)
            diff_beta = ~(beta_hat==beta)
            blk_err = blk_err + 1 - (beta_hat==beta).all()
            num_sec_errors = num_sec_errors + np.count_nonzero(diff_beta,axis=0)/2
            if blk_err ==0:
                break
        sec_err_rate[L_index,sd] = num_sec_errors/(itr*L)
        blk_err_rate[L_index,sd] = blk_err/itr


    print("done") 