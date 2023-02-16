import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # BE QUIET!!!! (info and warnings are not printed)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np 
import matplotlib.pyplot as plt
dir_name = "/home/saidinesh/Modulated_SPARCs/Mod_sparcs_Figures/"
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

# from generate_mm_matrix import generate_mm_matrix
from amp_demod import amp_demod

def is_power_of_2(x):
        return (x > 0) and ((x & (x - 1)) == 0)  # '&' id bitwise AND operation.

'''
def bin_arr_2_msg_vector(bin_arr, M, K=1):
    #begin comment
    Convert binary array (numpy.ndarray) to SPARC message vector

    M: entries per section of SPARC message vector
    K: parameter of K-PSK modulation for msg_vector (power of 2)
       If no modulation, K=1.
    #end comment
    assert type(M)==int and M>0 and is_power_of_2(M)
    logM = int(round(np.log2(M)))
    if K==1: # unmodulated case
        sec_size = logM  # sec_seize = 5
    else:
        assert type(K)==int and K>1 and is_power_of_2(K)
        logK = int(round(np.log2(K)))
        sec_size = logM + logK

    bin_arr_size = bin_arr.size   # bin_arr_size = 5000 (unmodulated)
    assert bin_arr_size % sec_size == 0
    L = bin_arr_size // sec_size   # Num of sections L = 1000

    if K==1 or K==2:
        msg_vector = np.zeros(L*M)    #length of msg_vector = 1000 * 32 = 32000
    else:
        msg_vector = np.zeros(L*M, dtype=complex)

    for l in range(L):
        idx = bin_arr_2_int(bin_arr[l*sec_size : l*sec_size+logM])   # idx = decimal equivalent of the 5 bits (eg: 10100 will give 20)
        if K==1:
            val = 1
        else:
            val = psk_mod(bin_arr[l*sec_size+logM : (l+1)*sec_size], K)
        msg_vector[l*M + idx] = val      # will make a 1 at the decimal equivalent in the l-th section

    return msg_vector
'''

def sc_basic(Q, omega, Lambda):
    '''
    Construct (omega, Lambda) spatially coupled base matrix
    with uncoupled base entry/vector Q.

    Q     : an np.ndarray
            1) base entry np.array(P) if regular SPARC, or
            2) base vector (power allocation) of length B
    omega : coupling width
    Lambda: coupling length
    '''

    assert type(Q) == np.ndarray

    if Q.ndim == 0: # No power allocation
        Lr = Lambda + omega - 1
        Lc = Lambda
        W_rc = Q*Lr/omega
        W    = np.zeros((Lr, Lc))
        for c in range(Lc):
            W[c : c+omega, c] = W_rc
    elif Q.ndim == 1: # With power allocation
        B  = Q.size
        Lr = Lambda + omega - 1
        Lc = Lambda * B
        W  = np.zeros((Lr, Lc))
        for c in range(Lambda):
            for r in range(c, c+omega):
                W[r, c*B :(c+1)*B] = Q*Lr/omega
    else:
        raise Exception('Something wrong with Q')

    assert np.isclose(W.mean(),np.mean(Q)),"Average base matrix values must equal P"
    return W

def create_base_matrix(code_params):
    '''
    Construct base entry/vector/matrix for Sparse Regression Codes

    For  power_allocated,  will need awgn_var, B, R and R_PA_ratio
    For spatially_coupled, will need omega and Lambda
    '''
    power_allocated,spatially_coupled = map(code_params.get,['power_allocated','spatially_coupled'])
    P,L,dist,n = map(code_params.get,['P','L','dist','n'])

    assert power_allocated==True ^ spatially_coupled== True, "Only either power_allocated or spatial coupling should be true (right now)"

    if power_allocated:
        Q = np.array(P).reshape([1,1]) # Make into np.ndarray to use .ndim==0
    else:
        # dist = map(code_params.get,['dist'])   # dist = 0 for flat and 1 for exponentially decaying
        if dist == 0:
            Q = (P)*np.ones([1,L]) 

    if not spatially_coupled:
        W = Q
    else:
        omega, Lambda = map(code_params.get,['omega','Lambda'])
        W = sc_basic(Q, omega, Lambda)

    return W

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

cols = 100
itr = 100

data=loadmat("/home/saidinesh/Modulated_SPARCs/MUB_2_6.mat")
A = np.array(data['B'])
n,_ = np.shape(A)  # (64*4160)
N = n**2
A_unitnorm = A[:,:N]

sections = np.array([4,8])               # Number of Sections
             # Number of Columns per section

EbN0_dB = np.array([0,2,4])

sec_err_ebno = np.zeros([np.size(sections),np.size(EbN0_dB)])
block_err_ebno = np.zeros([np.size(sections),np.size(EbN0_dB)])

for l in range(np.size(sections)):
    L = sections[l]
    M = int(N/L)
    P = L/n   
    A = np.sqrt(n*P/L)*A_unitnorm  #so power of each col = (nP)/L => E[||Ab||^2] = nP
    code_params   = {'P': P,    # Average codeword symbol power constraint
                    'n': n,     # Rate
                    'L': L,    # Number of sections
                    'M': M,      # Columns per section
                    'dist':0,
                    'modulated':True,
                    'power_allocated':True,
                    'spatially_coupled':False,
                    'dist':0,
                    'K':4,
                    }

    W = create_base_matrix(code_params)  # Is this required?

    delim = np.zeros([2,L])
    delim[0,0] = 0
    delim[1,0] = M-1

    for i in range(1,L):
        delim[0,i] = delim[1,i-1]+1
        delim[1,i] = delim[1,i-1]+M

    for e in range(np.size(EbN0_dB)):
        code_params.update({'EbNo_dB':EbN0_dB[e]})
        print("Running for L = {l} and Eb/N0 = {e}".format(l=sections[l], e=EbN0_dB[e]))
        K = code_params['K'] if code_params['modulated'] else 1

        if code_params['modulated'] and code_params['K']>2:
            code_params.update({'complex':True})
        else:
            code_params.update({'complex':False})
        
        decode_params = {'t_max':25 ,'rtol':1e-6}
        Eb_No_linear = np.power(10, np.divide(EbN0_dB[e],10))
        # N = int(L*M)

        bit_len = int(round(L*np.log2(K*M)))
        logM = int(round(np.log2(M)))
        logK = int(round(np.log2(K)))
        sec_size = logM + logK

        R = bit_len/n  # Rate
        Eb = n*P/bit_len
        awgn_var = Eb/Eb_No_linear
        sigma = np.sqrt(awgn_var)
        code_params.update({'awgn_var':awgn_var})
        snr_rx = P/awgn_var
        capacity = 0.5 * np.log2(1 + snr_rx)

        Lr,_ = W.shape
        Mr   = int(round(n/Lr))
        n    = Mr * Lr          # Actual codeword length
        R_actual = bit_len / n      # Actual rate
        code_params.update({'n':n, 'R_actual':R_actual})

        num_sec_errors = np.zeros((cols,itr))
        sec_err_rate = np.zeros((cols,itr))
        sec_err = 0

        for p in range(itr):
            if p%25==0:
                print("Running itr = {a} ".format(a=p))
            beta,c = generate_msg_mod_modified(code_params,rng,cols)
            np.savetxt("/home/saidinesh/Modulated_SPARCs/debug_csv_files/beta.csv", beta, delimiter=",", fmt="%.2f")
            x = np.matmul(A,beta)
            y = awgn_channel(x,awgn_var,cols,K,rand_seed=None)        

            beta_hat,t_final,nmse,psi = amp_demod(y, beta, A, W, c, code_params, decode_params,rng,delim,cols)

            diff_beta = ~(beta_hat==beta)
            num_sec_errors[:,p]= (np.count_nonzero(diff_beta,axis=0)/2)
            sec_err_rate[:,p] = num_sec_errors[:,p]/L
            sec_err = np.mean(sec_err_rate[:,p]) + sec_err
        sec_err_ebno[l,e] = sec_err/itr

fig, ax = plt.subplots()
ax.plot(EbN0_dB, sec_err_ebno[0,:],label='L=4')
ax.plot(EbN0_dB, sec_err_ebno[1,:],label='L=8')
# ax.plot(EbNo_dB, sec_err_ebno[2,:],label='L=16')
plt.legend(loc="upper left")
ax.set_yscale('log')
ax.set_title('Avg_Section_error_rate vs Eb/N0')
ax.set_xlabel('Eb/N0')
ax.set_ylabel('Section error rate')  
plt.savefig("Sec_err_rate_vs_EbN0_L48_K4_1e4.png")

print("Done")