import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # BE QUIET!!!! (info and warnings are not printed)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np 
import matplotlib.pyplot as plt

import math
import sys
import numpy.linalg as la
import numpy.matlib
# import tensorflow.compat.v1 as tf
from scipy.io import loadmat
from sklearn.preprocessing import PolynomialFeatures 


rng = np.random.RandomState(seed=None)
import pickle

from eta import eta
from power_dist import power_dist
from generate_message_modulated import generate_message_modulated
from tau_calculate import tau_calculate
from generate_mm_matrix import generate_mm_matrix
from sparc_amp import sparc_amp

EbNo_dB = np.array([10])
cols = 10
itr = 10
def is_power_of_2(x):
        return (x > 0) and ((x & (x - 1)) == 0)  # '&' id bitwise AND operation.

## Check code params:
def check_code_params(code_params):

        '''
        Check SPARC code parameters
        '''

        code_params_copy = {} # Will overwrite original (prevents unwanted params)

        # to check if all code parameters are present, and if yes, they will be copied into code_params_copy
        def in_code_param_list(code_param_list):
            if all( [ (item in code_params) for item in code_param_list ] ):
                for item in code_param_list:
                    code_params_copy[item] = code_params[item]
            else:
                raise Exception('Need code parameters {}.'.format(code_param_list))

        # Check SPARC type e.g. power allocated, spatially coupled
        sparc_type_list = ['complex',
                        'modulated',
                        'power_allocated',
                        'spatially_coupled']
        
        #assign boolean values for sparc type
        for item in sparc_type_list:  #changed "for item in" to "for item, key in"
            if item not in code_params:
                code_params[item] = False # default
            else:
                assert type(code_params[item]) == bool,\
                        "'{}' must be boolean".format(code_params[item])  #changed key to code_params[key]
            # code_params_copy[item] = copy(code_params[item])
            code_params_copy[item] = code_params[item]

        # Required SPARC code parameters (all SPARC types)
        code_param_list = ['P','R','L','M']
        in_code_param_list(code_param_list)  #checks if all 4 are present and then creates a copy

        P,R,L,M = map(code_params.get, code_param_list) 
        
        assert (type(P)==float or type(P)==np.float64) and P>0
        assert (type(R)==float or type(R)==np.float64) and R>0
        assert type(L)==int and L>0
        assert type(M)==int and M>0 and is_power_of_2(M)

        # Required SPARC code parameters (modulated)
        # ONLY SUPPORTS PSK MODULATION
        if code_params['modulated']:
            code_param_list = ['K']
            in_code_param_list(code_param_list)
            K = code_params['K']
            assert type(K)==int and K>1 and is_power_of_2(K)
            if not code_params['complex']:
                assert K==2, 'Real-modulated SPARCs requires K=2'

        # Required SPARC code parameters (power allocated)
        # ONLY SUPPORTS ITERATIVE POWER ALLOCATION
        if code_params['power_allocated']:
            code_param_list = ['B', 'R_PA_ratio']
            in_code_param_list(code_param_list)
            B, R_PA_ratio = map(code_params.get, code_param_list)
            assert type(B)==int and B>1
            assert L % B == 0, 'B must divide L'
            assert type(R_PA_ratio)==float or type(R_PA_ratio)==np.float64
            assert R_PA_ratio>=0

        # Required SPARC code parameters (spatially coupled)
        # ONLY SUPPORTS OMEGA, LAMBDA BASE MATRICES
        if code_params['spatially_coupled']:
            code_param_list = ['omega', 'Lambda']
            in_code_param_list(code_param_list)
            omega, Lambda = map(code_params.get, code_param_list)
            assert type(omega)==int and omega>1
            assert type(Lambda)==int and Lambda>=(2*omega-1)
            assert L % Lambda == 0, 'Lambda must divide L'

        if code_params['power_allocated'] and code_params['spatially_coupled']:
            assert L % (Lambda*B) == 0, 'Lambda*B must divide L'

        # Overwrite orignal
        code_params.clear()
        code_params.update(dict(code_params_copy))

def bin_arr_2_msg_vector(bin_arr, M, K=1):
    '''
    Convert binary array (numpy.ndarray) to SPARC message vector

    M: entries per section of SPARC message vector
    K: parameter of K-PSK modulation for msg_vector (power of 2)
       If no modulation, K=1.
    '''
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

# def create_base_matrix(P, power_allocated=False, spatially_coupled=False, **kwargs):
def create_base_matrix(P, power_allocated, spatially_coupled, **kwargs):
    '''
    Construct base entry/vector/matrix for Sparse Regression Codes

    For  power_allocated,  will need awgn_var, B, R and R_PA_ratio
    For spatially_coupled, will need omega and Lambda
    '''
    if not power_allocated:
        Q = np.array(P) # Make into np.ndarray to use .ndim==0
    # else:
    #     awgn_var,B,R,R_PA_ratio = map(kwargs.get,['awgn_var','B','R','R_PA_ratio'])
    #     Q = pa_iterative(P, awgn_var, B, R*R_PA_ratio)

    if not spatially_coupled:
        W = Q
    else:
        omega, Lambda = map(kwargs.get,['omega','Lambda'])
        W = sc_basic(Q, omega, Lambda)

    return W

# Def to generate measurement matrix
def generate_mm_matrix(W,code_params,rng):

    P,R,L,M,dist,n = map(code_params.get,['P','R','L','M','dist','n'])
    K = code_params['K'] if code_params['modulated'] else 1
    N = int(L*M)
    Lr,Lc = np.shape(W)

    assert n%Lr ==0
    assert N%Lc ==0

    Mr = int(n/Lr)  # size of row block
    Mc = int(N/Lc)  # size of column block

    A = np.zeros([n,N], dtype=complex)
    
    for r in range(Lr):
        for c in range(Lc):
            # A[int(r*Mr):int((r+1)*Mr) , int(c*Mc):int((c+1)*Mc) ]  = np.random.normal(size=(Mr, Mc), scale= W[r,c] / L).astype(np.float64)
            A[int(r*Mr):int((r+1)*Mr) , int(c*Mc):int((c+1)*Mc) ] = np.sqrt(W[r,c]/(2*L))*(rng.randn(Mr,Mc)+1j* rng.randn(Mr,Mc))

    return A 

def awgn_channel(in_array, awgn_var, cols,rand_seed=None,):
    '''
    Adds Gaussian noise to input array

    Real input_array:
        Add Gaussian noise of mean 0 variance awgn_var.

    Complex input_array:
        Add complex Gaussian noise. Indenpendent Gaussian noise of mean 0
        variance awgn_var/2 to each dimension.
    '''
    y = np.zeros(np.shape(in_array))
    for c in range(cols):
        input_array = in_array[:,c]
        assert input_array.ndim == 1, 'input array must be one-dimensional'
        assert awgn_var >= 0

        rng = np.random.RandomState(rand_seed)
        n   = input_array.size

        if input_array.dtype == np.float:
            y[:,c] =  input_array + np.sqrt(awgn_var)*rng.randn(n)

        elif input_array.dtype == np.complex:
            y[:,c] =  input_array + np.sqrt(awgn_var/2)*(rng.randn(n)+1j* rng.randn(n))

        else:
            raise Exception("Unknown input type '{}'".format(input_array.dtype))

    return y        

code_params   = {'P': 15.0,    # Average codeword symbol power constraint
                    'R': 1.3,     # Rate
                    'L': 800,    # Number of sections
                    'M': 32,      # Columns per section
                    'dist':0,
                    'complex':True,
                    'modulated':True,
                    'power_allocated':False,
                    'spatially_coupled':True,
                    'K':4,
                    'omega':6,
                    'Lambda':32,
                    'rho':0
                    }
                    
decode_params = {'t_max':25 ,'rtol':1e-6}

W = create_base_matrix(**code_params)

# check_code_params(code_params)

for e in range(np.size(EbNo_dB)):
    code_params['EbNo_dB']= EbNo_dB[e]
    # check_code_params(code_params)

    P,R,L,M,dist = map(code_params.get,['P','R','L','M','dist'])
    K = code_params['K'] if code_params['modulated'] else 1
    EbNo_dB = code_params['EbNo_dB']

    Eb_No_linear = np.power(10, np.divide(EbNo_dB,10))
    N = int(L*M)

    #Section sizes 
    bit_len = int(round(L*np.log2(K*M)))
    logM = int(round(np.log2(M)))
    sec_size = int(round(np.log2(K*M)))

    # Actual rate
    n = int(round(bit_len/R))
    
    # It's given in the paper that w ~ CN(0,sima^2) which implies sigma^2 goes to each of real and img parts. So total N0=sigma^2.
    Eb = n*P/bit_len
    awgn_var = Eb/Eb_No_linear    # sigma^2=N0
    sigma = np.sqrt(awgn_var)     
    code_params.update({'awgn_var':awgn_var})   #  =sqrt(N0)

    snr_rx = P/awgn_var
    capacity = 0.5 * np.log2(1 + snr_rx)

    # power_dist_params = {'P':P,
    #                     'L':L,
    #                     'n':n,
    #                     'capacity':capacity,
    #                     'dist':dist,        # dist=0 for flat and 1 for exponential and 2 for modified PA
    #                     'R': R
    #                     }
    # beta_non_zero_values, P_vec = power_dist(power_dist_params)

    # Update code_params   
    Lr,_ = W.shape
    Mr   = int(round(n/Lr))
    n    = Mr * Lr          # Actual codeword length
    R_actual = bit_len / n      # Actual rate
    code_params.update({'n':n, 'R_actual':R_actual})

    delim = np.zeros([2,L])
    delim[0,0] = 0
    delim[1,0] = M-1

    for i in range(1,L):
        delim[0,i] = delim[1,i-1]+1
        delim[1,i] = delim[1,i-1]+M

    A = generate_mm_matrix(W,code_params,rng)

    for p in range(itr):
        beta,c = generate_message_modulated(code_params,rng,cols)
        x = np.matmul(A,beta)
        y = awgn_channel(x,awgn_var,cols,rand_seed=None)        

        beta_hat = sparc_amp(y, beta, A, W, c, code_params, decode_params,rng,delim,cols)
print("done")        