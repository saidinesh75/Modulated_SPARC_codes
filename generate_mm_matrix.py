import numpy as np

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