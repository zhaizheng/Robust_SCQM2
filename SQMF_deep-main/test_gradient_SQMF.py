from gradient_SQMF import *

input_dim = 3  # Dimension of input data (X)
latent_dim = 2  # Dimension of latent representation (tau)
normal_dim = 1  # Dimension of normal subspace (related to V)
sample_n = 1000  # Number of samples
learning_rate = 0.01

#X = torch.randn(input_dim, sample_n)  # Example data (100 samples)
import numpy as np
X = np.random.randn(input_dim, sample_n)
nX = np.sqrt(np.sum(X*X,0))
Y = X/nX
n = 100

def to_repeat():
	center = Y[:,1]
	t = np.array([[center[0]]*sample_n,[center[1]]*sample_n,[center[2]]*sample_n])
	idd = np.argsort(np.sum((Y-t)*(Y-t),0))
	#print(idx)
	Xs = np.array(Y[:,idd[0:n]]+0.2*np.random.randn(input_dim,n))
	_,_,_,_,err = quadratic_manifold_factorization(Xs, d=2, s = 1, eta_Q=1e-4, eta_Theta=1e-4, eta_c=1e-4, eta_tau=1e-4, T=300, tol=1e-5)
	return err

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 3, figsize=(9, 9))

for i in range(3):
    for j in range(3):
        err = to_repeat()
        axes[i, j].plot(err)
        axes[i, j].set_title(f'Plot ({i+1},{j+1})')
plt.tight_layout()
#plt.plot(err, marker='o')
plt.show()