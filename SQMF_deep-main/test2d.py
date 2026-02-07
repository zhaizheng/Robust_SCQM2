import numpy as np
#from SQMF_gradient_2 import *
from back_tracking_SQMF import *
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True

# x in [-1, 1] with 20 points
t = np.linspace(0, 4, 50)

# noise epsilon
epsilon = 0.08 * np.random.randn(len(t))

# y = x^2 + epsilon
x = np.cos(t) + epsilon
y = np.sin(t) + epsilon

# stack into a 2 x 20 matrix
XY = np.vstack((x, y))

def eval(c,U,V,Theta, taus):
    result = []
    for i in range(len(taus)):
        tau = taus[i]
        vech_tt = vech_upper(np.outer(tau, tau))
        f = (c.flatten() + U @ tau + V @ (Theta.T @ vech_tt))
        result.append(f)
    return np.column_stack(result)


def rep_do(mode, p_value):
	Q, Theta, c, taus ,err, step = quadratic_manifold_factorization(XY, d = 1, s = 1,
									eta_Q=1e-2, eta_Theta=1e-2, eta_c=1e-2, eta_tau=1e-2, T=600, 
	 								tol=1e-6, mode=mode, delta=0.1, p=p_value)
	new_middle = np.linspace(np.min(taus),np.max(taus),50).reshape(-1,1).tolist()
	result = eval(c,Q[:,0].reshape(-1,1), Q[:,1].reshape(-1,1), Theta, taus)
	result2 = eval(c,Q[:,0].reshape(-1,1), Q[:,1].reshape(-1,1), Theta, new_middle)
	return XY, result, result2, err, step

plt.figure()

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

types = ['lp_p', 'lp_p', 'lp_p', 'lp_p' ]

out_types = [r'$\ell_p^p, p=1.0$', r'$\ell_p^p, p=1.3$' ,r'$\ell_p^p, p=1.6$',r'$\ell_p^p, p=2.0$']

P_value = [1, 1.3, 1.6, 2]
learning_rate = []
error = []

for i in range(4):
    XY, result, result2, err, step = rep_do(types[i], P_value[i])

    # Top row: scatter plots
    axes[i//2, i%2].scatter(XY[0, :], XY[1, :], marker='o', label='Noisy Data')
    axes[i//2, i%2].scatter(result[0, :], result[1, :], marker='x', label='Projection')
    axes[i//2, i%2].plot(result2[0, :], result2[1, :], color='red', label='Fitted Curve')
    for k in range(len(XY[0,:])):
        axes[i//2, i%2].annotate(text='', xytext=(XY[0, k], XY[1, k]),xy=(result[0, k], result[1, k]),arrowprops = dict(arrowstyle='->', linewidth=1, color='black'))
    axes[i//2, i%2].set_title(out_types[i])
    learning_rate.append(step)
    error.append(err)
    #axes[0, i].legend()

    # Bottom row: error curve
    #axes[1, i].plot(err)
    #axes[1, i].set_title('Error')

    #axes[2, 0].semilogy(step, marker='x')

plt.tight_layout()
plt.savefig("robustness2.pdf", format="pdf", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

plt.figure()
for i in range(4):
	plt.scatter([1,2,3,4],learning_rate[i],marker='x')
plt.legend(types)
plt.xlabel('Q, Theta, c, tau')
plt.savefig("learning_rate2.pdf", format="pdf", dpi=300, bbox_inches="tight")
plt.title('Final Step Size')
plt.show()
plt.close()

plt.figure()
plt.plot()
for i in range(4):
	plt.plot(np.linspace(1,len(error[i]),len(error[i])),error[i])
plt.legend(types)
plt.xlabel('iteration')
plt.savefig("error2.pdf", format="pdf", dpi=300, bbox_inches="tight")
plt.title('Approximation Error With Iterations')
plt.show()
plt.close()






