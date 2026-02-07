import numpy as np
#from SQMF_gradient_2 import *
from back_tracking_SQMF import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
mpl.rcParams['text.usetex'] = True

# x in [-1, 1] with 20 points

def generate_data(start, end, num, sigma):
    t = np.linspace(start, end, num)

    # multiplicative Gaussian noise
    epsilon = sigma * np.random.randn(len(t))
    epsilon -= np.sum(epsilon)/len(t)
    x = np.cos(t)
    y = np.sin(t)
    XY = np.vstack((x * (1 + epsilon), y * (1 + epsilon)))
    return XY

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
	 								tol=1e-8, mode=mode, delta=0.1, p=p_value)
	new_middle = np.linspace(np.min(taus),np.max(taus),50).reshape(-1,1).tolist()
	result = eval(c,Q[:,0].reshape(-1,1), Q[:,1].reshape(-1,1), Theta, taus)
	result2 = eval(c,Q[:,0].reshape(-1,1), Q[:,1].reshape(-1,1), Theta, new_middle)
	return XY, result, result2, err, step

os.makedirs("modified/figures", exist_ok=True)

plt.figure()

fig, axes = plt.subplots(1, 4, figsize=(32,8))
#axes.tick_params(axis='both',labelsize=14)

types = ['lp_p', 'lp_p', 'lp_p', 'l2' ]

out_types = [r'$\ell_p^p, p=1.0$', r'$\ell_p^p, p=1.5$' ,r'$\ell_p^p, p=2.0$', r'$\ell_2$']

P_value = [1, 1.5, 2, 0]
learning_rate = []
error = []
Data = []
partition = 4
for k in range(partition):
    temp_data = generate_data(start = 2*np.pi/partition*k, end = 2*np.pi/partition*(k+1) , num=15, sigma = 0.06)
    Data.append(temp_data)

for i in range(4):
    for k in range(partition):
        XY = Data[k]
        XY, result, result2, err, step = rep_do(types[i], P_value[i])
        axes[i].tick_params(axis='both',labelsize=16)

        t = np.linspace(0,2*np.pi,100)
        axes[i].plot(np.cos(t),np.sin(t),linestyle='--',linewidth=1.6)
        # Top row: scatter plots
        axes[i].scatter(XY[0, :], XY[1, :], marker='o', s=16, label='Noisy Data')
        axes[i].scatter(result[0, :], result[1, :], marker='x',s=16, label='Projection')
        axes[i].plot(result2[0, :], result2[1, :], color='red', label='Fitted Curve', linewidth=2)
        for k in range(len(XY[0,:])):
            axes[i].annotate(text='', xytext=(XY[0, k], XY[1, k]),xy=(result[0, k], result[1, k]),arrowprops = dict(arrowstyle='->', linewidth=1, color='black'))
        axes[i].set_title(out_types[i],fontsize=16)
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
plt.tick_params(axis='both',labelsize=16)
for i in range(4):
	plt.scatter([1,2,3,4],learning_rate[i],marker='x')
plt.legend(out_types)
plt.xlabel('Q, Theta, c, tau')
plt.savefig("learning_rate2.pdf", format="pdf", dpi=300, bbox_inches="tight")
plt.title('Final Step Size')
plt.show()
plt.close()

plt.figure()
plt.tick_params(axis='both',labelsize=16)
plt.plot()
for i in range(4):
	plt.plot(np.linspace(1,len(error[i]),len(error[i])),error[i])
plt.legend(out_types)
plt.xlabel('iteration')
plt.savefig("error2.pdf", format="pdf", dpi=300, bbox_inches="tight")
plt.title('Approximation Error With Iterations')
plt.show()
plt.close()






