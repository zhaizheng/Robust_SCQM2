import numpy as np
#from SQMF_gradient_2 import *
from back_tracking_SQMF_2 import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from data_generator import *
import os
mpl.rcParams['text.usetex'] = True

# x in [-1, 1] with 20 points

def generate_data(start, end, num):
    t = np.linspace(start, end, num)

    # multiplicative Gaussian noise
    #epsilon = sigma * np.random.randn(len(t))
    #epsilon -= np.sum(epsilon)/len(t)
    x = np.cos(t)
    y = np.sin(t)
    #XY = np.vstack((x * (1 + epsilon), y * (1 + epsilon)))
    XY = np.vstack((x,y))
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
									eta_Q=1e-2, eta_Theta=1e-2, eta_c=1e-2, eta_tau=1e-2, T=1000, 
	 								tol=1e-10, mode=mode, delta=0.1, p=p_value)
	new_middle = np.linspace(np.min(taus),np.max(taus),50).reshape(-1,1).tolist()
	result = eval(c,Q[:,0].reshape(-1,1), Q[:,1].reshape(-1,1), Theta, taus)
	result2 = eval(c,Q[:,0].reshape(-1,1), Q[:,1].reshape(-1,1), Theta, new_middle)
	return XY, result, result2, err, step

os.makedirs("modified/figures", exist_ok=True)

plt.figure()

fig, axes = plt.subplots(2, 3, figsize=(12,8))
#axes.tick_params(axis='both',labelsize=14)

types = ['lp_p']*5+['l2']

out_types = [r'$\ell_p^p, p=1.0$', r'$\ell_p^p, p=1.25$' ,r'$\ell_p^p, p=1.5$', r'$\ell_p^p, p=1.75$',r'$\ell_p^p, p=2.0$', r'$\ell_2$']

P_value = [1, 1.25, 1.5, 1.75, 2, 2]
learning_rate = []
error = []
Data = []
partition = 4
noise_function = [sample_lpp_noise]*5+[sample_l2_noise]#[sample_lpp_noise, sample_l2_noise, sample_l1_noise]
for f in range(6):
    k = 0
    #for k in range(partition):
    temp_data = generate_data(start = 2*np.pi/partition*k, end = 2*np.pi/partition*(k+1) , num=30)
    Data.append(temp_data+noise_function[0](n= 30, d = 2, p=f*0.25+1)*0.1)

for i in range(6):
    #for k in range(partition):
    XY = Data[i]
    XY, result, result2, err, step = rep_do(types[i], P_value[i])
    axes[i//3, i%3].tick_params(axis='both',labelsize=16)

    t = np.linspace(0,2*np.pi,100)
    axes[i//3, i%3].plot(np.cos(t),np.sin(t),linestyle='--',linewidth=1.6)
    # Top row: scatter plots
    axes[i//3, i%3].scatter(XY[0, :], XY[1, :], marker='o', s=16, label='Noisy Data')
    axes[i//3, i%3].scatter(result[0, :], result[1, :], marker='x',s=16, label='Projection')
    axes[i//3, i%3].plot(result2[0, :], result2[1, :], color='red', label='Fitted Curve', linewidth=2)
    for k in range(len(XY[0,:])):
        axes[i//3, i%3].annotate(text='', xytext=(XY[0, k], XY[1, k]),xy=(result[0, k], result[1, k]),arrowprops = dict(arrowstyle='->', linewidth=1, color='black'))
    axes[i//3, i%3].set_title(out_types[i],fontsize=16)
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
for i in range(5):
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
for i in range(5):
	plt.plot(np.linspace(1,len(error[i]),len(error[i])),error[i])
plt.legend(out_types)
plt.xlabel('iteration')
plt.savefig("error2.pdf", format="pdf", dpi=300, bbox_inches="tight")
plt.title('Approximation Error With Iterations')
plt.show()
plt.close()






