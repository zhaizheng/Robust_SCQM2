import numpy as np
#from SQMF_gradient_2 import *
from back_tracking_SQMF import *
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True

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
	 								tol=1e-6, mode=mode, delta=0.1, p= p_value)
	new_middle = np.linspace(np.min(taus),np.max(taus),50).reshape(-1,1).tolist()
	result = eval(c,Q[:,0].reshape(-1,1), Q[:,1].reshape(-1,1), Theta, taus)
	result2 = eval(c,Q[:,0].reshape(-1,1), Q[:,1].reshape(-1,1), Theta, new_middle)
	return XY, result, result2, err, step, [Q, Theta, c]


types = ['l1', 'lp_p', 'l2_squared', 'l2']
P_value = [1, 1.2, 1.4, 1.6, 1.8, 2, 1]
types = ['lp_p', 'lp_p', 'lp_p', 'lp_p', 'lp_p','l2_squared','l2']
out_types = [r'$\ell_p^p, p=1.0$', r'$p=1.2$', r'$p=1.4$', r'$p=1.6$', r'$p=1.8$',r'$\ell_2^2$',r'$\ell_2$']

Q_var = []
c_var = []
Theta_var = []

for j in range(len(types)):
    result_c = []
    result_Q = []
    result_Theta = []
    Q_var_trial = []
    c_var_trial = []
    Theta_trial = []
    for i in range(10):
        t = np.linspace(0, 4, 50)
        epsilon = 0.03 * 5 * np.random.randn(len(t))

        x = np.cos(t) + epsilon
        y = np.sin(t) + epsilon

        XY = np.vstack((x, y))  # only useful if rep_do uses it
        _, _, _, _, _, Q_Theta_c = rep_do(types[j], P_value[j])
        result_Q.append(Q_Theta_c[0])
        result_Theta.append(Q_Theta_c[1])
        result_c.append(Q_Theta_c[2])


    result_Q = np.squeeze(np.array(result_Q))
    mean_Q = result_Q.mean(axis=0,keepdims=True)

    result_c = np.squeeze(np.array(result_c))
    mean_c = result_c.mean(axis=0,keepdims=True)

    result_Theta = np.squeeze(np.array(result_Theta))
    mean_Theta = result_Theta.mean(axis=0,keepdims=True)

    # RMS Frobenius deviation
    for i in range(10):
        Q_var_trial.append(np.linalg.norm(result_Q[[i]] - mean_Q))
        c_var_trial.append(np.linalg.norm(result_c[[i]] - mean_c))
        Theta_trial.append(np.linalg.norm(result_Theta[[i]] - mean_Theta))

    Q_var.append(np.mean(np.array(Q_var_trial)))
    c_var.append(np.mean(np.array(c_var_trial)))
    Theta_var.append(np.mean(np.array(Theta_trial)))

fig, axes = plt.subplots(1, 3, figsize=(9, 9))
final_output = [Q_var, c_var, Theta_var]

for i in range(3):
    axes[i].plot(range(len(final_output[i])), final_output[i], marker='o')
    axes[i].set_xticks(range(len(out_types)))
    axes[i].set_xticklabels(out_types)

plt.tight_layout()
plt.show()


