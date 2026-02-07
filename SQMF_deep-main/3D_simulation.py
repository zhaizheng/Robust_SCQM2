import numpy as np
#from SQMF_gradient_2 import *
from back_tracking_SQMF_2 import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from data_generator import *
import os
mpl.rcParams['text.usetex'] = True

# x in [-1, 1] with 20 points

def sample_on_sphere(n, d, random_state=None):
    """在 S^{d-1} 上均匀采样 n 个点"""
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X.T

def k_nearest_on_sphere(X, x0, k):
    """从球面点 X 中选出以 x0 为中心的 k 个最近邻"""
    # 若 x0 不在球面上，可先归一化
    x0 = x0 / np.linalg.norm(x0)

    # 欧氏距离（在球面上等价于角距离的单调变换）
    dists = np.linalg.norm(X - x0, axis=0)

    idx = np.argsort(dists)[:k]
    return X[:,idx], idx, dists[idx]



def eval(c,U,V,Theta, taus):
    result = []
    for i in range(len(taus)):
        tau = taus[i]
        vech_tt = vech_upper(np.outer(tau, tau))
        f = (c.flatten() + U @ tau + V @ (Theta.T @ vech_tt))
        result.append(f)
    return np.array(result).T


def rep_do(XY, mode, p_value):
    Q, Theta, c, taus ,err, step = quadratic_manifold_factorization(XY, d = 2, s = 1,
                                    eta_Q=1e-1, eta_Theta=1e-1, eta_c=1e-1, eta_tau=1e-1, T=1000, 
                                    tol=1e-7, mode=mode, delta=0.1, p=p_value)
    #new_middle = np.linspace(np.min(taus),np.max(taus),50).reshape(-1,1).tolist()
    result = eval(c,Q[:,:2], Q[:,[2]], Theta, taus)

    taus = np.array(taus)
    x = np.linspace(taus[:,0].min(),taus[:,0].max(),20)
    y = np.linspace(taus[:,1].min(),taus[:,1].max(),20)
    X, Y = np.meshgrid(x,y)
    XY_grid = np.vstack([X.ravel(), Y.ravel()])
    XYn = [XY_grid[:,i] for i in range(XY_grid.shape[1])]
    result2 = eval(c,Q[:,:2], Q[:,[2]], Theta, XYn)
    #result2 = eval(c,Q[:,0].reshape(-1,1), Q[:,1].reshape(-1,1), Theta, new_middle)
    return XY, result, result2, err

os.makedirs("modified/figures", exist_ok=True)

plt.figure()

fig, axes = plt.subplots(2, 3, figsize=(12,8), subplot_kw={'projection': '3d'})
#axes.tick_params(axis='both',labelsize=14)

types = ['lp_p']*5+['l2']

out_types = [r'$\ell_p^p, p=1.0$', r'$\ell_p^p, p=1.25$' ,r'$\ell_p^p, p=1.5$', r'$\ell_p^p, p=1.75$',r'$\ell_p^p, p=2.0$', r'$\ell_2$']

P_value = [1, 1.25, 1.5, 1.75, 2, 2]
learning_rate = []
error = []
Data = []
partition = 4
noise_function = [sample_lpp_noise]*5+[sample_l2_noise]#[sample_lpp_noise, sample_l2_noise, sample_l1_noise]
n = 300
d = 3
k = 30
for f in range(6):
    #for k in range(partition):
    #temp_data = generate_data(start = 2*np.pi/partition*k, end = 2*np.pi/partition*(k+1) , num=30)
    #Data.append(temp_data+noise_function[0](n= 30, d = 2, p=f*0.25+1)*0.1)
    data_on_sphere = sample_on_sphere(n, d, random_state=None)
    data,_, _ = k_nearest_on_sphere(data_on_sphere, np.array([1,0,1]).reshape([-1,1]), k) 
    data += noise_function[0](n= k, d = 3, p=f*0.25+1)*0.1
    Data.append(data)


for i in range(6):
    #for k in range(partition):
    XY = Data[i]
    XY, result, result2, err = rep_do(XY, types[i], P_value[i])
    axes[i//3, i%3].tick_params(axis='both',labelsize=12)

    #t = np.linspace(0,2*np.pi,100)
    #axes[i//3, i%3].plot(np.cos(t),np.sin(t),linestyle='--',linewidth=1.6)
    # Top row: scatter plots
    axes[i//3, i%3].scatter(XY[0, :], XY[1, :], XY[2,:], marker='o', s=16, label='Noisy Data')
    #axes[i//3, i%3].scatter(result[0, :], result[1, :], result[2,:], marker='x',s=16, label='Projection')
    ns = int(np.sqrt(result2.shape[1]))
    axes[i//3, i%3].plot_surface(result2[0, :].reshape([ns,ns]), result2[1, :].reshape([ns,ns]), result2[2, :].reshape([ns,ns]), color='blue', alpha=0.4)


    phi = np.linspace(0, np.pi, 20)       # 从北极到南极
    theta = np.linspace(0, 2*np.pi, 20)   # 水平旋转
    phi, theta = np.meshgrid(phi, theta)

    # 单位球参数化
    X = np.sin(phi) * np.cos(theta)
    Y = np.sin(phi) * np.sin(theta)
    Z = np.cos(phi)

    # 半透明曲面
    axes[i//3, i%3].plot_surface(X, Y, Z, color='cyan', alpha=0.2)



    #axes[i//3, i%3].plot(result2[0, :], result2[1, :], color='red', label='Fitted Curve', linewidth=2)
    #for k in range(len(XY[0,:])):
    #    axes[i//3, i%3].annotate(text='', xytext=(XY[0, k], XY[1, k], XY[2,:]),xy=(result[0, k], result[1, k], result[2,k]),arrowprops = dict(arrowstyle='->', linewidth=1, color='black'))
    axes[i//3, i%3].set_title(out_types[i],fontsize=16)
    error.append(err)
        #axes[0, i].legend()

        # Bottom row: error curve
        #axes[1, i].plot(err)
        #axes[1, i].set_title('Error')

        #axes[2, 0].semilogy(step, marker='x')

plt.tight_layout()
plt.savefig("robustness2_3d.pdf", format="pdf", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

plt.figure()
plt.tick_params(axis='both',labelsize=16)
plt.plot()
for i in range(6):
    plt.plot(np.linspace(1,len(error[i]),len(error[i])),error[i])
plt.legend(out_types)
plt.xlabel('iteration')
plt.savefig("error2_3d.pdf", format="pdf", dpi=300, bbox_inches="tight")
plt.title('Approximation Error With Iterations')
plt.show()
plt.close()






