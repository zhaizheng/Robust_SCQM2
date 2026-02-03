import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from SQMF_v3 import *
import matplotlib.pyplot as plt

import os
os.makedirs("figures", exist_ok=True)


def normalize_img(img):
    img = img - img.min()
    img = img / (img.max() + 1e-8)
    return img

# MNIST to tensor
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=len(train_dataset),
    shuffle=False
)

images, labels = next(iter(train_loader))

mask = (labels == 2) | (labels == 6) | (labels == 8)

images_47 = images[mask]          # (n, 1, 28, 28)
labels_47 = labels[mask]

n_samples = 40   # you can increase later
images_47 = images_47[:n_samples]
labels_47 = labels_47[:n_samples]

# flatten to vectors
X = images_47.view(images_47.shape[0], -1)   # (n, 784)
X = X.numpy().T                              # (784, n)

# mean centering (important!)
#X = X - X.mean(axis=1, keepdims=True)

d = 2      # linear latent dimension
s = 20     # quadratic latent dimension

Q_lin, Theta_lin, c_lin, taus_lin, err_lin, _ = quadratic_manifold_factorization(
    X, d=d, s=s,
    eta_Q=1e-1, eta_Theta=1e-1, eta_c=1e-1, eta_tau=1e-1,
    T=1000, tol=1e-10, mode='l2', delta=0.1, p=2,
    set_Theta_zero=True
)

U_lin = Q_lin[:, :d]
V_lin = Q_lin[:, d:d+s]

F_lin, R_lin, _, _ = forward_all(X, taus_lin, c_lin, U_lin, V_lin, Theta_lin)

Q_quad, Theta_quad, c_quad, taus_quad, err_quad, _ = quadratic_manifold_factorization(
    X, d=d, s=s,
    eta_Q=1e-1, eta_Theta=1e-1, eta_c=1e-1, eta_tau=1e-1,
    T=1000, tol=1e-10, mode='l2', delta=0.1, p=2,
    set_Theta_zero=False
)

U_quad = Q_quad[:, :d]
V_quad = Q_quad[:, d:d+s]

F_quad, R_quad, _, _ = forward_all(X, taus_quad, c_quad, U_quad, V_quad, Theta_quad)


n = X.shape[1]

X_img      = X.T.reshape(n, 28, 28)
F_lin_img  = F_lin.T.reshape(n, 28, 28)
F_quad_img = F_quad.T.reshape(n, 28, 28)


n_show = 8
fig, axes = plt.subplots(n_show, 3, figsize=(6, 2*n_show))

for i in range(n_show):
    axes[i, 0].imshow(normalize_img(X_img[i]), cmap="gray")
    axes[i, 0].set_title("Original")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(normalize_img(F_lin_img[i]), cmap="gray")
    axes[i, 1].set_title("Linear (Θ = 0)")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(normalize_img(F_quad_img[i]), cmap="gray")
    axes[i, 2].set_title("Quadratic (Θ ≠ 0)")
    axes[i, 2].axis("off")


plt.tight_layout()
plt.savefig(
    "figures/mnist_reconstruction_comparison.pdf",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.02
)
plt.close()


taus_arr = np.array(taus_quad)   # shape (n, 2)

tau_min = taus_arr.min(axis=0)   # (2,)
tau_max = taus_arr.max(axis=0)   # (2,)

n_interp = 10

# 对每一维做线性插值
tau1_vals = np.linspace(tau_min[0], tau_max[0], n_interp)
tau2_vals = np.linspace(tau_min[1], tau_max[1], n_interp)


tau_grid = []

for t1 in tau1_vals:
    for t2 in tau2_vals:
        tau_grid.append(np.array([t1, t2]))

tau_grid = np.array(tau_grid)   # (100, 2)


taus_arr_lin = np.array(taus_lin)   # shape (n, 2)

tau_min_lin = taus_arr_lin.min(axis=0)   # (2,)
tau_max_lin = taus_arr_lin.max(axis=0)   # (2,)

n_interp = 10


# 对每一维做线性插值
tau1_vals_lin = np.linspace(tau_min_lin[0], tau_max_lin[0], n_interp)
tau2_vals_lin = np.linspace(tau_min_lin[1], tau_max_lin[1], n_interp)


tau_grid_lin = []

for t1 in tau1_vals_lin:
    for t2 in tau2_vals_lin:
        tau_grid_lin.append(np.array([t1, t2]))

tau_grid_lin = np.array(tau_grid_lin)   # (100, 2)

U = Q_quad[:, :d]
V = Q_quad[:, d:d+s]

D = X.shape[0]
n_grid = tau_grid.shape[0]


F_interp = np.zeros((D, n_grid))

for i in range(n_grid):
    tau = tau_grid[i]                          # (2,)
    vech_tt = vech_upper(np.outer(tau, tau))   # (d(d+1)/2,)
    quad = Theta_quad.T @ vech_tt               # (s,)
    F_interp[:, i] = c_quad[:, 0] + U @ tau + V @ quad

# ===== Linear interpolation (Θ = 0) =====

U = Q_lin[:, :d]
c = c_lin

F_interp_lin = np.zeros((D, n_grid))

for i in range(n_grid):
    tau = tau_grid_lin[i]              # (2,)
    F_interp_lin[:, i] = c[:, 0] + U @ tau

F_interp_lin_img = F_interp_lin.T.reshape(n_grid, 28, 28)

F_interp_img = F_interp.T.reshape(n_grid, 28, 28)

fig = plt.figure(figsize=(14, 6))

# ===============================
# Left: Linear interpolation
# ===============================
gs_left = fig.add_gridspec(
    nrows=n_interp,
    ncols=n_interp,
    left=0.03, right=0.48,
    top=0.90, bottom=0.05,
    wspace=0.05, hspace=0.05
)

for i in range(n_interp):
    for j in range(n_interp):
        idx = i * n_interp + j
        ax = fig.add_subplot(gs_left[i, j])
        ax.imshow(normalize_img(F_interp_lin_img[idx]), cmap="gray")
        ax.axis("off")

fig.text(0.25, 0.93, "Linear interpolation (Θ = 0)", 
         ha="center", va="center", fontsize=14)




# ===============================
# Right: Quadratic interpolation
# ===============================
gs_right = fig.add_gridspec(
    nrows=n_interp,
    ncols=n_interp,
    left=0.52, right=0.97,
    top=0.90, bottom=0.05,
    wspace=0.05, hspace=0.05
)

for i in range(n_interp):
    for j in range(n_interp):
        idx = i * n_interp + j
        ax = fig.add_subplot(gs_right[i, j])
        ax.imshow(normalize_img(F_interp_img[idx]), cmap="gray")
        ax.axis("off")

fig.text(0.75, 0.93, "Quadratic interpolation (Θ ≠ 0)", 
         ha="center", va="center", fontsize=14)

plt.savefig(
    "figures/mnist_manifold_interpolation.pdf",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.02
)

plt.close()
