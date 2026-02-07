import numpy as np

# ============================================================
# Utilities
# ============================================================

def vech_upper(A: np.ndarray) -> np.ndarray:
    """
    Upper-triangular vectorization (including diagonal)
    """
    idx = np.triu_indices(A.shape[0])
    return A[idx]


def build_M_tau(tau: np.ndarray) -> np.ndarray:
    """
    tau: (d,)
    returns M_tau: (m, d), m = d(d+1)/2
    """
    tau = np.asarray(tau).reshape(-1)
    d = tau.size
    rows = []
    for i in range(d):
        for j in range(i, d):
            row = np.zeros(d)
            row[j] = tau[i]
            rows.append(row)
    return np.vstack(rows)


def build_N_tau(tau: np.ndarray) -> np.ndarray:
    """
    tau: (d,)
    returns N_tau: (m, d)
    """
    tau = np.asarray(tau).reshape(-1)
    d = tau.size
    rows = []
    for i in range(d):
        for j in range(i, d):
            row = np.zeros(d)
            row[i] = tau[j]
            rows.append(row)
    return np.vstack(rows)


# ============================================================
# Loss functions
# ============================================================

def loss_factory(mode="l2", kwargs=None):
    if kwargs is None:
        kwargs = {}

    if mode == "l2":
        def loss(r):
            val = 0.5 * np.sum(r**2)
            grad = r
            return val, grad

    elif mode == "l1":
        delta = kwargs.get("delta", 1e-3)

        def loss(r):
            val = np.sum(np.sqrt(r**2 + delta))
            grad = r / np.sqrt(r**2 + delta)
            return val, grad

    else:
        raise ValueError("Unknown loss mode")

    return loss


# ============================================================
# Model evaluation
# ============================================================

def eval_one_tau(
    tau: np.ndarray,
    c: np.ndarray,
    Theta: np.ndarray,
    U: np.ndarray,
    V: np.ndarray
) -> np.ndarray:
    """
    tau: (d,)
    c: (D,1)
    returns f: (D,)
    """
    tau = np.asarray(tau).reshape(-1)
    vech_tt = vech_upper(np.outer(tau, tau))   # (m,)
    quad = Theta.T @ vech_tt                   # (s,)
    f = c[:, 0] + U @ tau + V @ quad           # (D,)
    return f


def objective_single(
    x_i: np.ndarray,
    tau: np.ndarray,
    c: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    Theta: np.ndarray,
    loss_func
) -> float:
    r = eval_one_tau(tau, c, Theta, U, V) - x_i
    return float(loss_func(r)[0])


# ============================================================
# Optimize tau for one sample
# ============================================================

def optimize_tau(
    x_i: np.ndarray,
    c: np.ndarray,
    Q: np.ndarray,
    Theta: np.ndarray,
    d: int,
    s: int,
    eta_tau: float,
    armijo_c: float = 1e-2,
    bt_max: int = 20,
    steps: int = 100,
    mode: str = "l2",
    tol: float = 1e-6,
    **kwargs
):
    loss_func = loss_factory(mode, kwargs)

    tau = np.zeros(d)      # ✅ 1D
    U = Q[:, :d]
    V = Q[:, d:d+s]

    last_obj = None

    for _ in range(steps):
        M_tau = build_M_tau(tau)
        N_tau = build_N_tau(tau)
        J_tau = U + V @ Theta.T @ (M_tau + N_tau)   # (D, d)

        r = eval_one_tau(tau, c, Theta, U, V) - x_i
        _, g = loss_func(r)                         # (D,)

        grad_tau = J_tau.T @ g                      # (d,)
        gnorm = np.linalg.norm(grad_tau)
        if gnorm < 1e-12:
            break

        desc = -grad_tau
        obj0 = objective_single(x_i, tau, c, U, V, Theta, loss_func)

        eta = eta_tau
        tau_new = tau

        for _ in range(bt_max):
            tau_try = tau + eta * desc
            obj_try = objective_single(x_i, tau_try, c, U, V, Theta, loss_func)
            if obj_try <= obj0 - armijo_c * eta * gnorm**2:
                tau_new = tau_try
                break
            eta *= 0.5

        if last_obj is not None and abs(obj_try - obj0) < tol:
            break

        last_obj = obj_try
        tau = tau_new

    return tau


# ============================================================
# Forward pass (all samples)
# ============================================================

def forward_all(X, taus, c, U, V, Theta):
    n = X.shape[1]
    D = X.shape[0]

    F = np.zeros((D, n))
    R = np.zeros((D, n))
    vech_list = []
    quad_list = []

    for i in range(n):
        tau = taus[i]
        vech_tt = vech_upper(np.outer(tau, tau))
        quad = Theta.T @ vech_tt

        F[:, i] = c[:, 0] + U @ tau + V @ quad
        R[:, i] = X[:, i] - F[:, i]

        vech_list.append(vech_tt)
        quad_list.append(quad)

    return F, R, vech_list, quad_list


# ============================================================
# Main SQMF routine
# ============================================================

def quadratic_manifold_factorization(
    X: np.ndarray,
    d: int,
    s: int,
    eta_Q=1e-1,
    eta_Theta=1e-1,
    eta_c=1e-1,
    eta_tau=1e-1,
    T=100,
    tol=1e-6,
    mode="l2",
    delta=1e-3,
    p=2,
    set_Theta_zero=False
):
    D, n = X.shape
    loss_func = loss_factory(mode, {"delta": delta})

    # ---- Initialization ----
    c = np.mean(X, axis=1, keepdims=True)  # (D,1)
    Xc = X - c
    U0, _, _ = np.linalg.svd(Xc, full_matrices=False)
    Q = U0[:, :d+s].copy()

    U = Q[:, :d]
    V = Q[:, d:d+s]
    Theta = np.zeros((d * (d + 1) // 2, s), dtype=X.dtype)

    taus = [(Q[:, :d].T @ (X[:, i] - c[:, 0])) for i in range(n)]

    err_hist = []

    for t in range(T):
        # ---- update tau ----
        for i in range(n):
            taus[i] = optimize_tau(
                X[:, i], c, Q, Theta,
                d=d, s=s,
                eta_tau=eta_tau,
                mode=mode,
                delta=delta
            )

        # ---- forward ----
        U, V = Q[:, :d], Q[:, d:d+s]
        F, R, vech_list, quad_list = forward_all(X, taus, c, U, V, Theta)

        # ---- update c ----
        c = np.mean(X - U @ np.array(taus).T - V @ np.array(quad_list).T,
                    axis=1, keepdims=True)

        # ---- update Theta ----
        if not set_Theta_zero:
            A = np.zeros_like(Theta)
            B = np.zeros_like(Theta)

            for i in range(n):
                A += np.outer(vech_list[i], quad_list[i])
                B += np.outer(vech_list[i], V.T @ R[:, i])

            Theta += eta_Theta * (B - A)

        # ---- update Q ----
        Z = np.vstack([np.array(taus).T, np.array(quad_list).T])
        Q = (X - c) @ Z.T
        Q, _ = np.linalg.qr(Q)

        # ---- error ----
        err = np.mean(np.linalg.norm(R, axis=0)**p)
        err_hist.append(err)

        if t > 0 and abs(err_hist[-2] - err) < tol:
            break

    return Q, Theta, c, taus, err_hist, t