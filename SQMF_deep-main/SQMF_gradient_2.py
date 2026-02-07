import numpy as np

# ----------------------------
# Helper functions
# ----------------------------

def sign(x):
    """Subgradient of L1 loss"""
    return np.sign(x)

def sym(A):
    return 0.5 * (A + A.T)

def retract_stiefel(Q):
    """QR-based retraction onto Stiefel manifold"""
    #Q, _ = np.linalg.qr(Q)
    Q, R = np.linalg.qr(Q)
    # Optional: ensure positive diagonal of R for uniqueness
    Q = Q * np.sign(np.diag(R))
    return Q

def vech_upper(A):
    """Upper-triangular vectorization (including diagonal)"""
    d = A.shape[0]
    return np.array([A[i, j] for i in range(d) for j in range(i, d)])

def build_M_tau(tau):
    """Matrix M_tau in R^{d(d+1)/2 x d}"""
    d = len(tau)
    rows = []
    for i in range(d):
        for j in range(i, d):
            row = np.zeros(d)
            row[j] = tau[i]
            rows.append(row)
    return np.vstack(rows)

def build_N_tau(tau):
    """Matrix N_tau in R^{d(d+1)/2 x d}"""
    d = len(tau)
    rows = []
    for i in range(d):
        for j in range(i, d):
            row = np.zeros(d)
            row[i] = tau[j]
            rows.append(row)
    return np.vstack(rows)


def loss_factory(loss, kwargs):
    """
    Returns a function f(x, y) -> (value, grad_x)
    """

    def l1(r):
        val = np.linalg.norm(r, 1)
        grad = np.sign(r)
        return val, grad

    def l2(r):
        norm = np.linalg.norm(r, 2)
        if norm == 0:
            return 0.0, np.zeros_like(r)
        return norm, r / norm

    def l2_squared(r):
        return np.dot(r, r), 2 * r

    def lp_p(r):
        p = kwargs.get("p", 2)
        val = np.sum(np.abs(r) ** p)
        grad = p * np.abs(r) ** (p - 1) * np.sign(r)
        return val, grad

    def huber(r):
        delta = kwargs.get("delta", 1.0)
        norm = np.linalg.norm(r, 2)

        if norm <= delta:
            return 0.5 * norm**2, r
        else:
            return delta * norm - 0.5 * delta**2, delta * r / norm

    def mahalanobis(r):
        M = kwargs["M"]
        val = np.sqrt(r.T @ M @ r)
        if val == 0:
            return 0.0, np.zeros_like(r)
        grad = M @ r / val
        return val, grad

    losses = {
        "l1": l1,
        "l2": l2,
        "l2_squared": l2_squared,
        "lp_p": lp_p,
        "huber": huber,
        "mahalanobis": mahalanobis,
    }

    return losses[loss]

def eval(X, taus, c, Q, Theta, d, s, func):
    U = Q[:, :d]
    V = Q[:, d:d+s]
    r_errr = []
    for i in range(len(taus)):
        tau = taus[i]
        vech_tt = vech_upper(np.outer(tau, tau))
        f = (
            c.flatten()
            + U @ tau
            + V @ (Theta.T @ vech_tt)
        )
        r = f - X[:, i]
        r_errr.append(r)

    return np.sum([func(e)[0] for e in r_errr])
# ----------------------------
# Main algorithm
# ----------------------------

def quadratic_manifold_factorization(
    X, d, s,
    eta_Q, eta_Theta, eta_c, eta_tau,
    T=100, tol=1e-6, mode='l1', **kwargs
):
    """
    X: D x n data matrix
    """
    D, n = X.shape

    # ---- Initialization ----
    c = np.mean(X, axis=1, keepdims=True)

    Xc = X - c
    U0, _, _ = np.linalg.svd(Xc, full_matrices=False)

    Q = U0[:, :d+s]

    U = Q[:, :d]
    V = Q[:, d:d+s]

    taus = [U.T @ (X[:, i:i+1] - c) for i in range(n)]
    taus = [tau.flatten() for tau in taus]

    #Theta = np.random.randn(d*(d+1)//2, s) * 0.01

    Theta = np.zeros((d * (d + 1) // 2, s))

    # ---- Main loop ----
    err = []
    for t in range(T):

        # Forward pass
        g_list = []
        f_list = []
        r_errr = []

        #if len(err)>2 and err[-1]>err[-2]:
        #    eta_Q = eta_Q/2
         #   eta_Theta = eta_Theta/2
        #    eta_c = eta_c/2
        #    eta_tau = eta_tau/2

        func = loss_factory(mode, kwargs)
        for i in range(n):
            tau = taus[i]
            vech_tt = vech_upper(np.outer(tau, tau))
            f = (
                c.flatten()
                + U @ tau
                + V @ (Theta.T @ vech_tt)
            )
            r = f - X[:, i]
            _, g = func(r) 
            #g = sign(r)
            #g = 2*r
            #g = (r)/np.linalg.norm(r,ord=2)

            g_list.append(g)
            f_list.append(f)
            r_errr.append(r)

        current_err = np.sum([func(e)[0] for e in r_errr])

        err.append(current_err)
        #print('iteration '+str(t)+": "+str(current_err)+'\n')

        # ---- Gradient w.r.t. Q ----
        grad_Q = np.zeros_like(Q)
        for i in range(n):
            tau = taus[i]
            vech_tt = vech_upper(np.outer(tau, tau))
            block = np.hstack([
                tau,
                Theta.T @ vech_tt
            ])
            grad_Q += np.outer(g_list[i], block)

        # Tangent projection
        grad_Q = grad_Q - Q @ sym(Q.T @ grad_Q)

        # Riemannian step + retraction
        Q_new = retract_stiefel(Q - eta_Q * grad_Q)

        if eval(X, taus, c, Q_new, Theta, d, s, func) > eval(X, taus, c, Q, Theta, d, s, func):
            eta_Q  = eta_Q/2
        else:
            Q = Q_new

        U = Q[:, :d]
        V = Q[:, d:d+s]

        # ---- Update Theta ----
        grad_Theta = np.zeros_like(Theta)
        for i in range(n):
            tau = taus[i]
            vech_tt = vech_upper(np.outer(tau, tau))
            grad_Theta += np.outer(
                vech_tt,
                V.T @ g_list[i]
            )
        Theta_new = Theta - eta_Theta * grad_Theta

        if eval(X, taus, c, Q, Theta_new, d, s, func) > eval(X, taus, c, Q, Theta, d, s, func):
            eta_Theta  = eta_Theta/2
        else:
            Theta = Theta_new

        # ---- Update c ----
        grad_c = np.sum(g_list, axis=0)
        c_new = c - eta_c * grad_c.reshape(-1, 1)

        if eval(X, taus, c_new, Q, Theta, d, s, func) > eval(X, taus, c, Q, Theta, d, s, func):
            eta_c  = eta_c/2
        else:
            c = c_new

        # ---- Update taus ----
        taus_new = taus
        for ss in range(5):
            for i in range(n):
                tau = taus[i]
                M_tau = build_M_tau(tau)
                N_tau = build_N_tau(tau)

                J_tau = U + V @ Theta.T @ (M_tau + N_tau)
                taus_new[i] = tau - eta_tau * (J_tau.T @ g_list[i])

        if eval(X, taus_new, c_new, Q, Theta, d, s, func) > eval(X, taus, c, Q, Theta, d, s, func):
            eta_tau  = eta_tau/2
        else:
            taus = taus_new


        if len(err)>20 and abs(err[-1]-err[-2]) < tol:
            break

    return Q, Theta, c, taus, err, [eta_Q, eta_Theta, eta_c, eta_tau]