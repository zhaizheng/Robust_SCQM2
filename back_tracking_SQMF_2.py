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

def eval_each(X, tau, c, Q, Theta, d, s, func):
    U = Q[:, :d]
    V = Q[:, d:d+s]
    vech_tt = vech_upper(np.outer(tau, tau))
    f = (
        c.flatten()
        + U @ tau
        + V @ (Theta.T @ vech_tt)
    )
    r = f - X

    return func(r)[0]



def update_error(func, c, U, V, X, Theta, taus):
    g_list = []
    f_list = []
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
        _, g = func(r) 
        #g = sign(r)
        #g = 2*r
        #g = (r)/np.linalg.norm(r,ord=2)

        g_list.append(g)
        f_list.append(f)
        r_errr.append(r)
    loss = np.sum([func(e)[0] for e in r_errr])
    return g_list, f_list, r_errr, loss

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
    func = loss_factory(mode, kwargs)

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
        g_list, f_list, r_errr, loss = update_error(func, c, U, V, X, Theta, taus)

        err.append(loss)
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

        # Riemannian gradient
        grad_Q = grad_Q - Q @ sym(Q.T @ grad_Q)
        desc_Q = -grad_Q

        def obj_Q(Qcand):
            return eval(X, taus, c, Qcand, Theta, d, s, func)

        eta_Q_bt = eta_Q
        for _ in range(20):
            Q_candidate = retract_stiefel(Q + eta_Q_bt * desc_Q)
            if obj_Q(Q_candidate) <= obj_Q(Q) - 1e-2 * eta_Q_bt * np.linalg.norm(desc_Q)**2:
                break
            eta_Q_bt *= 0.5

        Q = Q_candidate
        U = Q[:, :d]
        V = Q[:, d:d+s]

        # ---- Update Theta (with backtracking) ----
        grad_Theta = np.zeros_like(Theta)
        for i in range(n):
            tau = taus[i]
            vech_tt = vech_upper(np.outer(tau, tau))
            grad_Theta += np.outer(vech_tt, V.T @ g_list[i])

        desc_Theta = -grad_Theta

        def obj_Theta(Thetacand):
            return eval(X, taus, c, Q, Thetacand, d, s, func)

        eta_bt = eta_Theta
        for _ in range(20):
            Theta_candidate = Theta + eta_bt * desc_Theta
            if obj_Theta(Theta_candidate) <= obj_Theta(Theta) - 1e-2 * eta_bt * np.linalg.norm(desc_Theta)**2:
                break
            eta_bt *= 0.5

        Theta = Theta_candidate

        # ---- Update c (with backtracking) ----
        grad_c = np.sum(g_list, axis=0).reshape(-1, 1)
        desc_c = -grad_c

        def obj_c(ccand):
            return eval(X, taus, ccand, Q, Theta, d, s, func)

        eta_bt = eta_c
        for _ in range(20):
            c_candidate = c + eta_bt * desc_c
            if obj_c(c_candidate) <= obj_c(c) - 1e-2 * eta_bt * np.linalg.norm(desc_c)**2:
                break
            eta_bt *= 0.5

        c = c_candidate

        # ---- Update taus (block backtracking) ----
        #taus_candidate = [tau.copy() for tau in taus]
        eta_tau_t = eta_tau
    
        for i in range(n):
            tau = taus[i]
            M_tau = build_M_tau(tau)
            N_tau = build_N_tau(tau)
            J_tau = U + V @ Theta.T @ (M_tau + N_tau)
            total_dec = np.linalg.norm(J_tau.T @ g_list[i])
            for _ in range(20):
                tau_candidate_i = tau - eta_tau_t * (J_tau.T @ g_list[i])
                if eval_each(X[:,i], tau_candidate_i, c, Q, Theta, d, s, func) <= eval_each(X[:,i], tau, c, Q, Theta, d, s, func)- 1e-2 * eta_tau_t * total_dec:
                    break
                else:
                    eta_tau_t *= 0.5
            taus[i] = tau_candidate_i


        if len(err)>20 and abs(err[-1]-err[-2]) < tol:
            break

    return Q, Theta, c, taus, err, [eta_Q, eta_Theta, eta_c, eta_tau]