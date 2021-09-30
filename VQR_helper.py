import copy
import numpy as np
import gurobipy
import pandas as pd
import scipy


def VQRTp_aux(X, Y, U, mu, nu):
    n, d = Y.shape
    r = X.shape[1]
    m = U.shape[0]

    assert (n == len(X) or d == U.shape[2])

    x_bar = nu.T @ X
    c = -(U @ Y.T).T.flatten()
    A1 = scipy.sparse.kron(scipy.sparse.eye(n), np.ones((1, m)))
    A2 = scipy.sparse.kron(X.T, scipy.sparse.eye(m))
    f1 = nu.flatten()
    f2 = (mu @ x_bar).T.flatten()
    # e = np.ones(m * n)
    A = scipy.sparse.vstack([A1, A2])
    f = np.concatenate([f1, f2], axis=0)
    pi_init = (mu @ nu.T).flatten()

    model = gurobipy.Model()
    x = model.addMVar(c.shape, ub=1, name='x')
    x.start = pi_init
    model.addMConstr(A=A, x=x, sense='=', b=f)
    model.setMObjective(Q=None, constant=0, c=c, sense=gurobipy.GRB.MINIMIZE, xc=x)
    model.setParam(gurobipy.GRB.Param.NodefileStart, 16)
    model.setParam(gurobipy.GRB.Param.TimeLimit, 3600)  # 3600 sec = 1 hour
    # model.setParam(gurobipy.GRB.Param.Method, 2)
    model.optimize()

    if model.status == gurobipy.GRB.OPTIMAL or model.getAttr('SolCount') > 0:
        pivec = np.array(model.getAttr('x'))
        Lvec = np.array(model.getAttr('pi'))
    else:
        raise TimeoutError()

    pi = pivec.reshape(n,m).T
    L1 = Lvec[:n].reshape(1,n)
    L2 = Lvec[n:].reshape(r,m).T
    psi = -L1.T
    b = -L2
    val = (U.T @ pi @ Y).trace()

    return pi, psi, b, val

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def prepeareU2D(step, y_dim):
    T = np.arange(0, 1 + step, step)
    lT = len(T)
    m_prov = lT ** y_dim
    T_repeated = [T for _ in range(y_dim)]
    U_prov = cartesian_product(*T_repeated)
    mu_prov = np.ones((m_prov, 1)) * 1 / m_prov
    return m_prov, U_prov, mu_prov, T


def grad2D(f, T, U, step):
    EPS = 0.0001
    fact = 10 / step
    l = len(T)
    m = U.shape[0]
    D1, D2 = np.zeros((m, 1)), np.zeros((m, 1))
    #
    y_dim = U.shape[1]
    D = [np.zeros((m, 1)) for _ in range(y_dim)]

    idx = cartesian_product(*[np.array(list(range(1, l))) for _ in range(y_dim)])
    fact_vec = np.array([fact**i for i in range(y_dim)])
    for i in idx:
        u = np.array([T[ii] for ii in i])
        U_fact = fact_vec@U.T
        j = U_fact == fact_vec@u
        for ii in range(len(i)):
            tmp_u = copy.deepcopy(u)
            tmp_u[ii] -= step
            jprecii = (abs(U_fact - fact_vec@tmp_u) < EPS).nonzero()
            assert len(jprecii) == 1
            D[ii][j] = (f[j] - f[jprecii]) / step


    return D

def ComputeBetaEtAl2D(b_prov, T, U_prov, pi_prov, step):
    y_dim = U_prov.shape[1]
    r = b_prov.shape[1]
    n = pi_prov.shape[1]
    nonzind = ((U_prov[:, 0] != 0) & (U_prov[:, 1] != 0)).nonzero()[0]
    U = np.concatenate([U_prov[nonzind, 0].reshape(len(nonzind), 1), U_prov[nonzind, 1].reshape(len(nonzind), 1)], axis=1)
    m = U.shape[0]
    mu = np.ones((m,1)) / m
    betas = [np.zeros((m, r)) for _ in range(y_dim)]
    for k in range(r):
        D = grad2D(b_prov[:, k], T, U_prov, step)

        for ii in range(y_dim):
            betas[ii][:, k] = D[ii][nonzind, 0]


    pi = np.zeros((m, n))
    for i in range(n):
        pi[:, i] = pi_prov[nonzind, i]

    b = b_prov[nonzind]
    return betas, U, m, mu, pi, b


def VQRTp(X, Y, results_dir):
    n = len(X)
    step = 0.1
    nu = np.ones((X.shape[0], 1)) / n
    _, U, mu, T = prepeareU2D(step, Y.shape[1])
    pi, psi, b, val = VQRTp_aux(X, Y, U, mu, nu)
    betas, U, m, mu, pi, b = ComputeBetaEtAl2D(b, T, U, pi, step)

    for ii in range(Y.shape[1]):
        pd.DataFrame(betas[ii]).to_csv(f"{results_dir}/beta{ii+1}.csv")

    pd.DataFrame(U).to_csv(f"{results_dir}/U.csv")

