import numpy as np
from Basic_tool import get_hamiltonian_derivative, get_eigenvalues_for_tbm, get_overlap_derivative, get_overlap

DEGEN_THRESH = [1.0e-4]


def getdAw(tm, α, order, k):
    dAw = np.zeros((tm.norbits, tm.norbits), dtype=np.complex128)
    for R, pos in tm.positions.items():
        Rc = tm.lat @ R
        phase = np.exp(1j * 2 * np.pi * (k @ R))
        coeff = (1j * Rc[0]) ** order[0] * (1j * Rc[1]) ** order[1] * (1j * Rc[2]) ** order[2] * phase
        dAw += coeff * pos[α]
    return dAw.T


def getdAwbar(tm, α, order, k):
    dAw = getdAw(tm, α, order, k)
    V = get_eigenvalues_for_tbm(tm, k).vectors
    return V.T @ dAw @ V


def getAwbar(tm, α, k):
    return getdAwbar(tm, α, (0, 0, 0), k)


def getdHbar(tm, order, k):
    dH = get_hamiltonian_derivative(tm, order, k)
    V = get_eigenvalues_for_tbm(tm, k).vectors
    return np.conj(V.T) @ dH @ V


def getdSbar(tm, order, k):
    dS = get_overlap_derivative(tm, order, k)
    V = get_eigenvalues_for_tbm(tm, k).vectors
    return np.conj(V.T) @ dS @ V


def get_order(alpha):
    order = [0, 0, 0]
    order[alpha - 1] += 1
    return tuple(order)


def getD(tm, alpha, k):
    order = get_order(alpha)
    dHbar = getdHbar(tm, order, k)
    dSbar = getdSbar(tm, order, k)
    Es = get_eigenvalues_for_tbm(tm, k)["values"]
    D = np.zeros((tm["norbits"], tm["norbits"]), dtype=np.complex128)
    Awbar = getAwbar(tm, alpha, k)

    for m in range(tm["norbits"]):
        for n in range(tm["norbits"]):
            En = Es[n]
            Em = Es[m]
            if abs(En - Em) > DEGEN_THRESH[1]:
                D[n, m] = (dHbar[n, m] - Em * dSbar[n, m]) / (Em - En)
            else:
                D[n, m] = 1j * Awbar[n, m]

    return D


def Berry_connection(tm, k, alpha):
    berry_connection = 1j * getD(tm, α, k) + getAwbar(tm, α, k)
    return berry_connection







