import numpy as np
from src.Basic_tool import get_hamiltonian_derivative, get_eigen_for_tbm, get_overlap_derivative, construct_line_kpts, \
    get_overlap, get_order
import scipy.linalg as la

DEGEN_THRESH = [1.0e-4]


def get_dAw(tm, alpha, order, k):
    if alpha not in range(1, 4):
        raise ValueError("alpha incorrect!")
    dAw = np.zeros((tm.norbits, tm.norbits), dtype=np.complex128)
    for R, pos in tm.positions.items():

        # R in Cartesian coordinate
        Rc = tm.lat @ R

        # e^i(kR)*2pi
        phase = np.exp(1j * 2 * np.pi * (k @ R))

        coeff = (1j * Rc[0]) ** order[0] * (1j * Rc[1]) ** order[1] * (1j * Rc[2]) ** order[2] * phase

        dAw += coeff * pos[alpha - 1]
        # print("R", R, "pos", pos, "coeff", coeff)
    return dAw.conj().T


def get_Aw(tm, alpha, k):
    if alpha not in range(1, 4):
        raise ValueError("alpha incorrect!")
    return get_dAw(tm, alpha, (0, 0, 0), k)


def get_dAwbar(tm, alpha, order, k):
    if alpha not in range(1, 4):
        raise ValueError("alpha incorrect!")
    dAw = get_dAw(tm, alpha, order, k)
    V = get_eigen_for_tbm(tm, k)[1]
    return V.conj().T @ dAw @ V


def get_Awbar(tm, alpha, k):
    if alpha not in range(1, 4):
        raise ValueError("alpha incorrect!")
    return get_dAwbar(tm, alpha, (0, 0, 0), k)


def get_dHbar(tm, order, k):
    dH = get_hamiltonian_derivative(tm, order, k)
    # print("k", k)
    # print("dHsssssss", get_eigenvalues_for_tbm(tm, k))
    V = get_eigen_for_tbm(tm, k)[1]
    # return np.conj(V.T) @ dH @ V
    return V.conj().T @ dH @ V


def get_dSbar(tm, order, k):
    dS = get_overlap_derivative(tm, order, k)
    V = get_eigen_for_tbm(tm, k)[1]
    # return np.conj(V.T) @ dS @ V
    return V.conj().T @ dS @ V


def get_D(tm, alpha, k):
    if alpha not in range(1, 4):
        raise ValueError("alpha incorrect!")
    order = get_order(alpha)
    dHbar = get_dHbar(tm, order, k)
    dSbar = get_dSbar(tm, order, k)
    Es = get_eigen_for_tbm(tm, k)[0]
    D = np.zeros((tm.norbits, tm.norbits), dtype=np.complex128)
    Awbar = get_Awbar(tm, alpha, k)

    for m in range(tm.norbits):
        for n in range(tm.norbits):
            En = Es[n]
            Em = Es[m]
            if abs(En - Em) > DEGEN_THRESH[0]:
                D[n, m] = (dHbar[n, m] - Em * dSbar[n, m]) / (Em - En)
            else:
                D[n, m] = 1j * Awbar[n, m]

    return D


def Berry_connection(tm, k, alpha):
    if alpha not in range(1, 4):
        raise ValueError("alpha incorrect!")
    berry_connection = 1j * get_D(tm, alpha, k) + get_Awbar(tm, alpha, k)
    return berry_connection


# def get_wilson_spectrum(tm, k, alpha):
#     berry_connection = Berry_connection(tm, k, alpha)
#     Es = get_eigenvalues_for_tbm(tm, k)["values"]
#     return np.angle(np.linalg.det(berry_connection) * np.exp(1j * Es))
def parallel_transport(tm, getU, Ustart, kpaths, ndiv):
    kpts = construct_line_kpts(kpaths, ndiv)
    result = Ustart
    for i in range(kpts.shape[1] - 1):
        k1, k2 = kpts[:, i], kpts[:, i + 1]
        delta_k = k2 - k1
        # print("atm.rlat", tm.rlat)
        # print("delta_k", delta_k)
        S = get_overlap(tm, k1)
        U = getU(k2)
        print("get_Aw(atm, x, k1) for x in range(1, 4)", [get_Aw(tm, x, k1) for x in range(1, 4)])
        print("atm.rlat @ delta_k", tm.rlat @ delta_k)
        tmp1 = [get_Aw(tm, x, k1) for x in range(1, 4)]
        tmp2 = tm.rlat @ delta_k
        tmp3 = [tmp1[i] * tmp2[i] for i in range(3)]
        # print(tmp3)

        # Aw_delta_k = np.sum([get_Aw(tm, x, k1) for x in range(1, 4)] * (tm.rlat @ delta_k))
        Aw_delta_k = tmp3[0] + tmp3[1] + tmp3[2]
        print("Aw_delta_k", Aw_delta_k)
        tmp = -1j * Aw_delta_k + S
        result = U @ U.conj().T @ tmp.conj().T @ result
        print("U", U, "U.T", U.T, "tmp.T", tmp.T, "result", result)
        # result = U @ U.T @ tmp.T @ result
        print("result", result, "\n")
    return result


def get_wilson_spectrum_inner(tm, getU, kpaths, ndiv):
    kpts = construct_line_kpts(kpaths, ndiv)

    # print("kpts", kpts[:, 0])
    U_start = getU(kpts[:, 0])
    print("U_start", U_start)
    # print("get_overlap(tm, kpts[:, -1])", get_overlap(tm, kpts[:, -1]))
    # print("parallel_transport(tm, getU, U_start, kpaths, ndiv)", parallel_transport(tm, getU, U_start, kpaths, ndiv))
    W = U_start.conj().T @ get_overlap(tm, kpts[:, -1]) @ parallel_transport(tm, getU, U_start, kpaths, ndiv)
    # W = U_start.T @ get_overlap(tm, kpts[:, -1]) @ parallel_transport(tm, getU, U_start, kpaths, ndiv)
    print("W", W)
    tmp = np.linalg.eigvals(W)
    tmp = la.eig(W)[0]
    err = np.max(np.abs(tmp) - 1.0)
    if err > 0.01:
        print(f"W is not unitary! Error: {err}")
    result = np.sort(np.imag(np.log(tmp)))
    return result


# @property
# @pysnooper.snoop("./log/debug.log", prefix="--*--")
def get_wilson_spectrum(tm, band_indices, kpaths, ndiv):
    def get_U(k):
        # print(k)
        eig_result = get_eigen_for_tbm(tm, k)
        # print("eig_result", eig_result)
        return eig_result[1][:, band_indices]

    result = get_wilson_spectrum_inner(tm, get_U, kpaths, ndiv)
    return result


def get_dEs(tm, alpha, k):
    if alpha not in range(1, 4):
        raise ValueError("alpha incorrect!")

    order = get_order(alpha)
    dHbar = get_dHbar(tm, order, k)
    dSbar = get_dSbar(tm, order, k)
    Es = get_eigen_for_tbm(tm, k)[0]
    dEs = np.zeros(tm.norbits)
    for n in range(tm.norbits):
        dEs[n] = np.real(dHbar[n, n] - Es[n] * dSbar[n, n])
    return dEs
