import numpy as np
from src.tight_binding_model import TBModel
import scipy.linalg as la

# import

DEGEN_THRESH = [1.0e-4]


# def _get_overlap_derivative(sm, order, k):
#     Rs = len(sm.hoppings.keys())
#     n_half_Rs = (Rs.shape[1] + 1) // 2
#     n_orbits = sm.norbits
#     coefficients = (1j ** sum(order) *
#                     np.prod(sm.Rcs ** order, axis=1) *
#                     np.exp(1j * 2 * np.pi * (np.dot(k, sm.Rs.T))))
#     tmp = np.reshape(sm.S @ coefficients[0, :n_half_Rs], (n_orbits, n_orbits))
#     return tmp.T + tmp

def _get_overlap_derivative(nm, order, k):
    dS = np.zeros((nm.norbits, nm.norbits), dtype=np.complex128)
    for R, overlap in nm.overlaps.items():
        Rc = nm.lat @ R  # R in Cartesian coordinate
        phase = np.exp(1j * 2 * np.pi * np.dot(k, R))
        coeff = (1j * Rc[0]) ** order[0] * (1j * Rc[1]) ** order[1] * (1j * Rc[2]) ** order[2] * phase
        dS += coeff * overlap
    return dS


def get_overlap_derivative(tbm, order, k):
    if tbm.isorthogonal:
        return np.eye(tbm.norbits, dtype=complex) if order == (0, 0, 0) else np.zeros((tbm.norbits, tbm.norbits),
                                                                                      dtype=complex)
    else:
        return _get_overlap_derivative(tbm, order, k)


def get_overlap(tbm, k):
    return get_overlap_derivative(tbm, (0, 0, 0), k)


def make_hermitian(matrix: np.ndarray):
    return (matrix + matrix.conj().T) / 2


def get_hamiltonian_derivative(tbm: TBModel, order, k):
    norbits = tbm.norbits
    dH = np.zeros((norbits, norbits), dtype=np.complex128)

    for R, hopping in tbm.hoppings.items():
        Rc = tbm.lat @ R  # R in Cartesian coordinate
        # print("k", k, "R", R, "Rc", Rc)
        phase = np.exp(1j * 2 * np.pi * np.dot(k, R))
        # print("phase", phase)
        coeff = (1j * Rc[0]) ** order[0] * (1j * Rc[1]) ** order[1] * (1j * Rc[2]) ** order[2] * phase
        # print("coeff", coeff)
        # print("hopping", hopping)
        # print("\n")
        dH += coeff * hopping

    return dH


def get_hamiltonian_for_k(tbm: TBModel, k):
    # norbits = tbm.norbits
    # H = np.zeros((norbits, norbits), dtype=np.complex128)
    #
    # for R, hopping in tbm.hoppings.items():
    #     phase = np.exp(1j * 2 * np.pi * np.dot(k, R))
    #     H += phase * hopping

    return get_hamiltonian_derivative(tbm, [0, 0, 0], k)


def get_eigen_for_tbm(tbm: TBModel, k):
    # print("k", k)
    H = get_hamiltonian_for_k(tbm, k)
    # H = H.T
    # print("H", H)
    if tbm.isorthogonal:
        return la.eigh(make_hermitian(H))
    if not tbm.isorthogonal:
        S = get_overlap(tbm, k)
        # print("H", make_hermitian(H))
        # print("S", make_hermitian(S))
        return la.eigh(a=make_hermitian(H), b=make_hermitian(S))


def construct_line_kpts(kpath, pnkpts, connect_end_points=False):
    if kpath.shape[0] != 3:
        raise ValueError("kpath.shape[0] should be 3.")
    if not connect_end_points and kpath.shape[1] % 2 != 0:
        raise ValueError("kpath.shape[1] should be even if connect_end_points is False.")

    if connect_end_points:
        nkpath = np.zeros((3, 2 * kpath.shape[1] - 2))
        for i in range(kpath.shape[1] - 1):
            nkpath[:, 2 * i] = kpath[:, i]
            nkpath[:, 2 * i + 1] = kpath[:, i + 1]
    else:
        nkpath = kpath

    npath = nkpath.shape[1] // 2
    kpts = np.zeros((3, npath * pnkpts))

    for ipath in range(npath):
        kstart = nkpath[:, 2 * ipath]
        kend = nkpath[:, 2 * ipath + 1]
        dk = (kend - kstart) / (pnkpts - 1)
        for ikpt in range(pnkpts):
            kpts[:, ikpt + ipath * pnkpts] = kstart + ikpt * dk

    return kpts


def construct_kpts_for_vasp(kpath, nkpts, connect_end_points=False):
    assert kpath.shape[0] == 3, "kpath.shape[0] should be 3."
    assert kpath.shape[1] % 2 == 0, "kpath.shape[1] should be even."

    k_points = np.zeros((3, int(nkpts * kpath.shape[1]/2)))
    for i in range(int(kpath.shape[1] / 2)):
        k_points[0, nkpts * i:nkpts * (i + 1)] = np.linspace(start=kpath[0, 2 * i],
                                                             stop=kpath[0, 2 * i + 1],
                                                             num=nkpts,
                                                             endpoint=False)
        k_points[1, nkpts * i:nkpts * (i + 1)] = np.linspace(start=kpath[1, 2 * i],
                                                             stop=kpath[1, 2 * i + 1],
                                                             num=nkpts,
                                                             endpoint=False)
        k_points[2, nkpts * i:nkpts * (i + 1)] = np.linspace(start=kpath[2, 2 * i],
                                                             stop=kpath[2, 2 * i + 1],
                                                             num=nkpts,
                                                             endpoint=False)
    return k_points


def get_order(alpha):
    # print("alpha", alpha)
    order = [0, 0, 0]
    order[alpha - 1] += 1
    return tuple(order)


def get_two_order(alpha, beta):
    order = [0, 0, 0]
    order[alpha - 1] += 1
    order[beta - 1] += 1
    return tuple(order)
