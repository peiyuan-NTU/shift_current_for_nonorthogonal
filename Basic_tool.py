import numpy as np
from tbm import TBModel


def _get_overlap_derivative(sm, order, k):
    n_half_Rs = (sm.Rs.shape[1] + 1) // 2
    n_orbits = sm.norbits
    coefficients = (1j ** sum(order) *
                    np.prod(sm.Rcs ** order, axis=1) *
                    np.exp(1j * 2 * np.pi * (np.dot(k, sm.Rs.T))))
    tmp = np.reshape(sm.S @ coefficients[0, :n_half_Rs], (n_orbits, n_orbits))
    return tmp.T + tmp


# from functools import lru_cache
#
# @lru_cache(maxsize=None)
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


def get_hamiltonian_derivative(tbm: TBModel, order , k):
    norbits = tbm.norbits
    dH = np.zeros((norbits, norbits), dtype=np.complex128)

    for R, hopping in tbm.hoppings.items():
        Rc = tbm.lat @ R  # R in Cartesian coordinate
        phase = np.exp(1j * 2 * np.pi * np.dot(k, R))
        coeff = (1j * Rc[0]) ** order[0] * (1j * Rc[1]) ** order[1] * (1j * Rc[2]) ** order[2] * phase

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


def get_eigenvalues_for_tbm(tbm: TBModel, k):
    if tbm.isorthogonal:
        return np.linalg.eigvalsh(get_hamiltonian_for_k(tbm, k))
    if not tbm.isorthogonal:
        return np.linalg.eigvalsh(get_hamiltonian_for_k(tbm, k), b=tbm.overlaps[R0])
