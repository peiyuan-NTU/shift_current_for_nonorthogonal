from typing import Dict, List, Union, Tuple
import numpy as np
from typing import Dict, List, Optional, Union
import numpy as np

# from scipy.spatial.transform import Rotation as R


R0 = (0, 0, 0)
IMAG_TOL = 1.0e-4


def R2minus_R(R):
    return tuple(map(lambda x: -x, R))


class TBModel:
    def __init__(
            self,
            norbits: int,
            lat: np.ndarray,
            rlat: np.ndarray,
            hoppings: Dict[Tuple[int, int, int], np.ndarray],
            overlaps: Dict[Tuple[int, int, int], np.ndarray],
            # positions: Dict[Tuple[int, int, int], list[np.ndarray]],
            positions: Dict[Tuple[int, int, int], list],
            isorthogonal: bool,
            nsites: Union[int, None] = None,
            site_norbits: Union[List[int], None] = None,
            site_positions: Union[np.ndarray, None] = None,
            orbital_types: Union[List[List[int]], None] = None,
            isspinful: Union[bool, None] = None,
            is_canonical_ordered: Union[bool, None] = None
            fermi_energy: Union[float, None] = None
    ):
        self.norbits = norbits
        self.lat = lat
        self.rlat = rlat
        self.hoppings = hoppings
        self.overlaps = overlaps
        self.positions = positions
        self.isorthogonal = isorthogonal
        self.nsites = nsites
        self.site_norbits = site_norbits
        self.site_positions = site_positions
        self.orbital_types = orbital_types
        self.isspinful = isspinful
        self.is_canonical_ordered = is_canonical_ordered
        self.H_ftn58 = None
        self.S_ftn58 = None
        self.fermi_energy = fermi_energy

    def has_full_information(self):
        for field in ["nsites", "site_norbits", "site_positions", "orbital_types",
                      "isspinful", "is_canonical_ordered"]:
            if getattr(self, field, None) is None:
                return False
        return True

    def to_site_index(self, n):
        # l is
        l = [self.site_norbits[i] // (1 + self.isspinful) for i in range(self.nsites)]
        N = self.norbits // (1 + self.isspinful)
        s = 0
        if n > N:
            n -= N
            s = 1
        i = 0
        ss = 0
        for cnt in range(len(l)):
            ss += l[cnt]
            if n <= ss:
                i = cnt
                break
        return i + 1, n - sum(l[0:i]) + s * l[i]

    def to_orbital_index(self, site, site_orbital):
        # i, n = i_n_tuple
        i = site
        n = site_orbital
        if self.isspinful:
            # tmp1 = sum(self.site_norbits[:i - 1]) // 2
            tmp1 = sum(self.site_norbits[:i]) // 2
            tmp2 = self.site_norbits[i] // 2
            tmp3 = self.norbits // 2

            if n > tmp2:
                return tmp3 + tmp1 + n - tmp2
            else:
                return tmp1 + n
        else:
            return sum(self.site_norbits[:i]) + n


    """
    ⟨0n|``r_α``|Rm⟩ 
    position of orbital n in unit cell R in direction α
    """

    def set_position(self, R, n, m, alpha, pos, position_tolerance=1.0e-4):
        if not (0 <= n <= self.norbits and 0 <= m <= self.norbits):
            raise ValueError("n or m not in range 1-norbits.")
        if len(R) != 3:
            raise ValueError("R should be a 3-element vector")
        if not (1 <= alpha <= 3):
            raise ValueError("alpha not in 1-3.")
        if R == R0 and n == m and abs(pos.imag) > IMAG_TOL:
            raise ValueError("Position of one orbital with itself should be real.")
        if R == R0 and n == m and self.has_full_information():
            i = self.to_site_index(self, n)[0]
            if not np.isclose(self.site_positions[alpha - 1, i] * self.overlaps[R0][n, n], pos,
                              atol=position_tolerance):
                raise ValueError("pos incompatible with site_positions and overlaps.")

        if R not in self.positions:
            self.positions[R] = [np.zeros((self.norbits, self.norbits)) for _ in range(3)]
            self.positions[R2minus_R(R)] = [np.zeros((self.norbits, self.norbits)) for _ in range(3)]

        if R == R0 and n == m:
            self.positions[R][alpha - 1][n, m] = pos.real
        else:
            self.positions[R][alpha - 1][n, m] = pos
            tmp = self.overlaps.get(R2minus_R(R), np.zeros((self.norbits, self.norbits)))[m, n]
            # print("tmp", tmp)
            # print("pos", pos)
            # print("lat[alpha - 1]", self.lat[alpha - 1])
            # print("lat*tmp", self.lat[alpha - 1] * tmp)
            # print("np.conj(pos)-self.lat[alpha-1]*tmp", np.conj(pos) - self.lat[alpha - 1] * tmp)
            # print(type(np.conj(pos) - self.lat[alpha - 1] * tmp))
            self.positions[R2minus_R(R)][alpha - 1][m, n] = np.conj(pos) - (self.lat @ R)[alpha - 1] * tmp

    """
    ⟨0n|H|Rm⟩ 
    Hopping between orbitals n and m on sites 0 and R.
    """

    def set_hopping(self, R, n, m, hopping):
        if not (0 <= n <= self.norbits) or not (0 <= m <= self.norbits):
            raise ValueError("n or m not in range 1-norbits.")
        if len(R) != 3:
            raise ValueError("R should be a 3-element vector")

        if R == R0 and n == m and abs(hopping.imag) > IMAG_TOL:
            raise ValueError("On site energy should be real.")

        if R not in self.hoppings:
            # print("R not in self.hoppings", R)
            self.hoppings[R] = np.zeros((self.norbits, self.norbits), dtype=type(hopping))
            self.hoppings[R2minus_R(R)] = np.zeros((self.norbits, self.norbits), dtype=type(hopping))

        if R == R0 and n == m:
            self.hoppings[R][n, m] = hopping.real
        else:
            # print("R ")
            self.hoppings[R][n, m] = hopping
            self.hoppings[R2minus_R(R)][m, n] = np.conj(hopping)

    """
    ⟨0n|Rm⟩ 
    Overlap between orbitals n and m on sites 0 and R.
    """

    def set_overlap(self, R, n, m, overlap):
        if not (0 <= n <= self.norbits) or not (0 <= m <= self.norbits):
            raise ValueError("n or m not in range 1-norbits.")
        if len(R) != 3:
            raise ValueError("R should be a 3-element vector")
        if self.isorthogonal:
            raise ValueError("self is orthogonal.")
        if R == R0 and n == m:
            if abs(overlap) < 0.1:
                raise ValueError("An orbital should have substantial overlap with itself.")
            if abs(overlap.imag) > IMAG_TOL:
                raise ValueError("Overlap of one orbital with itself should be real.")

        if R not in self.overlaps:
            self.overlaps[R] = np.zeros((self.norbits, self.norbits), dtype=type(overlap))
            self.overlaps[R2minus_R(R)] = np.zeros((self.norbits, self.norbits), dtype=type(overlap))

        if R == R0 and n == m:
            self.overlaps[R][n, m] = overlap.real
        else:
            self.overlaps[R][n, m] = overlap
            self.overlaps[R2minus_R(R)][m, n] = np.conj(overlap)

        if R == R0 and n == m and self.has_full_information():
            i = self.to_site_index(self, n)[0]
            self.set_position(self, R0, n, n, self.site_positions[:, i] * self.overlaps[R0][n, n])

    def add_hopping_inner(self, R, n, m, hopping):
        # print("11111111111111111111111111111111111111111")

        norbits = self.norbits
        if n not in range(norbits) or m not in range(norbits):
            raise ValueError("n or m not in range 1-norbits.")

        if len(R) != 3:
            raise ValueError("R should be a 3-element vector.")

        # n, m = n - 1, m - 1
        if R == R0 and n == m and np.imag(hopping) > 1e-10:
            raise ValueError("On site energy should be real.")

        if R not in self.hoppings:
            self.hoppings[R] = np.zeros((norbits, norbits), dtype=complex)
            self.hoppings[R2minus_R(R)] = np.zeros((norbits, norbits), dtype=complex)

        if R == R0 and n == m:
            self.hoppings[R][n, m] += np.real(hopping)
        else:
            self.hoppings[R][n, m] += hopping
            self.hoppings[R2minus_R(R)][m, n] += np.conj(hopping)
        # print("self.hoppings[R]", self.hoppings[R])
        # print("\n")

    def add_hopping(self, R, i_p, j_q, hopping):
        # print("444444444444444444444444444444444444444444444444444444444444444444")

        if not self.has_full_information():
            raise ValueError("No site information is provided in the model.")

        i, p = i_p
        j, q = j_q
        i, j, p, q = i - 1, j - 1, p - 1, q - 1

        if i not in range(self.nsites) or j not in range(self.nsites):
            raise ValueError("i or j not in range 1-nsites.")

        if p not in range(self.site_norbits[i]) or q not in range(self.site_norbits[j]):
            raise ValueError("n or m not in range 1-site_norbits.")

        # p, q = p - 1, q - 1
        if len(R) != 3:
            raise ValueError("R should be a 3-element vector.")
        # print("R", R, "i_p", i_p, "j_q", j_q, "hopping", hopping, "self._to_orbital_index((i, p))",
        #       self._to_orbital_index((i, p)), "self._to_orbital_index((j, q))", self._to_orbital_index((j, q)))
        self.add_hopping_inner(R, self.to_orbital_index(i, p), self.to_orbital_index(j, q), hopping)


def create_TBModel(n_orbits: int, lat: np.ndarray, orthogonal=True):
    if lat.shape != (3, 3):
        raise ValueError("Size of lat is not correct.")
    rlat = 2 * np.pi * np.linalg.inv(lat).T
    overlaps = {' '.join(list(map(lambda x: str(x), list(R0)))): np.eye(n_orbits, dtype=float)}
    return TBModel(n_orbits, lat, rlat, {}, {}, {}, orthogonal, None, None, None, None, None, overlaps)


def create_info_missing_tb_model(lat: np.ndarray, site_positions, orbital_types, isspinful=False, isorthogonal=True,
                                 is_canonical_ordered=False):
    if lat.shape != (3, 3):
        raise ValueError("Size of lat is not correct.")

    rlat = 2 * np.pi * np.linalg.inv(lat).T
    n_sites = site_positions.shape[1]
    n_spins = 1 + isspinful
    site_norbits = np.array([sum(2 * np.array(orbital_types[i]) + 1) for i in range(n_sites)]) * n_spins
    n_orbits = sum(site_norbits)

    """
    n_orbit
        number of orbitals in the system
    lat
        lattice vectors
    rlat
        reciprocal lattice vectors
    hoppings
        hopping matrices, keys are R vectors
    positions
        position matrices, keys are R vectors
    overlaps
        overlap matrices, keys are R vectors
    isorthogonal
        whether the bases are orthogonal
    nsites
        number of atoms in the system
    site_norbits
        number of orbitals on each atom
    site_positions
        
    isspinful
        whether the system is spinful
    is_canonical_ordered
        whether the orbitals are in canonical order
        
    """
    tm = TBModel(norbits=n_orbits,
                 lat=lat,
                 rlat=rlat,
                 hoppings=dict(),
                 positions=dict(),
                 overlaps=dict(),
                 isorthogonal=isorthogonal,
                 nsites=n_sites,
                 site_norbits=site_norbits,
                 site_positions=site_positions,
                 orbital_types=orbital_types,
                 isspinful=isspinful,
                 is_canonical_ordered=is_canonical_ordered)
    R0 = (0, 0, 0)
    tm.overlaps[R0] = np.eye(n_orbits)
    tm.positions[R0] = [np.zeros((n_orbits, n_orbits), dtype=complex) for _ in range(3)]

    for i in range(n_sites):
        for p in range(site_norbits[i]):
            for alpha in range(3):
                # n = tm.to_orbital_index((i, p))
                n = tm.to_orbital_index(site=i,
                                        site_orbital=p)
                # print("n", n)
                tm.positions[R0][alpha][n, n] = site_positions[alpha, i]

    return tm

