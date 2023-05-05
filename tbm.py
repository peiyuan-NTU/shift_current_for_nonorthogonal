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
            positions: Dict[Tuple[int, int, int], list[np.ndarray]],
            isorthogonal: bool,
            nsites: Union[int, None] = None,
            site_norbits: Union[List[int], None] = None,
            site_positions: Union[np.ndarray, None] = None,
            orbital_types: Union[List[List[int]], None] = None,
            isspinful: Union[bool, None] = None,
            is_canonical_ordered: Union[bool, None] = None
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

    # class TBModel:
    #     def __init__(self,
    #                  norbits: int,
    #                  lat: np.ndarray,
    #                  rlat: np.ndarray,
    #                  hoppings: Dict[np.ndarray, np.ndarray],
    #                  overlaps: Dict[np.ndarray, np.ndarray],
    #                  positions: Dict[np.ndarray, List[np.ndarray]],
    #                  isorthogonal: bool,
    #                  nsites: Optional[int] = None,
    #                  site_norbits: Optional[np.ndarray] = None,
    #                  site_positions: Optional[np.ndarray] = None,
    #                  orbital_types: Optional[List[List[int]]] = None,
    #                  isspinful: Optional[bool] = None,
    #                  is_canonical_ordered: Optional[bool] = None):
    #         self.norbits = norbits
    #         self.lat = lat
    #         self.rlat = rlat
    #         self.hoppings = hoppings
    #         self.overlaps = overlaps
    #         self.positions = positions
    #         self.isorthogonal = isorthogonal
    #         self.nsites = nsites
    #         self.site_norbits = site_norbits
    #         self.site_positions = site_positions
    #         self.orbital_types = orbital_types
    #         self.isspinful = isspinful
    #         self.is_canonical_ordered = is_canonical_ordered

    def has_full_information(self):
        for field in ["nsites", "site_norbits", "site_positions", "orbital_types",
                      "isspinful", "is_canonical_ordered"]:
            if getattr(self, field, None) is None:
                return False
        return True

    def _to_site_index(self, n):
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
        return (i + 1, n - sum(l[0:i]) + s * l[i])

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
            i = self._to_site_index(self, n)[0]
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
            i = self._to_site_index(self, n)[0]
            self.set_position(self, R0, n, n, self.site_positions[:, i] * self.overlaps[R0][n, n])

    def add_hopping(self,):


def create_TBModel(norbits: int, lat: np.ndarray, isorthogonal=True):
    # print(lat.shape)
    if lat.shape != (3, 3):
        raise ValueError("Size of lat is not correct.")
    rlat = 2 * np.pi * np.linalg.inv(lat).T
    # overlaps = {R0: np.eye(norbits)}
    overlaps = {' '.join(list(map(lambda x: str(x), list(R0)))): np.eye(norbits, dtype=float)}
    return TBModel(norbits, lat, rlat, {}, {}, {}, isorthogonal, None, None, None, None, None, overlaps)


def create_info_missing_tb_model(lat: np.ndarray, orbital_positions, isorthogonal=True):
    if lat.shape != (3, 3):
        raise ValueError("Size of lat is not correct.")
    if orbital_positions.shape[0] != 3:
        raise ValueError("orbital_positions should have three rows.")

    norbits = orbital_positions.shape[1]
    rlat = 2 * np.pi * np.linalg.inv(lat).T

    # self.nsites = nsites
    # self.site_norbits = site_norbits
    # self.site_positions = site_positions
    # self.orbital_types = orbital_types
    # self.isspinful = isspinful
    # self.is_canonical_ordered = is_canonical_ordered

    tm = TBModel(norbits, lat, rlat, hoppings=dict(), positions=dict(), overlaps=dict()
                 , isorthogonal=isorthogonal, nsites=None, site_norbits=None, site_positions=None, orbital_types=None,
                 isspinful=None, is_canonical_ordered={})

    tm.overlaps[R0] = np.eye(norbits)

    for n in range(norbits):
        for alpha in range(1, 4):
            tm.set_position(R0, n, n, alpha, orbital_positions[alpha-1, n])
    return tm

# def set_position(tm, R, n, m, α, val):
#     if R not in tm["positions"]:
#         tm["positions"][R] = np.zeros((tm["norbits"], tm["norbits"], 3))
#     tm["positions"][R][n, m, α] = val



###


# def R2str(R):
#     return ' '.join(list(map(lambda x: str(x), list(R))))
