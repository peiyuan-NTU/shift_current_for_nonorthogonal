import numpy as np

from data.parse_ftn58 import read_ftn58
from src.tight_binding_model import create_TBModel

ftn58_dict = read_ftn58()
n_orbital = ftn58_dict["n_orbital"]
supercell_vector = ftn58_dict["supercell_vector"]
hopping = ftn58_dict["hopping"]
orbital_index = ftn58_dict["orbital_index"]
lattice_vector = ftn58_dict["lattice_vector"]
full_info = ftn58_dict["full_info"]
atom_and_orbital = ftn58_dict["atom_and_orbital"]
orbital2atom_orb = ftn58_dict["orbital2atom_orb"]

# atom_pos = [i[1][0] for i in full_info["Ainfo"][0][0][0]]
# atom_pos = np.transpose(atom_pos)

atom_pos = np.array([[0.143999989000000, 0.143999989000000, 0.143999989000000],
                     [0.356000011000000, 0.855999960000000, 0.643999989000000],
                     [0.855999960000000, 0.643999989000000, 0.356000011000000],
                     [0.643999989000000, 0.356000011000000, 0.855999960000000],
                     [0.839999961000000, 0.839999961000000, 0.839999961000000],
                     [0.660000039000000, 0.160000026000000, 0.339999910000000],
                     [0.160000026000000, 0.339999910000000, 0.660000039000000],
                     [0.339999910000000, 0.660000039000000, 0.160000026000000]]).T

import numpy as np


# def createmodelwannier(tbfile, wsvecfile):
#     # read wsvec file
#     wsvecs = {}
#     wsndegen = {}
#
#     with open(wsvecfile, 'r') as f:
#         next(f)
#         for line in f:
#             if line.strip() == '':
#                 break
#             key = list(map(int, line.split()))
#             ndegen = int(next(f))
#             vecs = [list(map(int, next(f).split())) for _ in range(ndegen)]
#             wsvecs[tuple(key)] = vecs
#             wsndegen[tuple(key)] = ndegen
#
#     lat = np.zeros((3, 3))
#     norbits = 0
#     hoppings = {}
#     positions = {}
#
#     with open(tbfile, 'r') as f:
#         next(f)
#         for i in range(3):
#             lat[:, i] = list(map(float, next(f).split()))
#         norbits = int(next(f))
#         nrpts = int(next(f))
#         rndegen = []
#         for line in f:
#             if line.strip() == '':
#                 break
#             rndegen.extend(list(map(int, line.split())))
#         assert len(rndegen) == nrpts
#
#         # other parts of the function
#
#     # instantiate a TBModel object
#     tm = TBModel(norbits, lat, isorthogonal=True)
#
#     for R, hopping in hoppings.items():
#         for m in range(norbits):
#             for n in range(norbits):
#                 sethopping(tm, R, n, m, hopping[n, m])
#
#     for R, position in positions.items():
#         for m in range(norbits):
#             for n in range(norbits):
#                 for α in range(3):
#                     setposition(tm, R, n, m, α, position[α][n, m])
#
#     return tm


nm = create_TBModel(n_orbital, lattice_vector, isorthogonal=True)
for ind in range(len(supercell_vector)):
    nm.set_hopping(R=tuple(supercell_vector[ind, :]),
                   n=orbital_index[ind, 0],
                   m=orbital_index[ind, 1],
                   hopping=hopping[ind][0])

# OLP_r = [[np.zeros((n_orbital, n_orbital)) for _ in range(8)] for __ in range(8)]
# for ind in range(len(supercell_vector)):
#     for alpha in range(1, 4):
#         i = orbital2atom_orb[str(orbital_index[ind, 0])]
#         j = orbital2atom_orb[str(orbital_index[ind, 1])]
#         OLP_r[i][j] += atom_pos[alpha - 1, i] * nm.overlaps.get(tuple(supercell_vector[ind, :]),
#                                                                                     np.eye(n_orbital))
#
#
# for ind in range(len(supercell_vector)):
#     # OLP_r = atom_pos[alpha, i]*nm.OLP[tuple(dd[ind, :])]
#     i = orbital2atom_orb[str(orbital_index[ind, 0])]
#     j = orbital2atom_orb[str(orbital_index[ind, 1])]
#         # nm.overlaps[]
#     for alpha in range(1, 4):
#         nm.set_position(R=tuple(supercell_vector[ind, :]),
#                         n=orbital_index[ind, 0],
#                         m=orbital_index[ind, 1],
#                         alpha=alpha,
#                         pos=OLP_r[alpha - 1][i][j][orbital_index[ind, 0], orbital_index[ind, 1]])
#
#         # nm.set_position(R=tuple(dd[ind, :]),
#         #                 n=numorb_base[i] + ii,
#         #                 m=numorb_base[natn[i][j] - 1] + jj,
#         #                 alpha=alpha,
#         #                 pos=OLP_r[alpha - 1][orbital2atom_orb[]][j][jj, ii])
#
#     # for axis in range(3):
#     #     for i in range(atomnum):
#     #         for j in range(FNAN[i]):
#     #             OLP_r[axis][i][j] += atom_pos[axis, i] * nm.OLP[i][j]
