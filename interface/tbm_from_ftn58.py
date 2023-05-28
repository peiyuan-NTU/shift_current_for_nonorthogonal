import numpy as np

from data.parse_ftn58 import read_ftn58
from src.tight_binding_model import create_TBModel

n_orbital, dd, tt, ij, BR, full_info = read_ftn58()
atom_pos = [i[1][0] for i in full_info["Ainfo"][0][0][0]]
atom_pos = np.transpose(atom_pos)
OLP_r = []
nm = create_TBModel(n_orbital, BR, isorthogonal=True)
for ind in range(len(dd)):
    nm.set_hopping(R=tuple(dd[ind, :]),
                   n=ij[ind, 0],
                   m=ij[ind, 1],
                   hopping=tt[ind][0])

for ind in range(len(dd)):
    # OLP_r = atom_pos[alpha, i]*nm.OLP[tuple(dd[ind, :])]
    for alpha in range(1, 4):
        OLP_r = atom_pos[alpha - 1, ij[ind, 0]] * nm.overlaps[tuple(dd[ind, :])]
        nm.set_position(R=tuple(dd[ind, :]),
                        n=ij[ind, 0],
                        m=ij[ind, 1],
                        alpha=alpha,
                        pos=OLP_r[alpha - 1][ij[ind, 0], ij[ind, 1]])

        nm.set_position(R=tuple(dd[ind, :]),
                        n=numorb_base[i] + ii,
                        m=numorb_base[natn[i][j] - 1] + jj,
                        alpha=alpha,
                        pos=OLP_r[alpha - 1][i][j][jj, ii])

    # for axis in range(3):
    #     for i in range(atomnum):
    #         for j in range(FNAN[i]):
    #             OLP_r[axis][i][j] += atom_pos[axis, i] * nm.OLP[i][j]
