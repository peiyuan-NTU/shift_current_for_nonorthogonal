import numpy as np
from scipy import sparse
import re
# from test.test_haldane_topology import getHaldane
from src.Basic_tool import construct_kpts_for_vasp
from scipy.linalg import eigh
from interface.tbm_from_openmx import create_TBModel_from_openmx39
from band.add_ftn58 import transform_H_into_ftn58
import matplotlib.pyplot as plt

tm = create_TBModel_from_openmx39("data/rhsi.scfout")
transform_H_into_ftn58(tm)

# vasp_kpath = """
# 0.0000000000   0.0000000000   0.0000000000     GAMMA
#    0.2550211533   0.2550211533   0.0000000000     C
#
#   -0.2550211533   0.7449788467   0.0000000000     C_2
#   -0.5000000000   0.5000000000   0.0000000000     Y_2
#
#   -0.5000000000   0.5000000000   0.0000000000     Y_2
#    0.0000000000   0.0000000000   0.0000000000     GAMMA
#
#    0.0000000000   0.0000000000   0.0000000000     GAMMA
#   -0.5000000000   0.5000000000   0.5000000000     M_2
#
#   -0.5000000000   0.5000000000   0.5000000000     M_2
#   -0.2544326032   0.7455673968   0.5000000000     D
#
#    0.2544326032   0.2544326032   0.5000000000     D_2
#    0.0000000000   0.0000000000   0.5000000000     A
#
#    0.0000000000   0.0000000000   0.5000000000     A
#    0.0000000000   0.0000000000   0.0000000000     GAMMA
#
#    0.0000000000   0.5000000000   0.5000000000     L_2
#    0.0000000000   0.0000000000   0.0000000000     GAMMA
#
#    0.0000000000   0.0000000000   0.0000000000     GAMMA
#    0.0000000000   0.5000000000   0.0000000000     V_2
# """
# vasp_kpath=vasp_kpath.strip()

vasp_kpath = """
0.0000000000   0.0000000000   0.0000000000     GAMMA
   0.0000000000   0.5000000000   0.0000000000     X

   0.0000000000   0.5000000000   0.0000000000     X
   0.5000000000   0.5000000000   0.0000000000     M

   0.5000000000   0.5000000000   0.0000000000     M
   0.0000000000   0.0000000000   0.0000000000     GAMMA

   0.0000000000   0.0000000000   0.0000000000     GAMMA
   0.5000000000   0.5000000000   0.5000000000     R

   0.5000000000   0.5000000000   0.5000000000     R
   0.0000000000   0.5000000000   0.0000000000     X

   0.5000000000   0.5000000000   0.5000000000     R
   0.5000000000   0.5000000000   0.0000000000     M

   0.5000000000   0.5000000000   0.0000000000     M
   0.5000000000   0.0000000000   0.0000000000     X_1
"""

tmp1 = vasp_kpath.replace("\n\n", "\n")
tmp2 = tmp1.split("\n")
tmp3 = list(filter(("").__ne__, tmp2))
tmp4 = [i.strip(" ") for i in tmp3]
# tmp5 = [i.split(r"\s*") for i in tmp4]
tmp5 = [re.split(r"\s+", i) for i in tmp4]
tmp_6_1 = [list(i[:3]) for i in tmp5]  # kpoint coordinates
tmp_6_2 = [i[3] for i in tmp5]  # kpoint labels

k_path = np.array(tmp_6_1, dtype=float).T

assert len(tmp5) % 2 == 0, "length of kpath should be even number"

all_kpoints = construct_kpts_for_vasp(k_path, 10)

x_coordinates = [0]
k_labels = [tmp_6_2[0]]
k_label_positions = [0]
for i in range(np.shape(all_kpoints)[1] - 1):
    vector = all_kpoints[:, i + 1] - all_kpoints[:, i]
    reciprocal_vector = tm.rlat @ vector
    length = np.linalg.norm(reciprocal_vector)
    print(length)
    x = x_coordinates[-1] + length
    x_coordinates.append(x)

for i in range(int(len(tmp_6_2) / 2) - 1):
    if tmp_6_2[i * 2 + 1] == tmp_6_2[i * 2 + 2]:
        k_labels.append(tmp_6_2[i * 2 + 1])
    if tmp_6_2[i * 2 + 1] != tmp_6_2[i * 2 + 2]:
        k_labels.append(tmp_6_2[i * 2 + 1] + "|" + tmp_6_2[i * 2 + 2])

for i in range(int(np.shape(k_path)[1] / 2) - 1):
    vector = k_path[:, i * 2 + 1] - k_path[:, i * 2]
    reciprocal_vector = tm.rlat @ vector
    length = np.linalg.norm(reciprocal_vector)
    x = k_label_positions[-1] + length
    k_label_positions.append(x)

H_ftn58 = tm.H_ftn58
S_ftn58 = tm.S_ftn58
E_fermi = -0.134773478314*27.2114079527
all_eigenvalues = []

for i in range(np.shape(all_kpoints)[1]):
    k_vector = all_kpoints[:, i]

    h_data = np.multiply(np.exp(1j * H_ftn58[:, 4:7].astype(int) @ k_vector*2*np.pi), H_ftn58[:, 3])
    h_row = H_ftn58[:, 1].astype(int)
    h_col = H_ftn58[:, 2].astype(int)
    sparseMatrix = sparse.csc_matrix((h_data,
                                      (h_row, h_col)),
                                     shape=(tm.norbits, tm.norbits)).toarray()
    H = (sparseMatrix + sparseMatrix.conjugate().T) / 2

    s_data = np.multiply(np.exp(1j * S_ftn58[:, 4:7].astype(int) @ k_vector * 2 * np.pi) , S_ftn58[:, 3])
    s_row = S_ftn58[:, 1].astype(int)
    s_col = S_ftn58[:, 2].astype(int)
    sparseMatrix = sparse.csc_matrix((s_data,
                                      (s_row, s_col)),
                                     shape=(tm.norbits, tm.norbits)).toarray()
    S = (sparseMatrix + sparseMatrix.conjugate().T) / 2

    eigen_values, eigen_vectors = eigh(H, S)
    all_eigenvalues.append(eigen_values.astype(float) - E_fermi)

all_eigenvalues = np.array(all_eigenvalues)
# plt.figure((10,10))
for i in range(np.shape(all_eigenvalues)[1]):
    plt.plot(x_coordinates, all_eigenvalues[:, i], "bo-")
plt.ylim(-3, 3)
plt.show()
# plt.plot(x_coordinates, all_eigenvalues)
