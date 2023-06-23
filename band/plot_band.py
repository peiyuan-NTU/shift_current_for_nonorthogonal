import numpy as np
from scipy import sparse
import re
# from test.test_haldane_topology import getHaldane
from src.Basic_tool import construct_kpts_for_vasp
from scipy.linalg import eigh
from interface.tbm_from_openmx import create_TBModel_from_openmx39
from band.add_ftn58 import transform_H_into_ftn58
import matplotlib.pyplot as plt


def cal_kpts_x_val(kpaths, reciprocal_lattice, n_lines, n_points_per_line):
    x_coordinates_inner = np.zeros(np.shape(all_kpoints)[1])
    for i in range(n_lines):
        if i == 0:
            x_coordinates_inner[i * n_points_per_line] = 0
        if i != 0:
            x_coordinates_inner[i * n_points_per_line] = x_coordinates_inner[i * n_points_per_line - 1]
        for j in range(n_points_per_line - 1):
            kpt_index = j + i * n_points_per_line
            vector_inner = all_kpoints[:, kpt_index + 1] - all_kpoints[:, kpt_index]
            reciprocal_vector_inner = reciprocal_lattice @ vector_inner
            length_inner = np.linalg.norm(reciprocal_vector_inner)
            x_coordinates_inner[kpt_index + 1] = x_coordinates_inner[kpt_index] + length_inner
    return x_coordinates_inner


def cal_klabels_position(k_label_vectors, k_label_names_inner, reciprocal_lattice):
    k_labels_inner = []
    k_x_positions_inner = []

    for i in range(int(len(k_label_names_inner) / 2)):
        if i == 0:
            k_labels_inner.append(k_label_names_inner[0])
            continue
        if k_label_names_inner[i * 2] == k_label_names_inner[i * 2 - 1]:
            k_labels_inner.append(k_label_names_inner[i * 2])  # end point of previous line
            k_labels_inner.append(k_label_names_inner[i * 2])  # start point of next line
        if k_label_names_inner[i * 2] != k_label_names_inner[i * 2 - 1]:
            k_labels_inner.append(
                k_label_names_inner[i * 2 - 1] + "|" + k_label_names_inner[i * 2])  # end point of previous line
            k_labels_inner.append(
                k_label_names_inner[i * 2 - 1] + "|" + k_label_names_inner[i * 2])  # start point of next line

    for i in range(int(len(k_label_names_inner) / 2)):
        # print("k_label_vectors[i * 2 + 1]", k_label_vectors[:, i * 2 + 1])
        # print("k_label_vectors[i * 2]", k_label_vectors[:, i * 2])
        vector = k_label_vectors[:, i * 2 + 1] - k_label_vectors[:, i * 2]
        # print("vector", vector)
        reciprocal_vector = reciprocal_lattice @ vector
        length = np.linalg.norm(reciprocal_vector)
        if i == 0:
            k_x_positions_inner.append(0)
        if i != 0:
            k_x_positions_inner.append(k_x_positions_inner[-1])
        x = k_x_positions_inner[-1] + length
        k_x_positions_inner.append(x)

    return k_labels_inner, k_x_positions_inner


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
tmp_6_1 = [list(i[:3]) for i in tmp5]
k_label_names = [i[3] for i in tmp5]  # kpoint labels
k_path = np.array(tmp_6_1, dtype=float).T  # kpoint coordinates

assert len(tmp5) % 2 == 0, "length of kpath should be even number"

n_points = 10
all_kpoints = construct_kpts_for_vasp(k_path, n_points)

x_coordinates = cal_kpts_x_val(kpaths=k_path,
                               reciprocal_lattice=tm.rlat,
                               n_lines=int(len(k_label_names) / 2),
                               n_points_per_line=n_points)

k_labels, k_label_positions = cal_klabels_position(k_label_vectors=k_path,
                                                   k_label_names_inner=k_label_names,
                                                   reciprocal_lattice=tm.rlat)

H_ftn58 = tm.H_ftn58
S_ftn58 = tm.S_ftn58
E_fermi = -0.134773478314 * 27.2114079527
all_eigenvalues = []

for i in range(np.shape(all_kpoints)[1]):
    k_vector = all_kpoints[:, i]

    h_data = np.multiply(np.exp(1j * H_ftn58[:, 4:7].astype(int) @ k_vector * 2 * np.pi), H_ftn58[:, 3])
    h_row = H_ftn58[:, 1].astype(int)
    h_col = H_ftn58[:, 2].astype(int)
    sparseMatrix = sparse.csc_matrix((h_data,
                                      (h_row, h_col)),
                                     shape=(tm.norbits, tm.norbits)).toarray()
    H = (sparseMatrix + sparseMatrix.conjugate().T) / 2

    s_data = np.multiply(np.exp(1j * S_ftn58[:, 4:7].astype(int) @ k_vector * 2 * np.pi), S_ftn58[:, 3])
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
    plt.plot(x_coordinates, all_eigenvalues[:, i], "bo-", markersize=2)
# plt.xticks(k_label_positions, k_labels)
plt.ylim(-3, 3)
plt.show()
# plt.plot(x_coordinates, all_eigenvalues)
