# import numpy as np
import scipy.io
import os

atom0_type = 'Rh'
atom1_type = 'Rh'
atom2_type = 'Rh'
atom3_type = 'Rh'
atom4_type = 'Si'
atom5_type = 'Si'
atom6_type = 'Si'
atom7_type = 'Si'
atom0_orbital = [1, 2, 3, 4, 5]
atom1_orbital = [6, 7, 8, 9, 10]
atom2_orbital = [11, 12, 13, 14, 15]
atom3_orbital = [16, 17, 18, 19, 20]
atom4_orbital = [21, 22, 23]
atom5_orbital = [24, 25, 26]
atom6_orbital = [27, 28, 29]
atom7_orbital = [30, 31, 32]

site_orbital = {"0": atom0_orbital,
                "1": atom1_orbital,
                "2": atom2_orbital,
                "3": atom3_orbital,
                "4": atom4_orbital,
                "5": atom5_orbital,
                "6": atom6_orbital,
                "7": atom7_orbital}


def orbital_index_to_atom_and_orbital(orbital_index: int) -> (int, int):
    # print(orbital_index)
    # print(type(orbital_index))
    # if not isinstance(orbital_index, int):
    #     raise TypeError("orbital_index_to_atom_orbital: orbital_index must be int")

    assert orbital_index in range(1, 33)
    for atom_index, orbital_list in site_orbital.items():
        if orbital_index in orbital_list:
            return int(atom_index), orbital_list.index(orbital_index)
    raise ValueError("orbital_index_to_atom_orbital: orbital_index not found")


def atom_and_orbital_to_orbital_index(atom_index: int, orbital_index: int):
    assert atom_index in range(8)
    return site_orbital[str(atom_index)][orbital_index]


# import h5py
def read_ftn58(file_name="Rhsi_ftn58.mat"):
    input_data_dir = "./data"
    # file_name =
    weyl_path = os.path.join(input_data_dir, file_name)
    # print(weyl_path)
    mat = scipy.io.loadmat(weyl_path)
    # ftn58 = mat['ftn58']
    ftn58 = mat['ftn58sparse']
    dd = ftn58["dd"][0][0]  # supercell vector
    tt = ftn58["tt"][0][0]  # hopping
    ij = ftn58["ij"][0][0] - 1  # orbital index
    BR = ftn58["BR"][0][0]  # lattice vector
    n_orbital = int(ftn58["norb"])
    # elements = ftn58[1:, :]
    dd.astype(int)
    ij.astype(int)
    atom_and_orbital = [[orbital_index_to_atom_and_orbital(i[0] + 1),
                         orbital_index_to_atom_and_orbital(i[1] + 1)] for i in ij]

    orbital2atom_orb = {}
    for orbital in range(32):
        orbital2atom_orb[str(orbital)] = [orbital_index_to_atom_and_orbital(orbital + 1)][0][0]

    # atom_orb2orbital={}
    # for atom in range(8):

    result = {
        "n_orbital": n_orbital,
        "supercell_vector": dd,
        "hopping": tt,
        "orbital_index": ij,
        "lattice_vector": BR,
        "full_info": ftn58,
        "atom_and_orbital": atom_and_orbital,
        "orbital2atom_orb": orbital2atom_orb
    }
    return result


if __name__ == "__main__":
    # n_orbital, dd, tt, ij, BR, full_info = read_ftn58("ftn58sparse.mat")
    ftn58_dict = read_ftn58()
    n_orbital = ftn58_dict["n_orbital"]
    supercell_vector = ftn58_dict["supercell_vector"]
    hopping = ftn58_dict["hopping"]
    orbital_index = ftn58_dict["orbital_index"]
    lattice_vector = ftn58_dict["lattice_vector"]
    full_info = ftn58_dict["full_info"]
    atom_and_orbital = ftn58_dict["atom_and_orbital"]
    orbital2atom_orb = ftn58_dict["orbital2atom_orb"]
    # print(read_ftn58())
