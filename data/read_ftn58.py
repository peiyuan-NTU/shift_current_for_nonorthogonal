# import numpy as np
import scipy.io
import os


# import h5py
def read_ftn58(file_name="Rhsi_ftn58.mat"):
    input_data_dir = "./data"
    # file_name =
    weyl_path = os.path.join(input_data_dir, file_name)
    # print(weyl_path)
    mat = scipy.io.loadmat(weyl_path)
    # ftn58 = mat['ftn58']
    ftn58 = mat['ftn58sparse']
    dd = ftn58["dd"][0][0]
    tt = ftn58["tt"][0][0]
    ij = ftn58["ij"][0][0] - 1
    BR = ftn58["BR"][0][0]
    n_orbital = int(ftn58["norb"])
    # elements = ftn58[1:, :]
    dd.astype(int)
    ij.astype(int)
    return n_orbital, dd, tt, ij, BR, ftn58


if __name__ == "__main__":
    n_orbital, dd, tt, ij, BR, full_info = read_ftn58("ftn58sparse.mat")
    print(read_ftn58())
