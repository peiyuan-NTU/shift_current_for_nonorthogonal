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
    dd = ftn58["dd"]
    tt = ftn58["tt"]
    ij = ftn58["ij"]
    BR = ftn58["BR"]
    n_orbital = int(ftn58["norb"])
    # elements = ftn58[1:, :]
    return n_orbital, dd, tt, ij, BR


if __name__ == "__main__":
    print(read_ftn58())
