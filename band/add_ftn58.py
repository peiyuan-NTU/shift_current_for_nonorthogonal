import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from src.shift_conductivity import get_shift_cond_inner
from src.shift_conductivity import get_shift_cond_abc_parallel
from interface.tbm_from_openmx import create_TBModel_from_openmx39


def transform_H_into_ftn58(tm):
    assert tm.hoppings is not None
    H_ftn58 = np.zeros([tm.norbits * tm.norbits * len(tm.hoppings), 7], dtype=np.float64)
    S_ftn58 = np.zeros([tm.norbits * tm.norbits * len(tm.hoppings), 7], dtype=np.float64)
    h_counter = 0
    for key, item in tm.hoppings.items():
        x, y, z = key
        # print("x", x, "y", y, "z", z)
        for orbital1 in range(item.shape[0]):
            for orbital2 in range(item.shape[1]):
                # print("orbital1", orbital1, "orbital2", orbital2)
                row_index = h_counter * (tm.norbits ** 2) + orbital1 * tm.norbits + orbital2
                H_ftn58[row_index, :] = np.array(
                    [row_index, orbital1, orbital2, item[orbital1, orbital2], x, y, z])
        h_counter += 1
    s_counter = 0
    for key, item in tm.overlaps.items():
        x, y, z = key
        # print("x", x, "y", y, "z", z)
        for orbital1 in range(item.shape[0]):
            for orbital2 in range(item.shape[1]):
                # print("orbital1", orbital1, "orbital2", orbital2)
                row_index = s_counter * (tm.norbits ** 2) + orbital1 * tm.norbits + orbital2
                S_ftn58[row_index, :] = np.array(
                    [row_index, orbital1, orbital2, item[orbital1, orbital2], x, y, z])
        s_counter += 1
    tm.H_ftn58 = H_ftn58
    tm.S_ftn58 = S_ftn58


if __name__ == '__main__':
    # print("Hello world!")
    tm = create_TBModel_from_openmx39("data/rhsi.scfout")
    transform_H_into_ftn58(tm)
