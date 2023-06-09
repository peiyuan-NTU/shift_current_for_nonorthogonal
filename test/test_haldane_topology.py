import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tight_binding_model import create_info_missing_tb_model
from src.topology import *

def getHaldane():
    lat = np.array([[1, 1 / 2, 0], [0, np.sqrt(3) / 2, 0], [0, 0, 10.0]])
    site_positions = lat @ np.array([[1 / 3, 1 / 3, 0], [2 / 3, 2 / 3, 0]]).T

    haldane = create_info_missing_tb_model(lat, site_positions, [[0], [0]])

    delta = 0.0
    t1 = -1.0
    t2 = 0.15j
    t2c = np.conj(t2)

    haldane.add_hopping(tuple([0, 0, 0]), (1, 1), (1, 1), -delta)
    haldane.add_hopping(tuple([0, 0, 0]), (2, 1), (2, 1), delta)

    haldane.add_hopping(tuple([0, 0, 0]), (1, 1), (2, 1), t1)
    haldane.add_hopping(tuple([1, 0, 0]), (2, 1), (1, 1), t1)
    haldane.add_hopping(tuple([0, 1, 0]), (2, 1), (1, 1), t1)

    haldane.add_hopping(tuple([1, 0, 0]), (1, 1), (1, 1), t2)
    haldane.add_hopping(tuple([1, -1, 0]), (2, 1), (2, 1), t2)
    haldane.add_hopping(tuple([0, 1, 0]), (2, 1), (2, 1), t2)
    haldane.add_hopping(tuple([1, 0, 0]), (2, 1), (2, 1), t2c)
    haldane.add_hopping(tuple([1, -1, 0]), (1, 1), (1, 1), t2c)
    haldane.add_hopping(tuple([0, 1, 0]), (1, 1), (1, 1), t2c)

    return haldane


# getHaldane()
if __name__ == '__main__':
    haldane = getHaldane()
    tmp = get_wilson_spectrum(tm=haldane,
                              band_indices=[0],
                              kpaths=np.array([(0, 1.0 / 3, 0), (1, 1.0 / 3, 0)]).T,
                              ndiv=1000)
    print(tmp)
    print("error1", tmp[0] / np.pi + 0.5497)
    # tmp2 = get_wilson_spectrum(tm=haldane,
    #                            band_indices=[0],
    #                            kpaths=np.array([(0, 1.0 / 6, 0), (1, 1.0 / 6, 0)]).T,
    #                            ndiv=1000)
    # print("error2", tmp2[0] / np.pi + 0.8374)


    k_path = np.array([(0, 1.0 / 3, 0), (1, 1.0 / 3, 0)]).T
    n_div = 1000
    k_points = construct_line_kpts(k_path, n_div)
    k = k_points[:, 0]
    eigen_result = get_eigen_for_tbm(haldane, k)

    get_Aw(haldane, 1, k_points[:, 0])
