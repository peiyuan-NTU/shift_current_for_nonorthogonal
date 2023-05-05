from tbm import create_info_missing_tb_model
import numpy as np
from


# def getHaldane():
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

    # return haldane

# getHaldane()

