from src.shift_conductivity import get_shift_cond_inner
from interface.tbm_from_openmx import create_TBModel_from_openmx39
import numpy as np
import matplotlib.pyplot as plt

tm = create_TBModel_from_openmx39("data/met.scfout")
omega_s = np.linspace(0, 1, 100)
shift_conductivity = get_shift_cond_inner(tm=tm,
                                          alpha=1,
                                          beta=2,
                                          gamma=3,
                                          omega_s=omega_s,
                                          mu=0.0,
                                          mesh_size=[100, 100, 1],
                                          epsilon=np.sqrt(0.1))

plt.plot(omega_s, shift_conductivity)
