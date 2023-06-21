import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from src.shift_conductivity import get_shift_cond_inner
from src.shift_conductivity import get_shift_cond_abc_parallel
from interface.tbm_from_openmx import create_TBModel_from_openmx39
import numpy as np
import matplotlib.pyplot as plt

tm = create_TBModel_from_openmx39("data/met.scfout")
omega_s = np.linspace(0, 1, 11)
mesh_size = [100, 100, 100]
fermi = -0.134773478314*27.211407952
shift_conductivity_xyz = get_shift_cond_abc_parallel(tm=tm,
                                                     alpha=1,
                                                     beta=2,
                                                     gamma=3,
                                                     omega_s=omega_s,
                                                     mu=fermi,
                                                     mesh_size=mesh_size,
                                                     epsilon=np.sqrt(0.1))


# plt.plot(omega_s, shift_conductivity_zxy + shift_conductivity_yzx + shift_conductivity_xyz)
