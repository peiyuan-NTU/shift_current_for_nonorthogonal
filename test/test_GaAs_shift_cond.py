# import test.
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.shift_conductivity import get_shift_cond_abb
from interface.tbm_from_openmx import create_TBModel_from_openmx39
import numpy as np

tm = create_TBModel_from_openmx39()
tmp1 = get_shift_cond_abb(tm=tm,
                          alpha=2,
                          beta=1,
                          omega_s=[0.5, 1.0, 1.5],
                          mu=0.0,
                          mesh_size=[100, 100, 1],
                          epsilon=np.sqrt(0.1))
print(tmp1)

# tmp2 = get_wilson_spectrum(tm=tm,
#                            band_indices=[0],
#                            kpaths=np.array([(0, 1.0 / 3, 0), (1, 1.0 / 3, 0)]).T,
#                            ndiv=1000)
# [-42.63600819496399, -508.1923796077782, -507.1431205823643]
