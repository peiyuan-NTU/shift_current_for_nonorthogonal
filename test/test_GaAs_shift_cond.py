from shift_conductivity import get_shift_cond
from tbm_from_openmx import create_TBModel_from_openmx39
import numpy as np

tm = create_TBModel_from_openmx39()
print(get_shift_cond(tm=tm,
                     alpha=2,
                     beta=1,
                     omega_s=[0.5, 1.0, 1.5],
                     mu=0.0,
                     mesh_size=[100, 100, 1],
                     epsilon=np.sqrt(0.1)))

# [-42.63600819496399, -508.1923796077782, -507.1431205823643]