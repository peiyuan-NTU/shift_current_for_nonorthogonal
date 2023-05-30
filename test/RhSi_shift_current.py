# from src.shift_conductivity import get_shift_cond
from interface.tbm_from_openmx import create_TBModel_from_openmx39
import numpy as np

tm = create_TBModel_from_openmx39("data/met.scfout")
