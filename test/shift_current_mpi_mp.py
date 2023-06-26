import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from src.shift_conductivity import get_shift_cond_inner
from src.shift_conductivity import get_shift_cond_abc_parallel, get_shift_cond_k
from interface.tbm_from_openmx import create_TBModel_from_openmx39
import numpy as np
from src.mesh import create_uniform_mesh
import matplotlib.pyplot as plt
from mpi4py import MPI
import multiprocessing as mp

comm = MPI.COMM_WORLD

rank = comm.Get_rank()  # rank of current process
size = comm.Get_size()  # number of processes

tm = create_TBModel_from_openmx39("data/rhsi.scfout")
mesh_size = [150, 150, 150]
nks = np.prod(mesh_size)
brillouin_zone_volume = abs(np.linalg.det(tm.rlat))
all_mesh = list(create_uniform_mesh(mesh_size))

mesh_for_node = np.array_split(all_mesh, size)[rank]

n_omega = 11
omega_s = np.linspace(0, 1, n_omega)
results = np.zeros(n_omega, dtype=np.float64)
fermi = -0.395995217210 * 27.211407952


def collect_result(result):
    global results
    results += result


alpha = 1
beta = 2
gamma = 3
epsilon = np.sqrt(0.1)

pool = mp.Pool(48)
for k in all_mesh:
    pool.apply_async(get_shift_cond_k, args=(tm, alpha, beta, gamma, omega_s, fermi, k, epsilon),
                     callback=collect_result)
pool.close()
pool.join()

data_to_send = results
comm.send(data_to_send, dest=0)

if rank == 0:
    final_result = np.zeros(n_omega, dtype=np.float64)
    nks = np.prod(mesh_size)
    # colelct results from all nodes
    for i in range(1, size):
        final_result += comm.recv(source=i)
    final_result = final_result * brillouin_zone_volume / nks
    print("final_result = ", final_result)
    plt.plot(omega_s, final_result)
    plt.savefig("rhsi_shift_current.png", dpi=300)

