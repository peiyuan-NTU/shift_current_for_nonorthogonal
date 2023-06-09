#!/home/p.cui/.Anaconda/envs/mpi/bin/python
#SBATCH --job-name=rhsi_shift_cond
#SBATCH --output=rhsi_%A_%a.out
#SBATCH --error=rhsi_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=56
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --mem=100G
#SBATCH --partition=short


import os
import sys
print(os.getcwd())
print(os.path.abspath(__file__))
print(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))
from src.shift_conductivity import get_shift_cond_k
from interface.tbm_from_openmx import create_TBModel_from_openmx39
import numpy as np
from src.mesh import create_uniform_mesh
import multiprocessing as mp


def collect_result(result):
    global results
    results += result


with open("job_id.txt", "r") as f:  # read n_jobs and job_id from file
    n_jobs = int(f.readline())
    job_id = int(f.readline())

print("job_id = ", job_id)

tm = create_TBModel_from_openmx39("../../../data/rhsi.scfout")
mesh_size = [100, 100, 100]
nks = np.prod(mesh_size)
brillouin_zone_volume = abs(np.linalg.det(tm.rlat))
all_mesh = list(create_uniform_mesh(mesh_size))
mesh_for_node = np.array_split(all_mesh, n_jobs)[job_id]

n_omega = 11
omega_s = np.linspace(0, 1, n_omega)
results = np.zeros(n_omega, dtype=np.float64)
fermi = tm.fermi_energy

alpha = 1
beta = 2
gamma = 3
epsilon = np.sqrt(0.1)

# n_cores = int(os.environ["SLURM_CPUS_PER_TASK"])
n_cores = mp.cpu_count()
pool = mp.Pool(n_cores)
for k in mesh_for_node:
    print("k = ", k)
    pool.apply_async(get_shift_cond_k,
                     args=(tm, alpha, beta, gamma, omega_s, fermi, k, epsilon),
                     callback=collect_result)
pool.close()
pool.join()

# write results to file
with open(str(job_id) + "results.txt", "w") as f:
    f.write(str(results))
