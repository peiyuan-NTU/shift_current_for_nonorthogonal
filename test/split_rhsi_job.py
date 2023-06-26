import os
import sys
import subprocess

# 1. create job directories
# 2. write job id and n_jobs to file
# 3. write python code
# 4. write slurm script
# 5. submit job


python_code = """
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

tm = create_TBModel_from_openmx39("data/rhsi.scfout")
mesh_size = [150, 150, 150]
nks = np.prod(mesh_size)
brillouin_zone_volume = abs(np.linalg.det(tm.rlat))
all_mesh = list(create_uniform_mesh(mesh_size))
mesh_for_node = np.array_split(all_mesh, n_jobs)[job_id]

n_omega = 11
omega_s = np.linspace(0, 1, n_omega)
results = np.zeros(n_omega, dtype=np.float64)
fermi = -0.395995217210 * 27.211407952

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

## write results to file
with open(str(job_id) + "results.txt", "w") as f:
    f.write(str(results))
"""

slurm_script = """
#!/bin/bash
#SBATCH --job-name=rhsi_shift_cond
#SBATCH --output=rhsi.out
#SBATCH --error=rhsi.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=10G
#SBATCH --partition=short

source activate /home/p.cui/.Anaconda/envs/mpi
python job.py
"""


n_jobs = 100

prefix = os.path.join("rhsi")
for i in range(n_jobs):
    dir_name = os.path.join(prefix, "job_" + str(i))
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        with open(dir_name + "/job_id.txt", "w") as f:  # write job id and n_jobs to file
            f.write(str(n_jobs) + "\n")
            f.write(str(i) + "\n")
        with open(dir_name + "/job.py", "w") as f:  # write python code
            f.write(python_code)
        with open(dir_name + "/job.slurm", "w") as f:  # write slurm script
            f.write(slurm_script)
        subprocess.run(["sbatch", "job.slurm"], cwd=dir_name)



# read job id from file
# with open("job_id.txt", "r") as f:
#     job_id = int(f.read())
