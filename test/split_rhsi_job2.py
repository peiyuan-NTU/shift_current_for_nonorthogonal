import os
import sys
import subprocess

# 1. create job directories
# 2. write job id and n_jobs to file
# 3. write python code
# 4. write slurm script
# 5. submit job


f = open("job.py", "r")
python_code = f.read()
f.close()

n_jobs = 100
wd= os.getcwd()
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
        print(os.path.join(wd, dir_name))
        os.chdir(os.path.join(wd, dir_name))

        cmd_output = subprocess.run(["sbatch", "job.py"], capture_output=True, text=True, cwd=dir_name)
        os.chdir(wd)
        print(cmd_output.stdout)

# read job id from file
# with open("job_id.txt", "r") as f:
#     job_id = int(f.read())
