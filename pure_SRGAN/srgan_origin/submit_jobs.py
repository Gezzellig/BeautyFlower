import os, subprocess, time

def run_jobs(amt_jobs):
	settings = ['upEnd', 'upBegin', 'upBicubic']

	for s in settings:
		for _ in range(int(amt_jobs)):
			filename = "batch_gpu.txt"
			script 	 = open(filename, "w+")
			jobstring = template.format(time_job=time_job, mem=mem, nodes=nodes, ntasks=ntasks, job_name=job_name, upsampling=s)
			script.write(jobstring)
			script.close()
			try:
				subprocess.call(["sbatch", filename])
				pass
			except OSError:
				script.close()
				print("sbatch not found or filename wrong")
				
			os.remove(filename)
			print ("Submitted job: ", filename)
			print (jobstring)
			time.sleep(1)

amt_jobs = input("Amount of jobs per experiment: ")
time_job = input("Amount of time_job per experiment (eg. 01:00:00): ")
mem = input("Amount of memory per experiment: ")
nodes = input("Amount of nodes per experiment: ")
ntasks = input("Amount of ntasks per experiment: ")
job_name = input("Name of the jobs: ")

template = """#!/bin/bash
#SBATCH --time_job={time_job}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem={mem}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks}
#SBATCH --job-name={job_name}{upsampling}

python3 ./main.py --upsampling {upsampling}"""


run_jobs(amt_jobs)


