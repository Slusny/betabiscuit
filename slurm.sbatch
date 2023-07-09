#!/bin/bash
####
#a) Define slurm job parameters
####

#SBATCH --job-name=BetaBiscuit_Pendulum_test1

#resources:

#SBATCH --cpus-per-task=2
# the job can use and see 4 CPUs (from max 24).

#SBATCH --partition=day
# the slurm partition the job is queued to.

#SBATCH --mem-per-cpu=3G
# the job will need 12GB of memory equally distributed on 4 cpus.(251GB are available in total on one node)

#SBATCH --gres=gpu:1
#the job can use and see 1 GPUs (4 GPUs are available in total on one node) use SBATCH --gres=gpu:1080ti:1 to explicitly demand a Geforce 1080 Ti GPU. Use SBATCH --gres=gpu:A4000:1 to explicitly demand a RTX A4000 GPU

#SBATCH --time=01:00:00
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours","days-hours:minutes" and "days-hours:minutes:seconds"

#SBATCH --error=job.%J.err
# write the error output to job.*jobID*.err

#SBATCH --output=job.%J.out
# write the standard output to job.*jobID*.out

#SBATCH --mail-type=ALL
#write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user=lennart.slusny@student.uni-tuebingen.de
# your mail address

####
#b) copy all needed data to the jobs scratch folder
# We copy the cifar10 datasets which is already available in common datasets folder to our job’s scratch folder.
# Note: For this script, cifar-10 would be downloaded directly from internet, if you didn't copy it yourself.
#### 

# cp -R /common/datasets/cifar_tutorial/ /scratch/$SLURM_JOB_ID/

####
#c) Execute your code in a specific singularity container
#d) Write your checkpoints to your home directory, so that you still have them if your job fails
####

# pip3 -V
# python -V
# python3 -V
# #sudo apt-get install python3-setuptools
# pip3 install -r requirements_pip.txt 
# python3 -m pip install git+https://github.com/martius-lab/laser-hockey-env.git
singularity exec --nv /home/stud35/betabiscuit/betabiscuit_container2.sif python3 main.py --algo ddpg --env hockey-train-defence --wandb --maxepisodes 20000 --notes 'run on cluster' --tags 'defence only' --saveinterval 2000

echo DONE!