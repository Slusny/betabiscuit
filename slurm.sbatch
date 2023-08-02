#!/bin/bash
####
#a) Define slurm job parameters
####

#SBATCH --job-name=BetaBiscuit_Pendulum_test1

#resources:

#SBATCH --cpus-per-task=2
# the job can use and see 4 CPUs (from max 24).

#SBATCH --partition=week
# the slurm partition the job is queued to.

#SBATCH --mem-per-cpu=8G
# the job will need 12GB of memory equally distributed on 4 cpus.(251GB are available in total on one node)

#SBATCH --gres=gpu:1
#the job can use and see 1 GPUs (4 GPUs are available in total on one node) use SBATCH --gres=gpu:1080ti:1 to explicitly demand a Geforce 1080 Ti GPU. Use SBATCH --gres=gpu:A4000:1 to explicitly demand a RTX A4000 GPU

#SBATCH --time=3-3
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours","days-hours:minutes" and "days-hours:minutes:seconds"

#SBATCH --error=jobs/job.%J.err
# write the error output to job.*jobID*.err

#SBATCH --output=jobs/job.%J.out
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
singularity exec --nv /home/stud35/betabiscuit/betabiscuit_container2.sif python3 python main.py --algo ddpg --env hockey --max_episodes 1000000 --notes "run on pc" --save_interval 4000 --batch_size 512 --hidden_sizes_actor [128,128] --hidden_sizes_critic [128,128,64] --legacy --bootstrap "betabiscuit/hockey - ddpg/model:v54" --wandb_resume "lqcmv838"

#main.py --algo td3 --env hockey-train-shooting --max_episodes 1000000 --notes "shooting teacher" --batch_size 512 --hidden_sizes_actor [300,300] --hidden_sizes_critic [300,300] -u 2 --discount 0.95 --learning_rate_actor 0.0001 --learning_rate_critic 0.0001 --lr 0.0001 --save_interval 5000 --wandb



#main.py --algo ddpg --env hockey --wandb --max_episodes 1000000 --notes 'run on cluster' --save_interval 4000 --batch_size 512 --hidden_sizes_actor "[256,256]" --hidden_sizes_critic "[256,256,64]"

echo DONE!