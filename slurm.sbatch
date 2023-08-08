#!/bin/bash
####
#a) Define slurm job parameters
####

#SBATCH --job-name=BetaBiscuit_Pendulum_test1

#resources:

#SBATCH --cpus-per-task=4
# the job can use and see 4 CPUs (from max 24).

#SBATCH --partition=day
# the slurm partition the job is queued to.

#SBATCH --mem-per-cpu=8G
# the job will need 12GB of memory equally distributed on 4 cpus.(251GB are available in total on one node)

#SBATCH --gres=gpu:1
#the job can use and see 1 GPUs (4 GPUs are available in total on one node) use SBATCH --gres=gpu:1080ti:1 to explicitly demand a Geforce 1080 Ti GPU. Use SBATCH --gres=gpu:A4000:1 to explicitly demand a RTX A4000 GPU

#SBATCH --time=1-0
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
singularity exec --nv /home/stud35/betabiscuit/betabiscuit_container3.sif python3 selfplay.py --max_episodes_per_pair 500 --save_interval 500 --max_timesteps 350 --notes "self play normal R,  Single Train DQN" --agents self-play/agents/pretty-shape-22_199.json self-play/agents/gallant-butterfly-13_199.json self-play/agents/gallant-butterfly-13_139.json self-play/agents/fallen-shape-8_203.json self-play/agents/dulcet-leaf-12_270.json --wandb --val_episodes 100 --replay_buffer_fill --replay_buffer_fill_ratio 100 --save_buffer_interval 3 --scripted_agent --replay_ratio 0.25 --cpu --all_against_one self-play/agents/fallen-shape-8_203.json --train_only_one

# Trying to resurect warm-mountain - all against (simple reward)
#selfplay.py --max_episodes_per_pair 1000 --save_interval 1000 --max_timesteps 350 --notes "self play simple reward all against one restarted from morning silence" --agents self-play/agents/pretty-shape-22_199.json self-play/agents/gallant-butterfly-13_199.json self-play/agents/fallen-shape-8_203.json self-play/agents/dulcet-leaf-12_270.json self-play/agents/dulcet-leaf-12_305.json --wandb --val_episodes 100 --simple_reward --train_only_one --save_buffer_interval 5 --replay_ratio 0.25 --replay_buffer_fill --replay_buffer_fill_ratio 100 --scripted_agent --all_against_one self-play/agents/gallant-butterfly-13_139.json --all_against_one_bootstrap "betabiscuit/self-play/morning-silence-67sp-gallant-butterfly-13_139_model:v7" --collect_buffer_from_run "warm-mountain-69" --buffer_identifyer "end" --cpu

#Hardy sun self play
#selfplay.py --max_episodes_per_pair 500 --save_interval 500 --max_timesteps 350 --notes "self play started from Hard Sun" --agents self-play/agents/pretty-shape-22_199.json self-play/agents/gallant-butterfly-13_199.json self-play/agents/gallant-butterfly-13_139.json self-play/agents/fallen-shape-8_203.json self-play/agents/dulcet-leaf-12_270.json --wandb --val_episodes 100 --replay_buffer_fill --replay_buffer_fill_ratio 100 --save_buffer_interval 3 --scripted_agent --replay_ratio 0.25 --cpu

#main.py --algo td3 --env hockey --max_episodes 1000000 --notes "resumed from dulcet-leaf at 1.3M try to retrain" --save_interval 5000 --batch_size 512 --hidden_sizes_actor "[300,300]" --hidden_sizes_critic "[300,300]" --bootstrap "betabiscuit/hockey - td3/dulcet-leaf-12_model:v264" --update 2 --discount 0.99 --learning_rate_actor 0.00001 --learning_rate_critic 0.00001 --policy_noise 0.1 --noise_clip 0.3 --per --cpu --replay_ratio 0.2 --eps 0.1 --validation_interval 500 --filled_buffer_ratio 1


#main.py --algo td3 --env hockey --max_episodes 1000000 --notes "resume fallen shape without bc" --batch_size 512 --hidden_sizes_actor [200,200] --hidden_sizes_critic [200,200] -u 2 --discount 0.99 --learning_rate_actor 0.0001 --learning_rate_critic 0.0001 --lr 0.0001 --save_interval 5000 --wandb --bootstrap "betabiscuit/hockey - td3/fallen-shape-8_model:v203" --per --cpu --replay_ratio 0.2 --eps 0.1 --wandb_resume "h8l02eg1"

# Restarted dulcet-leaf
#main.py --algo td3 --env hockey --max_episodes 1000000 --notes "fixed bc scaling" --save_interval 5000 --batch_size 512 --hidden_sizes_actor "[300,300]" --hidden_sizes_critic "[300,300]" --bootstrap "betabiscuit/hockey - td3/dulcet-leaf-12_model:v183" --update 2 --discount 0.98 --lr 0.0001 --learning_rate_actor 0.0001 --learning_rate_critic 0.0001 --bc --per --cpu --wandb --wandb_resume "kzxa5iwv" --eps 0.3 --replay_ratio 0.2 --bc_lambda 0.5

# fallen-shape
# main.py --algo td3 --env hockey --max_episodes 1000000 --notes "run on pc" --batch_size 512 --hidden_sizes_actor [200,200] --hidden_sizes_critic [200,200] -u 2 --discount 0.99 --learning_rate_actor 0.0001 --learning_rate_critic 0.0001 --lr 0.0001 --save_interval 5000 --wandb --bootstrap "betabiscuit/hockey - td3/fallen-shape-8_model:v104" --per --cpu --replay_ratio 0.2 --bc_lambda 0.5 --eps 0.3

# main.py --algo td3 --env hockey --max_episodes 1000000 --notes "cluster" --tags "c" --batch_size 256 --hidden_sizes_actor [256,128,128] --hidden_sizes_critic [256,128,128] -u 2 --discount 0.99 --learning_rate_actor 0.0001 --learning_rate_critic 0.0001 --lr 0.0001 --save_interval 5000 --wandb --cpu --per --replay_ratio 0.2

#main.py --algo td3 --env hockey --max_episodes 1000000 --notes "resumed from divine" --save_interval 5000 --batch_size 512 --hidden_sizes_actor [300,300] --hidden_sizes_critic [300,300] --bootstrap "betabiscuit/hockey - td3/divine-pond-9_model:v62" --train_iter 2 --update 2 --discount 0.99 --lr 0.0001 --learning_rate_actor 0.0001 --learning_rate_critic 0.0001 --policy_noise 0.2 --noise_clip 0.5 --bc --per --cpu --wandb

#devout-frog
#main.py --algo ddpg --env hockey --max_episodes 1000000 --notes "bootstraped from dauntless-lake" --save_interval 4000 --batch_size 512 --hidden_sizes_actor [128,128] --hidden_sizes_critic [128,128,64] --legacy --bootstrap "betabiscuit/hockey - ddpg/real_opponent:v0" --wandb --per --discount 0.99 --bc --cpu

#main.py --algo td3 --env hockey --max_episodes 1000000 --notes "resumed from divine ep 160k" --save_interval 5000 --batch_size 512 --hidden_sizes_actor [300,300] --hidden_sizes_critic [300,300] --bootstrap "betabiscuit/hockey - td3/divine-pond-9_model:v18" --train_iter 2 --update 2 --discount 0.99 --lr 0.0001 --learning_rate_actor 0.0001 --learning_rate_critic 0.0001 --policy_noise 0.2 --noise_clip 0.5 --bc --per --cpu --wandb

#restarted ddpg
#main.py --algo ddpg --env hockey --max_episodes 1000000 --notes "bootstraped from dauntless-lake" --save_interval 5000 --batch_size 512 --hidden_sizes_actor [128,128] --hidden_sizes_critic [128,128,64] --legacy --bootstrap "betabiscuit/hockey - ddpg/real_opponent:v0" --wandb --per --dense_reward --discount 0.99

#main.py --algo td3 --env hockey-train-shooting --max_episodes 1000000 --notes "shooting teacher" --batch_size 512 --hidden_sizes_actor [300,300] --hidden_sizes_critic [300,300] -u 2 --discount 0.95 --learning_rate_actor 0.0001 --learning_rate_critic 0.0001 --lr 0.0001 --save_interval 5000 --wandb



#main.py --algo ddpg --env hockey --wandb --max_episodes 1000000 --notes 'run on cluster' --save_interval 4000 --batch_size 512 --hidden_sizes_actor "[256,256]" --hidden_sizes_critic "[256,256,64]"

echo DONE!