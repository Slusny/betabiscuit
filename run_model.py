import argparse
import gymnasium as gym
import sys
from pathlib import Path
import laserhockey.hockey_env as h_env
import torch
sys.path.insert(0,'./DDPG')
from DDPG import DDPGAgent
import wandb
import numpy as np
import pyvirtualdisplay

virtual_disp = False
max_timesteps = 100
max_episodes = 2
entity, project, runid = "betabiscuit", "hockey - ddpg" , 'il0xhoe5' # set to your entity and project
artifact_name = 'model:v4'

if virtual_disp :
    _display = pyvirtualdisplay.Display(visible=True,  # use False with Xvfb
                                        size=(1400, 900))
    _display.start()

api = wandb.Api()
runs = api.runs(entity + "/" + project)
# run = api.runs(entity + "/" + project + "/" + runid)
args = runs[0].config

art = api.artifact(entity + "/" + project + "/" + artifact_name, type='model')
print(art.file())
artifact_dir = art.download()
# run = wandb.init(mode='offline')
# artifact = run.use_artifact('betabiscuit/hockey - ddpg/model:v4', type='model')
# artifact_dir = artifact.download()
state = torch.load(art.file())
# creating environment
env_name = args['env_name']
if env_name == "lunarlander":
    env = gym.make("LunarLander-v2", continuous = True)
elif env_name == "pendulum":
    env = gym.make("Pendulum-v1")
elif env_name == "hockey":
    # reload(h_env)
    env = h_env.HockeyEnv()
elif env_name == "hockey-train-shooting":
    # reload(h_env)
    env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)
elif env_name == "hockey-train-defence":
    env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE)
else:
    env = gym.make(env_name)

#create save path
savepath = 'results_run'
Path().mkdir(parents=True, exist_ok=True)

if args['algo'] == "ddpg":
    agent = DDPGAgent(env, env_name, args['seed'], savepath, wandb_run=False,
                    eps = args['eps'], 
                    learning_rate_actor = args['lr'],
                    update_target_every = args['update_every'])
    agent.restore_state(state)
    

    for i_episode in range(1, max_episodes+1):
        ob, _info = env.reset()
        timestep = 0
        total_reward = 0
        for t in range(max_timesteps):
            env.render()
            timestep += 1
            done = False
            a = agent.act(ob)
            a2 = [0,0.,0,0]
            (ob_new, reward, done, trunc, _info) = env.step(np.hstack([a,a2]))
            total_reward+= reward
            ob=ob_new
            if done: break

if virtual_disp :
    _display.stop()



def get_run_names(runs):
    summary_list = []
    config_list = []
    name_list = []
    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)
        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k,v in run.config.items()
                if not k.startswith('_')})
        # .name is the human-readable name of the run.
        name_list.append(run.name)
        print(name_list)

    # print(summary_list)
    print(config_list)
    # print(name_list)
