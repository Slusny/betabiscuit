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
import time


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-v','--vir', action='store_true')
parser.add_argument('-e','--max_episodes', type=int, default=2)
parser.add_argument('-t','--max_timesteps', type=int, default=100)
parser.add_argument('-p','--project', type=str, default="hockey - ddpg")
parser.add_argument('-r','--run_name', type=str, default="latest")
parser.add_argument('--run_id', type=str, default="latest")
parser.add_argument('-a','--artifact', type=str, default='model:v4')
parser.add_argument('-s','--sleep', type=float, default=0., help="slow down simulation by sleep x seconds")
parser.add_argument('-w','--weak_opponent', action='store_true')

run_args = parser.parse_args()

entity = "betabiscuit"

if run_args.vir :
    _display = pyvirtualdisplay.Display(visible=True,  # use False with Xvfb
                rfbport=55901, backend="xvnc", size=(700, 450))
    _display.start()

api = wandb.Api()
runs = api.runs(entity + "/" + run_args.project)
if (run_args.run_name == "latest"):
    args = runs[0].config
else:
    found = False
    for run in runs:
        if (run.name == run_args.run_name):
            args = run.config
            found = True
    if not found :
        print("counld find run " + run_args.run_name)
        print("available runs:")
        for run in runs:
            print(run.name)
        _display.stop()
        exit(1)

art = api.artifact(entity + "/" + run_args.project + "/" + run_args.artifact, type='model')
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
elif env_name == "hockey-train-defense":
    env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE)
else:
    env = gym.make(env_name)

if (run_args.weak_opponent):
    player = h_env.BasicOpponent(weak=True)
else :
    player = h_env.BasicOpponent(weak=False)

if not "use_derivative" in args:
    args["use_derivative"] = False

def opponent_action(obs):
    if (env_name == "hockey"):
        return player.act(obs)
    else:
        return np.array([0,0.,0,0])

#create save path
savepath = 'results_run'
Path().mkdir(parents=True, exist_ok=True)

action_n = 8
derivative_indices = []

# #test
# player_normal = h_env.BasicOpponent(weak=False)
# player_weak = h_env.BasicOpponent(weak=True)

if args['algo'] == "ddpg":
    agent = DDPGAgent(env, env_name, action_n, None, args["savepath"], False,
            eps = args["eps"], 
            learning_rate_actor = args["lr"],
            update_target_every = args["update_every"],
            # past_states = args.past_states,
            derivative = args["use_derivative"],
            derivative_indices = derivative_indices)
    agent.restore_state(state)
    

    for i_episode in range(1, run_args.max_episodes+1):
        ob, _info = env.reset()
        timestep = 0
        total_reward = 0
        for t in range(run_args.max_timesteps):
            time.sleep(run_args.sleep)
            env.render()
            timestep += 1
            done = False
            a = agent.act(ob)
            a = a[:4]
            print("a: ",a)
            a2 = opponent_action(ob)
            print("a2: ",a)
            (ob_new, reward, done, trunc, _info) = env.step(np.hstack([a,a2]))
            total_reward+= reward
            ob=ob_new
            if done: break

if run_args.vir :
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
