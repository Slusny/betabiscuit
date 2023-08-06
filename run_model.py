import argparse
import gymnasium as gym
import sys
from pathlib import Path
import laserhockey.hockey_env as h_env
import torch
sys.path.insert(0,'./DDPG')
sys.path.insert(0,'./TD3')
sys.path.insert(0,'./DQN')
from DDPG import DDPGAgent
from TD3 import TD3Agent
from DQN import DQNAgent
import wandb
import numpy as np
import pyvirtualdisplay
import time


# added more actions
def discrete_to_continous_action(discrete_action):
    ''' converts discrete actions into continuous ones (for each player)
        The actions allow only one operation each timestep, e.g. X or Y or angle change.
        This is surely limiting. Other discrete actions are possible
        Action 0: do nothing
        Action 1: -1 in x
        Action 2: 1 in x
        Action 3: -1 in y
        Action 4: 1 in y
        Action 5: -1 in angle
        Action 6: 1 in angle
        Action 7: shoot (if keep_mode is on)
        Action 8: -1 in x, -1 in y
        Action 9: -1 in x, 1 in y
        Action 10: 1 in x, -1 in y
        Action 11: 1 in x, 1 in y
        '''
    action_cont = [((discrete_action == 1) | (discrete_action == 8) | (discrete_action == 9)) * -1 + ((discrete_action == 2) | (discrete_action == 10) | (discrete_action == 11)) * 1,  # player x
                   ((discrete_action == 3) | (discrete_action == 8) | (discrete_action == 10)) * -1 + ((discrete_action == 4) | (discrete_action == 9) | (discrete_action == 11)) * 1,  # player y
                   (discrete_action == 5) * -1 + (discrete_action == 6) * 1]  # player angle
    if True: # keep_mode
      action_cont.append(discrete_action == 7)
    return action_cont


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
parser.add_argument('-l','--legacy', action='store_true')
parser.add_argument('--action_n', action='store', default=4)

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
    derivative_indices = []
else:
    if(args["use_derivative"]):
        derivative_indices = [3,4,5,9,10,11,14,15]
    else:
        derivative_indices = []


def opponent_action(obs):
    if (env_name == "hockey"):
        return player.act(obs)
    else:
        return np.array([0,0.,0,0])

#create save path
savepath = 'results_run'
Path().mkdir(parents=True, exist_ok=True)

action_n = run_args.action_n

# #test
player_normal = h_env.BasicOpponent(weak=False)
player_weak = h_env.BasicOpponent(weak=True)

if run_args.legacy and args['algo'] == "ddpg" :
    action_n = 8
    agent = DDPGAgent(env, env_name, action_n, None, args["savepath"], False,
            eps = 0.0, 
            learning_rate_actor = args["lr"],
            update_target_every = args["update_every"],
            # past_states = args.past_states,
            derivative = args["use_derivative"],
            derivative_indices = derivative_indices)
#     agent.restore_state(state)
elif args['algo'] == "ddpg":
    agent = DDPGAgent(env, env_name, action_n, args["seed"], "/home/lenny", False,
                    eps = 0.0, 
                    update_target_every = args["update_every"],
                    # past_states = args.past_states,
                    derivative = args["use_derivative"],
                    derivative_indices = derivative_indices,
                    buffer_size=args["buffer_size"],
                    discount=args["discount"],
                    batch_size=args["batch_size"],
                    learning_rate_actor = args["learning_rate_actor"],
                    learning_rate_critics=args["learning_rate_critic"],
                    hidden_sizes_actor=eval(args["hidden_sizes_actor"]),
                    hidden_sizes_critic=eval(args["hidden_sizes_critic"]),
                    bootstrap=args["bootstrap"],
                    cpu=True,
                    )
elif args['algo'] == "td3":
    agent = TD3Agent(env, env_name, action_n, args["seed"], "/home/lenny", False,
                    eps = 0.0, 
                    update_target_every = args["update_every"],
                    # past_states = args.past_states,
                    derivative = args["use_derivative"],
                    derivative_indices = derivative_indices,
                    buffer_size=args["buffer_size"],
                    discount=args["discount"],
                    batch_size=args["batch_size"],
                    learning_rate_actor = args["learning_rate_actor"],
                    learning_rate_critics=args["learning_rate_critic"],
                    hidden_sizes_actor=eval(args["hidden_sizes_actor"]),
                    hidden_sizes_critic=eval(args["hidden_sizes_critic"]),
                    bootstrap=args["bootstrap"],
                    tau=args["tau"],
                    policy_noise=args["policy_noise"],
                    noise_clip=args["noise_clip"],
                    cpu=True,
                    batchnorm=args["legacy"],
                    )
elif args['algo'] == "dqn":
    print("if you changed the layer sizes, this needs some change")
    agent = DQNAgent(env, env_name, action_n, args["seed"], "/home/lenny", False,
                    eps = 0.0, 
                    learning_rate = args["lr"],
                    update_target_every = args["update_every"],
                    # past_states = args.past_states,
                    derivative = args["use_derivative"],
                    derivative_indices = derivative_indices,
                    buffer_size=args["buffer_size"],
                    discount=args["discount"],
                    batch_size=args["batch_size"],
                    # hidden_sizes=eval(args["hidden_sizes"]),
                    # hidden_sizes_values=eval(args["hidden_sizes_values"]),
                    # hidden_sizes_advantages=eval(args["hidden_sizes_advantages"]),
                    bootstrap=args["bootstrap"],
                    tau=args["tau"],
                    per=args["per"],
                    dense_reward=args["dense_reward"],
                    bc=args["bc"],
                    bc_lambda=args["bc_lambda"],
                    cpu=True,
                    replay_ratio=args["replay_ratio"],
                    dueling=args["dueling"],
                    double=args["double"],
                    per_own_impl=args["per_own_impl"],
                    beta=args["beta"],
                    alpha=args["alpha"],
                    alpha_decay=args["alpha_decay"],
                    beta_growth=args["beta_growth"],
                    eps_decay=args["eps_decay"],
                    min_eps=args["min_eps"],
                    )

for i_episode in range(1, run_args.max_episodes+1):
    ob, _info = env.reset()
    timestep = 0
    total_reward = 0
    for t in range(run_args.max_timesteps):
        time.sleep(run_args.sleep)
        env.render()
        timestep += 1
        done = False
        obs_agent2 = env.obs_agent_two()
        a2 = opponent_action(obs_agent2)
        a = agent.act(ob,eps=0.0)
        if args['algo'] == "dqn" :
            a = discrete_to_continous_action(a)
        a = a[:4]
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

