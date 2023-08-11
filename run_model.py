import argparse
import gymnasium as gym
import sys
from pathlib import Path
import laserhockey.hockey_env as h_env
import torch
import wandb
import numpy as np
import pyvirtualdisplay
import time
from utility import instanciate_agent
import json
from utility import save_checkpoint

sys.path.insert(0,'./DQN')
from DQN import DQNAgent
from DQN import QFunction
from DQN import DuelingQFunction

player_normal = h_env.BasicOpponent(weak=False)
player_weak = h_env.BasicOpponent(weak=True)

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

def opponent_action(obs,env_name,player):
    if (env_name == "hockey"):
        return player.act(obs)
    else:
        return np.array([0,0.,0,0])
    
def add_derivative(obs,pastobs):
            return np.append(obs,(obs-pastobs)[args["derivative_indices"]])


def parse_arguments_and_get_agent(run_args):
    # load agent config file from local storage
    if run_args.local_config != "":
        string = Path(run_args.local_config).stem
        with open(run_args.local_config, 'r') as f:
            args = json.load(f)  
    # load agent config file from local storage   
    elif run_args.project:
        string = run_args.run_name
        entity = "betabiscuit"
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

    if (run_args.bootstrap_local):
        args["bootstrap_local"] = True
    # Instantiate agent
    agent, env = instanciate_agent(args,False,run_args.bootstrap)
    
    return (agent,env,args,string)
    #create save path
    # savepath = 'results_run'
    # Path().mkdir(parents=True, exist_ok=True)

    # action_n = run_args.action_n

def run(run_args,agents,env,args_list,strings):
    if (run_args.weak_opponent):
        player = player_weak
        string_opponent = "Weak Opponent"
    else:
        player = player_normal
        string_opponent = "Normal Opponent"
    
    for i in range(len(agents)):
        string = strings[i]
        agent = agents[i]
        args = args_list[i]
        length = []
        rewards = []
        print("\n"+string +" vs " +string_opponent)
        for i_episode in range(1, run_args.max_episodes+1):
            ob, _info = env.reset()
            past_obs = ob.copy()
            total_reward = 0
            for t in range(run_args.max_timesteps):
                time.sleep(run_args.sleep)
                if not run_args.validate :env.render()
                done = False
                obs_agent2 = env.obs_agent_two()
                a2 = opponent_action(obs_agent2,args['env_name'],player)
                if args["use_derivative"]: a = agent.act(add_derivative(ob, past_obs),eps=0.0)
                else: a = agent.act(ob,eps=0.0)
                if args['algo'] == "dqn" :
                    a = discrete_to_continous_action(a)
                a = a[:4]
                (ob_new, reward, done, trunc, _info) = env.step(np.hstack([a,a2]))    
                reward = env._compute_reward()/10
                total_reward+= reward
                past_obs = ob
                ob=ob_new
                if done: 
                    length.append(t)
                    rewards.append(total_reward)
                    break
        win_rate = ((np.array(rewards) + 1 ) /2).mean().round(4)
        print("\t avg length: ",np.array(length).mean())
        print("\t avg reward: ",np.array(rewards).mean())
        print("\t win rate: ",win_rate)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v','--vir', action='store_true')
    parser.add_argument('-e','--max_episodes', type=int, default=100)
    parser.add_argument('-t','--max_timesteps', type=int, default=250)
    parser.add_argument('-p','--project', type=str, default=False)
    parser.add_argument('--local_config', type=str, default="")
    parser.add_argument('-r','--run_name', type=str, default="latest")
    parser.add_argument('--run_id', type=str, default="latest")
    parser.add_argument('-b','--bootstrap', type=str, default=False)
    parser.add_argument('--bootstrap_local', action='store_true')
    parser.add_argument('-s','--sleep', type=float, default=0., help="slow down simulation by sleep x seconds")
    parser.add_argument('-w','--weak_opponent', action='store_true')
    parser.add_argument('-l','--legacy', action='store_true')
    parser.add_argument('--action_n', action='store', default=4)
    parser.add_argument('--validate', action='store_true')

    run_args = parser.parse_args()


    if run_args.vir :
        _display = pyvirtualdisplay.Display(visible=True,  # use False with Xvfb
                    rfbport=55901, backend="xvnc", size=(700, 450))
        _display.start()

    agent,env,args,string = parse_arguments_and_get_agent(run_args)
    run(run_args,[agent],env,[args],[string])

    if run_args.vir :
        _display.stop()
