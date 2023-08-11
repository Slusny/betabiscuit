import argparse
import json
import gymnasium as gym
import sys
from pathlib import Path
# import laserhockey.hockey_env as 
import hockey_env as h_env
from importlib import reload
import wandb
import torch
import numpy as np
import time
import random
import itertools
import math
import pandas as pd 
from utility import save_checkpoint
from utility import instanciate_agent


class ScriptedAgent():
    def __init__(self,weak=False):
        self.player = h_env.BasicOpponent(weak)
    def act(self,obs,eps=0.0):
        return self.player.act(obs)
    
    def save_agent_wandb(self,x1,x2,x3,x4,x5):
        return

    def train_innerloop(self,x):
        return (0,0)

    def store_transition(self,x):
        return
    
    def save_buffer(self,x):
        return

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

def instanciate_agent_old(args,wandb_run,bootstrap_overwrite):
    
    if bootstrap_overwrite is not None:
        args["bootstrap"] = bootstrap_overwrite

    # creating environment
    env_name = args["env_name"]
    if env_name == "lunarlander":
        env = gym.make("LunarLander-v2", continuous = True)
        action_n = env.action_space.shape[0]
    elif env_name == "pendulum":
        env = gym.make("Pendulum-v1")
        action_n = env.action_space.shape[0]
    elif env_name == "hockey":
        env = h_env.HockeyEnv()
        action_n = 4
        # vx1, vy1, rot1, vx2, vy2, rot2, puck_vx, puck_vy
        if args["use_derivative"]:
            derivative_indices = [3,4,5,9,10,11,14,15]
        else:
            derivative_indices = []
    elif env_name == "hockey-train-shooting":
        # reload(h_env)
        env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)
        if args["use_derivative"]:
            derivative_indices = [3,4,5,9,10,11,14,15]
        else:
            derivative_indices = []
        action_n = 4
    elif env_name == "hockey-train-defense":
        env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE)
        action_n = 4
        if args["use_derivative"]:
            derivative_indices = [3,4,5,9,10,11,14,15]
        else:
            derivative_indices = []
    else:
        env = gym.make(env_name)

    #create save path
    Path(args["savepath"]).mkdir(parents=True, exist_ok=True)

    if args["algo"] == "ddpg":
        sys.path.insert(0,'./DDPG')
        from DDPG import DDPGAgent
        sys.path.pop(0)
        agent = DDPGAgent(env, env_name, action_n, args["seed"], args["savepath"], wandb_run,
                        eps = args["eps"], 
                        update_target_every = args["update_every"],
                        # past_states = args["past_states,
                        derivative = args["use_derivative"],
                        derivative_indices = derivative_indices,
                        buffer_size=args["buffer_size"],
                        discount=args["discount"],
                        batch_size=args["batch_size"],
                        learning_rate_actor = args["learning_rate_actor"],
                        learning_rate_critics=args["learning_rate_critic"],
                        hidden_sizes_actor=eval(args["hidden_sizes_actor"]),
                        hidden_sizes_critic=eval(args["hidden_sizes_critic"]),
                        per=args["per"],
                        dense_reward=args["dense_reward"],
                        bootstrap=args["bootstrap"],
                        legacy=args["legacy"],
                        bc=args["bc"],
                        bc_lambda=args["bc_lambda"],
                        cpu=args["cpu"],
                        replay_ratio=args["replay_ratio"],
                        )
    elif args["algo"] == "td3":
        sys.path.insert(0,'./TD3')
        from TD3 import TD3Agent
        sys.path.pop(0)
        agent = TD3Agent(env, env_name, action_n, args["seed"], args["savepath"], wandb_run,
                        eps = args["eps"], 
                        update_target_every = args["update_every"],
                        # past_states = args["past_states,
                        derivative = args["use_derivative"],
                        derivative_indices = derivative_indices,
                        buffer_size=args["buffer_size"],
                        discount=args["discount"],
                        batch_size=args["batch_size"],
                        learning_rate_actor = args["learning_rate_actor"],
                        learning_rate_critic=args["learning_rate_critic"],
                        hidden_sizes_actor=eval(args["hidden_sizes_actor"]),
                        hidden_sizes_critic=eval(args["hidden_sizes_critic"]),
                        tau=args["tau"],
                        policy_noise=args["policy_noise"],
                        noise_clip=args["noise_clip"],
                        per=args["per"],
                        dense_reward=args["dense_reward"],
                        bootstrap=args["bootstrap"],
                        HiL=args["hil"],
                        bc=args["bc"],
                        bc_lambda=args["bc_lambda"],
                        cpu=args["cpu"],
                        replay_ratio=args["replay_ratio"],
                        batchnorm=args["batchnorm"],
                        validation_episodes=args["validation_episodes"],
                        validation_interval=args["validation_interval"],
                        filled_buffer_ratio=args["filled_buffer_ratio"],
                        )
    elif args["algo"] == "dqn":
        sys.path.insert(0,'./DQN')
        from DQN import DQNAgent
        sys.path.pop(0)
        agent = DQNAgent(env, env_name, 12 , args["seed"], args["savepath"], wandb_run,
                        eps = args["eps"], 
                        update_target_every = args["update_every"],
                        # past_states = args["past_states,
                        derivative = args["use_derivative"],
                        derivative_indices = derivative_indices,
                        buffer_size=args["buffer_size"],
                        discount=args["discount"],
                        batch_size=args["batch_size"],
                        learning_rate=args["lr"],
                        hidden_sizes=eval(args["hidden_sizes"]),
                        hidden_sizes_values=eval(args["hidden_sizes_values"]),
                        hidden_sizes_advantages=eval(args["hidden_sizes_advantages"]),
                        tau=args["tau"],
                        per=args["per"],
                        dense_reward=args["dense_reward"],
                        bootstrap=args["bootstrap"],
                        bc=args["bc"],
                        bc_lambda=args["bc_lambda"],
                        cpu=args["cpu"],
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
    return agent


def add_derivative(obs,pastobs):
    return np.append(obs,(obs-pastobs)[derivative_indices])
        

def validate(agents,names, idx1, idx2,val_episodes,max_timesteps,visualize,sleep):
    def act_val(obs,pastobs,agents,config_agents,idx):
        if config_agents[idx]["use_derivative"]:    a_s = agents[idx].act(add_derivative(obs,pastobs),eps=0.)
        else :                                      a_s = agents[idx].act(obs,eps=0.)
        if config_agents[idx]["algo"] == "dqn":     a = discrete_to_continous_action(int(a_s))
        else:                                       a = a_s
        return a, a_s

    length = []
    rewards = []
    touches = []
    for i_episode in range(1, val_episodes+1):
        ob1, _info = env.reset()
        ob2 = env.obs_agent_two()
        # Incorporate  Acceleration
        past_obs1 = ob1.copy()
        past_obs2 = ob2.copy()
        total_reward=0
        agent1_touch_puck = []
        agent2_touch_puck = []
        for t in range(max_timesteps):

            if visualize: env.render()
            if sleep: time.sleep(sleep)
            
            a1, a1_s = act_val(ob1,past_obs1,agents,config_agents,idx1)
            a2, a2_s = act_val(ob2,past_obs2,agents,config_agents,idx2)

            (ob_new1, reward, done, trunc, _info) = env.step(np.hstack([a1,a2]))
            ob_new2 = env.obs_agent_two()

            agent1_touch_puck.append(env._get_info()["reward_touch_puck"])
            agent2_touch_puck.append(env.get_info_agent_two()["reward_touch_puck"])

            reward = env._compute_reward()/10

            total_reward+= reward
            past_obs1 = ob1
            past_obs2 = ob2
            ob1=ob_new1
            ob2=ob_new2
            if done or trunc: break
        rewards.append(total_reward)
        length.append(t)
        touches.append((sum(agent1_touch_puck) + sum(agent2_touch_puck)) > 0.0)
    win_rate = np.array(rewards)
    draw_rate = np.sum(win_rate == 0) / win_rate.size
    draw_rate_with_touch = ((win_rate[touches] + 1) / 2).mean().round(2)
    touch_rate = round(sum(touches) / len(touches),2)
    win_rate = (win_rate + 1 ) /2
    win_rate = win_rate.mean().round(3)
    print("\t win rate ",names[idx1], " vs ",names[idx2], ": ",np.round(win_rate,2), " - draws: ",draw_rate, ", touch_rate: ",touch_rate, ", draw_rate_with_touch: ",draw_rate_with_touch, " max length: ",np.array(length).max(), " avg length: ",np.array(length).mean())
    return win_rate, draw_rate

def validation(args_main,agents, config_agents, names, env,val_episodes,visualize,sleep,maxtimesteps):
    
    num_agents = len(agents)
    win_rates = np.empty((num_agents,num_agents-1)).tolist()
    # for i in range(num_agents):
    #     for j in range(num_agents-1):
    #         win_rates[i][j].pop()

    pairings = list(itertools.combinations(range(num_agents), 2))

    for i in range(len(pairings)):
        # go orderly through all combinations of agent pairings
        idx1, idx2 = pairings[i]
        
        # if args_main.visualize: 
        print(names[idx1]," vs ",names[idx2])

        # inital validation:    
        print("Validating...")
        win_rate, draw_rate = validate(agents,names, idx1, idx2,val_episodes,maxtimesteps,visualize,sleep)
        # the array doesn't contain the diagonal (win_rate to it self) so we need to shift indices
        if idx1 < idx2 :   idx2_w = idx2 -1 ; idx1_w = idx1
        else:              idx2_w = idx2    ; idx1_w = idx1 -1
        # win_rates[idx1][idx2_w].append(win_rate)
        # win_rates[idx2][idx1_w].append(1-win_rate)
        win_rates[idx1][idx2_w] = win_rate
        win_rates[idx2][idx1_w] = 1-win_rate
    print("\n")
    print("\n")
    win_rates_CSV = win_rates.copy()
    for i in range(num_agents):
        last_row = np.array(win_rates[i])
        print(names[i] + " overall win rate: ",round(last_row.mean(),4))
        own = 0
        for j in range(num_agents -1):
            if i  == j: own = 1
            print("\t",names[j+own],": ",round(last_row[j],4)) 
        win_rates_CSV[i].insert(i,0)
    if not args_main.visualize:
        df = pd.DataFrame(win_rates_CSV)
        if args_main.csv_file_name != "":
            file = args_main.csv_file_name
        else: file = "-".join(names)+".csv"
        df.to_csv("self-play/eval/"+file, header=names, index=names)
    return


if __name__ == '__main__':

    # Load agent config from files
    parser_main = argparse.ArgumentParser()
    parser_main.add_argument('-a','--agents', nargs='+', help='json config files defining an agent', required=True)
    parser_main.add_argument('-b','--bootstrap_overwrite', nargs='+', help='json config files defining an agent')
    parser_main.add_argument('-v','--visualize', action="store_true")
    parser_main.add_argument('-s','--sleep', default=0., type=float)
    parser_main.add_argument('--val_episodes', default=50, type=int)
    parser_main.add_argument('--max_timesteps', default=350, type=int)
    parser_main.add_argument('--weak', action="store_true")
    parser_main.add_argument('--strong', action="store_true")

    parser_main.add_argument('--csv_file_name', default="", type=str)


    args_main = parser_main.parse_args()

    # Start
    print('self-play started with this configuration: ')
    print(args_main)

    config_agents = []
    agents = []
    names = []
    for i, file in enumerate(args_main.agents):
        string = Path(file).stem 
        if args_main.bootstrap_overwrite :
            string += "_" + args_main.bootstrap_overwrite[i]
        names.append(string)
        with open(file, 'r') as f:
            config = json.load(f)
            config_agents.append(config) 
            # instanciate agents
            if args_main.bootstrap_overwrite :
                agent, env = instanciate_agent(config,False,args_main.bootstrap_overwrite[i])
            else :
                agent, env = instanciate_agent(config,False,False)
            agents.append(agent)

    # print agent configs
    for i, agent_config in enumerate(config_agents):
        print(names[i])
        print(agent_config)
        print("\n") 


    # scripted agent

    
    if( args_main.weak):
        config_scripted_weak = {"algo": "scripted_weak","use_derivative":False}
        name_scripted_weak = "scripted_agent_weak"
        scripted_agent_weak = ScriptedAgent(args_main.weak)
        names.append(name_scripted_weak)
        config_agents.append(config_scripted_weak)  
        agents.append(scripted_agent_weak)

    if( args_main.strong): 
        config_scripted_strong = {"algo": "scripted_strong","use_derivative":False}
        name_scripted_strong = "scripted_agent_strong"
        scripted_agent_strong = ScriptedAgent(False)
        names.append(name_scripted_strong)
        config_agents.append(config_scripted_strong)  
        agents.append(scripted_agent_strong)

    # creating environment
    env_name = config_agents[0]["env_name"]
    if env_name == "lunarlander":
        env = gym.make("LunarLander-v2", continuous = True)
        action_n = env.action_space.shape[0]
    elif env_name == "pendulum":
        env = gym.make("Pendulum-v1")
        action_n = env.action_space.shape[0]
    elif env_name == "hockey":
        env = h_env.HockeyEnv()
        action_n = 4
        # vx1, vy1, rot1, vx2, vy2, rot2, puck_vx, puck_vy
        derivative_indices = [3,4,5,9,10,11,14,15]
    elif env_name == "hockey-train-shooting":
        # reload(h_env)
        env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)
        derivative_indices = [3,4,5,9,10,11,14,15]
        action_n = 4
    elif env_name == "hockey-train-defense":
        env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE)
        action_n = 4
        derivative_indices = [3,4,5,9,10,11,14,15]
    else:
        env = gym.make(env_name)

    validation(args_main,agents, config_agents,names, env, args_main.val_episodes,args_main.visualize,args_main.sleep,args_main.max_timesteps)
   