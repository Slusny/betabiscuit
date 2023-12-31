
import gymnasium as gym
import sys
sys.path.insert(0,'../')
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
from datetime import datetime
import os
import argparse
import json

# from laserhockey.hockey_env import BasicOpponent
from client.remoteControllerInterface import RemoteControllerInterface
from client.backend.client import Client

class RemoteBasicOpponent(RemoteControllerInterface):

    def __init__(self, Agent,algo, use_derivative=False):
        RemoteControllerInterface.__init__(self, identifier='TD3')
        self.agent = Agent
        self.pastobs = None
        self.algo = algo
        self.use_derivative = use_derivative

    def remote_act(self,
            obs : np.ndarray,
           ) -> np.ndarray:
        if self.use_derivative:
            if self.newGame:
                print("new game")
                self.newGame = False
                self.pastobs = obs.copy()
            a = self.agent.act(self.add_derivative(obs,self.pastobs),eps=0.0)
            self.pastobs = obs.copy()
        else: 
            a = self.agent.act(obs,eps=0.0)
        if self.algo == "dqn":
            a = self.discrete_to_continous_action(int(a))
        # print(a)
        # print(type(a))
        return a
        return np.array([0.0,0.0,0.0,0.0])

    def add_derivative(self,obs,pastobs):
        return np.append(obs,(obs-pastobs)[[3,4,5,9,10,11,14,15]])
    

    # added more actions
    def discrete_to_continous_action(self,discrete_action):
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
        return np.array(action_cont)

def instanciate_agent(args,wandb_run,bootstrap_overwrite=None, cpu=False):
    
    if bootstrap_overwrite is not None:
        args["bootstrap"] = bootstrap_overwrite

    if cpu:
        args["cpu"] = True

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
        sys.path.insert(0,'../DDPG')
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
        sys.path.insert(0,'../TD3')
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
        sys.path.insert(0,'../DQN')
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

if __name__ == '__main__':
    parser_main = argparse.ArgumentParser()
    parser_main.add_argument('-c','--config', default="competitors/td3/celestial_lake_fallen_shape.json", help='json config files defining an agent')
    parser_main.add_argument('--cpu',action='store_true', help='json config files defining an agent')
    parser_main.add_argument('--games', default=None, help='json config files defining an agent', type=int)
    args = parser_main.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    agent = instanciate_agent(config,False,cpu=args.cpu)

    controller = RemoteBasicOpponent(agent,config["algo"],config["use_derivative"])

    # Play n (None for an infinite amount) games and quit
    client = Client(username="BetaBiscuit",
                    password="uf4Aephei0",
                    controller=controller,
                    output_path='recordings/td3', # rollout buffer with finished games will be saved in here
                    interactive=False,
                    op='start_queuing',
                    # server_addr='localhost',
                    num_games=args.games)

    # Start interactive mode. Start playing by typing start_queuing. Stop playing by pressing escape and typing stop_queueing
    # client = Client(username='user0',
    #                 password='1234',
    #                 controller=controller,
    #                 output_path='logs/basic_opponents',
    #                )
