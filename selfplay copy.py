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
from datetime import datetime
import os

# from laserhockey.hockey_env import BasicOpponent
from client.remoteControllerInterface import RemoteControllerInterface
from client.backend.client import Client

timestep_max = 350
   

class ScriptedAgent():
    def __init__(self):
        self.player = h_env.BasicOpponent(weak=False)
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

def fill_replay_buffer(agents,env,max_samples,idx):
    print("filling replay buffer for each agent with ",max_samples," samples ... may take a while")
    num_samples = 0
    while True:
        while True: # continous loop for visualization
            ob1, _info = env.reset()
            ob2 = env.obs_agent_two()
            # Incorporate  Acceleration - keep one past observation for derivative of velocities
            past_obs1 = ob1.copy()
            past_obs2 = ob2.copy()
            total_reward=0

            agent1_touch_puck = []
            agent2_touch_puck = []
            temp_buffer=[]
            A_puck_got_touched = False
            for t in range(350):
                
                a1, a1_s = act(ob1,past_obs1,agents,config_agents,idx)
                a2 = scripted_agent.act(ob2)
                a2_s = a2
                # a2, a2_s = act(ob2,past_obs2,agents,config_agents,idx_scripted)

                (ob_new1, reward, done, trunc, _info) = env.step(np.hstack([a1,a2]))
                ob_new2 = env.obs_agent_two()


                agent1_touch_puck.append(env._get_info()["reward_touch_puck"])
                agent2_touch_puck.append(env.get_info_agent_two()["reward_touch_puck"])
                
                # The simple reward only returns 10 and -10 for goals
                if args_main.simple_reward:
                    reward1 = env._compute_reward()
                    reward2 = - reward1
                # The standard reward also considers closeness to puck
                else: 
                    reward1 = reward
                    reward2 = env.get_reward_agent_two(env.get_info_agent_two())

                # The dense reward also considers the direction of the puck - not used anymore
                # if(self._config["dense_reward"]): 
                #     reward = reward + _info["reward_closeness_to_puck"] + _info["reward_touch_puck"] + _info["reward_puck_direction"]
                total_reward+= reward
                temp_buffer.append((ob1,ob2,past_obs1,past_obs2,a1_s,a2_s,reward1,reward2,ob_new1,ob_new2,done,done))                      
                
                # added_transitions += 1
                past_obs1 = ob1
                past_obs2 = ob2
                ob1=ob_new1
                ob2=ob_new2
            
                if done or trunc: break
            
            if sum(agent1_touch_puck) + sum(agent2_touch_puck) > 0.: A_puck_got_touched = True
            if A_puck_got_touched : 
                num_samples += len(temp_buffer)
                for data in temp_buffer:
                    store_transition(agents,config_agents,idx,data[0*2],data[1*2],data[2*2],data[3*2],data[4*2],data[5*2])
                break
        if max_samples < num_samples: return print("\t done")

def add_derivative(obs,pastobs):
    return np.append(obs,(obs-pastobs)[derivative_indices])
        
def store_transition(agents,config_agents,idx,obs,pastobs,action,reward,next_obs,done):
            if config_agents[idx]["use_derivative"]:    agents[idx].store_transition((add_derivative(obs,pastobs),action,reward,add_derivative(next_obs,obs),done))
            else :                                      agents[idx].store_transition((obs,action,reward,next_obs,done))

def act(obs,pastobs,agents,config_agents,idx):
    if config_agents[idx]["use_derivative"]:    a_s = agents[idx].act(add_derivative(obs,pastobs))
    else :                                      a_s = agents[idx].act(obs)
    if config_agents[idx]["algo"] == "dqn":     a = discrete_to_continous_action(int(a_s))
    else:                                       a = a_s
    return a, a_s
def validate(agents,names, idx1, idx2,val_episodes,max_timesteps):
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
            
            a1, a1_s = act_val(ob1,past_obs1,agents,config_agents,idx1)
            a2, a2_s = act_val(ob2,past_obs2,agents,config_agents,idx2)

            (ob_new1, reward, done, trunc, _info) = env.step(np.hstack([a1,a2]))
            ob_new2 = env.obs_agent_two()

            reward = env._compute_reward()/10

            agent1_touch_puck.append(env._get_info()["reward_touch_puck"])
            agent2_touch_puck.append(env.get_info_agent_two()["reward_touch_puck"])

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
    print("\t win rate ",names[idx1], " vs ",names[idx2], ": ",np.round(win_rate,2), " - draws: ",draw_rate, ", touch_rate: ",touch_rate, ", draw_rate_with_touch: ",draw_rate_with_touch," max length: ",np.array(length).max(), " avg length: ",np.array(length).mean())
    return win_rate, draw_rate


def train(agents, config_agents,names, env, iter_fit, max_episodes_per_pair, max_timesteps, log_interval,save_interval,val_episodes,tables,all_agains_one,loner_idx):
    
    num_agents = len(agents)
    if all_agains_one:
        win_rates = [[]]*(num_agents-1)
        win_rate_steps = [0]*(len(names)-1) # adjust step for plotting in wandb logging
    else:
        win_rates = np.empty((num_agents,num_agents-1,1)).tolist()
        for i in range(num_agents):
            for j in range(num_agents-1):
                win_rates[i][j].pop()
        win_rate_steps = [0]*len(names) # adjust step for plotting in wandb logging

    if all_agains_one:
        print("All against One")
        pairings = [(loner_idx,x) for x in range(num_agents) if x != loner_idx]
    else:
        pairings = list(itertools.combinations(range(num_agents), 2))

    cycle = 0
    current_pairing = 0
    wandb_steps = [0]*len(pairings) # adjust step for plotting in wandb logging
    

    # Set up wandb logging metrics
    if args_main.wandb:
        if all_agains_one:
            for i in range(num_agents -1):
                wandb.define_metric(names[i]+"_step")
                wandb.define_metric(names[i]+"_loss", step_metric=names[i]+"_step")
                wandb.define_metric(names[i]+"_reward", step_metric=names[i]+"_step")
                wandb.define_metric(names[i]+"length", step_metric=names[i]+"_step")
                wandb.define_metric(names[i]+"win_rate_step")
                wandb.define_metric(names[i]+"win_rate", step_metric=names[i]+"win_rate_step")
        else:   
            for pair in pairings:
                idx1, idx2 = pair
                wandb.define_metric(names[idx1] +"-"+ names[idx2]+"_step")
                wandb.define_metric(names[idx1]+"_loss", step_metric=names[idx1] +"-"+ names[idx2]+"_step")
                wandb.define_metric(names[idx2]+"_loss", step_metric=names[idx1] +"-"+ names[idx2]+"_step")
                wandb.define_metric(names[idx1] +"-"+ names[idx2] +"_reward", step_metric=names[idx1] +"-"+ names[idx2]+"_step")
                wandb.define_metric(names[idx1] +"-"+ names[idx2] +"length", step_metric=names[idx1] +"-"+ names[idx2]+"_step")
            for i,name in enumerate(names):
                wandb.define_metric(name+"win_rate_step")
                wandb.define_metric(name+"win_rate", step_metric=name+"win_rate_step")

    # fill replay buffer with runs with scripted agent
    if args_main.replay_buffer_fill:
        buffer = 1000000
        ratio = args_main.replay_buffer_fill_ratio
        if all_agains_one:
            fill_replay_buffer(agents,env,buffer//ratio,loner_idx)
        else:
            for idx in range(num_agents):
                fill_replay_buffer(agents,env,buffer//ratio,idx)

    while True: # stop manually
        # randomly get pairing of agents
        # idx1, idx2 = random.sample(range(len(agents)), 2)

        # go orderly through all combinations of agent pairings
        current_pairing_idx = current_pairing % len(pairings)
        idx1, idx2 = pairings[current_pairing_idx]
        timesteps = [max_timesteps]*len(pairings)

        
        # if args_main.visualize: 
        print(names[idx1]," vs ",names[idx2])
        
        # log the last 2 validations (pre and post) to wandb at the start of a new pairing cycle
        if current_pairing_idx == 0 and current_pairing != 0:
            if args_main.wandb:
                if all_agains_one:
                    for l in [2,1]:
                        log_dict = dict()
                        for j in range(num_agents-1):
                            last_row = win_rates[j][-l]
                            win_rate_steps[j] += 1
                            log_dict[names[j]+"win_rate"] = round(last_row)
                            log_dict[names[j]+"win_rate_step"] = win_rate_steps[j]
                        wandb.log(log_dict)
                else:
                    for l in [2,1]:
                        log_dict = dict()
                        for i,table in enumerate(tables):
                            last_row = np.array(win_rates[i])[:,-l]
                            win_rate_steps[i] += 1
                            table.add_data(*last_row)
                            log_dict[names[i]+"win_rate"] = round(last_row.mean(),4)
                            log_dict[names[i]+"win_rate_step"] = win_rate_steps[i]
                        wandb.log(log_dict)
        
        # save replay buffers
        if cycle % args_main.save_buffer_interval == 0:
            date_str = datetime.today().strftime('%Y-%m-%dT%H.%M')
            
            for i,agent in enumerate(agents):
                agent.save_buffer("self-play/buffers/"+run_name+"/"+names[i]+"_buffer_"+str(cycle)+"_"+date_str) 

        if current_pairing_idx == 0:
            print("\n------------------ new Cycle ------------------\n")
            cycle += 1

        # inital validation:    
        print("Pre Validating...")
        win_rate, draw_rate = validate(agents,names, idx1, idx2,val_episodes,timesteps[current_pairing_idx])
        #if draw rate is too high or to low, we adjust the timesteps to drive episodes to conclusion
        if draw_rate > 0.10 and timesteps[current_pairing_idx] <= timestep_max : timesteps[current_pairing_idx] += 50
        if draw_rate < 0.05: timesteps[current_pairing_idx] -= 25
        if all_agains_one:
            win_rates[idx2].append(win_rate)
        else:
            # the array doesn't contain the diagonal (win_rate to it self) so we need to shift indices
            if idx1 < idx2 :   idx2_w = idx2 -1 ; idx1_w = idx1
            else:              idx2_w = idx2    ; idx1_w = idx1 -1
            win_rates[idx1][idx2_w].append(win_rate)
            win_rates[idx2][idx1_w].append(1-win_rate)
        
        # logging variables
        rewards = []
        lengths = []
        losses = []
        
        # training loop
        for i_episode in range(1, max_episodes_per_pair+1):

            while True: # continous loop for visualization
                ob1, _info = env.reset()
                ob2 = env.obs_agent_two()
                # Incorporate  Acceleration - keep one past observation for derivative of velocities
                past_obs1 = ob1.copy()
                past_obs2 = ob2.copy()
                total_reward=0
                added_transitions = 0

                agent1_touch_puck = []
                agent2_touch_puck = []
                temp_buffer=[]
                A_puck_got_touched = False
                for t in range(timesteps[current_pairing_idx]):
                    
                    a1, a1_s = act(ob1,past_obs1,agents,config_agents,idx1)
                    a2, a2_s = act(ob2,past_obs2,agents,config_agents,idx2)

                    (ob_new1, reward, done, trunc, _info) = env.step(np.hstack([a1,a2]))
                    ob_new2 = env.obs_agent_two()


                    agent1_touch_puck.append(env._get_info()["reward_touch_puck"])
                    agent2_touch_puck.append(env.get_info_agent_two()["reward_touch_puck"])
                    
                    # The simple reward only returns 10 and -10 for goals
                    simple_reward = env._compute_reward()
                    if args_main.simple_reward:
                        reward1 = simple_reward
                        reward2 = - reward1
                    # The standard reward also considers closeness to puck
                    else: 
                        reward1 = reward
                        reward2 = env.get_reward_agent_two(env.get_info_agent_two())

                    # The dense reward also considers the direction of the puck - not used anymore
                    # if(self._config["dense_reward"]): 
                    #     reward = reward + _info["reward_closeness_to_puck"] + _info["reward_touch_puck"] + _info["reward_puck_direction"]
                    
                    # Don't log standard reward, we can save one plot per pairing that way (standard reward is not symmetric)
                    total_reward+= simple_reward
                    
                    if not args_main.visualize:
                        temp_buffer.append((ob1,ob2,past_obs1,past_obs2,a1_s,a2_s,reward1,reward2,ob_new1,ob_new2,done,done))
                    else:
                        env.render()
                        time.sleep(args_main.sleep)                        
                    
                    # added_transitions += 1
                    past_obs1 = ob1
                    past_obs2 = ob2
                    ob1=ob_new1
                    ob2=ob_new2
                
                    if done or trunc: break
                
                if sum(agent1_touch_puck) + sum(agent2_touch_puck) > 0.: A_puck_got_touched = True
                if A_puck_got_touched and not args_main.visualize: 
                    added_transitions = len(temp_buffer)
                    for data in temp_buffer:
                        store_transition(agents,config_agents,idx1,data[0*2],data[1*2],data[2*2],data[3*2],data[4*2],data[5*2])
                        if not args_main.train_only_one:
                            store_transition(agents,config_agents,idx2,data[0*2+1],data[1*2+1],data[2*2+1],data[3*2+1],data[4*2+1],data[5*2+1])
                    break

            if(args_main.replay_ratio != 0.):
                iter_fit = int(added_transitions * args_main.replay_ratio) + 1  

            l1 = agents[idx1].train_innerloop(iter_fit)
            if not args_main.train_only_one:
                l2 = agents[idx2].train_innerloop(iter_fit)

            rewards.append(total_reward)
            lengths.append(t)

            # logging
            if args_main.wandb: 
                wandb_steps[current_pairing_idx] += 1
                if not args_main.train_only_one:
                    wandb.log({names[idx1]+"_loss": np.array(l1[0]).mean() , names[idx2]+"_loss": np.array(l2[0]).mean() ,
                            names[idx1] +"-"+ names[idx2] +"_reward": total_reward, names[idx1] +"-"+ names[idx2] +"length":t,
                            names[idx1] +"-"+ names[idx2]+"_step":wandb_steps[current_pairing_idx] })
                else:
                    wandb.log({names[idx1]+"_loss": np.array(l1[0]).mean(),
                            names[idx1] +"-"+ names[idx2] +"_reward": total_reward, names[idx1] +"-"+ names[idx2] +"length":t,
                            names[idx1] +"-"+ names[idx2]+"_step":wandb_steps[current_pairing_idx] })
            
            # save models
            if i_episode % save_interval == 0:
                agents[idx1].save_agent_wandb(wandb_steps[current_pairing_idx], rewards, lengths, losses,"sp-"+names[idx1])
                if not args_main.train_only_one: 
                    agents[idx2].save_agent_wandb(wandb_steps[current_pairing_idx], rewards, lengths, losses,"sp-"+names[idx2])           

            # logging
            if i_episode % log_interval == 0:
                avg_reward = np.mean(rewards[-log_interval:])
                avg_length = int(np.mean(lengths[-log_interval:]))
                print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, avg_reward))

        # validate after training
        print("Post Validating...")
        win_rate, draw_rate = validate(agents, names, idx1, idx2,val_episodes,timesteps[current_pairing_idx])
        #if draw rate is too high or to low, we adjust the timesteps
        if draw_rate > 0.10 and timesteps[current_pairing_idx] <= timestep_max : timesteps[current_pairing_idx] += 50
        if draw_rate < 0.05: timesteps[current_pairing_idx] -= 25
        if all_agains_one:
            win_rates[idx2].append(win_rate)
        else:
            # the array doesn't contain the diagonal (win_rate to it self) so we need to shift indices
            if idx1 < idx2 :   idx2_w = idx2 -1 ; idx1_w = idx1
            else:              idx2_w = idx2    ; idx1_w = idx1 -1
            win_rates[idx1][idx2_w].append(win_rate)
            win_rates[idx2][idx1_w].append(1-win_rate)
                
        # Prepare next pair
        print("\n") 
        current_pairing += 1       
    return losses


if __name__ == '__main__':

    # Load agent config from files
    parser_main = argparse.ArgumentParser()
    parser_main.add_argument('--agents', nargs='+', help='json config files defining an agent', required=True)
    parser_main.add_argument('--max_episodes_per_pair',help='how many episodes should be spend training before switching agents. The training only terminates manually with Strg+C.', default=1000000, type=int)
    parser_main.add_argument('--log_interval', default=20, type=int)
    parser_main.add_argument('--save_interval', help='when should a model be saved in terms of episodes_per_pair. Should be less or equal to episodes_per_pair.', default=5000, type=int)
    parser_main.add_argument('--max_timesteps', default=350, type=int)
    parser_main.add_argument('--iter_fit', default=10, type=int)
    parser_main.add_argument('--replay_ratio', default=0.25, type=float)
    parser_main.add_argument('--notes', default="",type=str)
    parser_main.add_argument('--wandb', action="store_true")
    parser_main.add_argument('--visualize', action="store_true")
    parser_main.add_argument('-s','--sleep', default=0., type=float)
    parser_main.add_argument('--simple_reward', action="store_true")
    parser_main.add_argument('--val_episodes', default=20, type=int)
    parser_main.add_argument('-g','--all_against_one', default=False, type=str)
    parser_main.add_argument('--all_against_one_bootstrap', default=False, type=str)
    parser_main.add_argument('--scripted_agent', action="store_true")
    parser_main.add_argument('--replay_buffer_fill', action="store_true")
    parser_main.add_argument('--replay_buffer_fill_ratio', default=100, type=int)
    parser_main.add_argument('--save_buffer_interval', default=10000, type=int)
    parser_main.add_argument('-b','--bootstrap_overwrite', nargs='+', help='json config files defining an agent', default=False)
    parser_main.add_argument('--train_only_one', action="store_true")
    parser_main.add_argument('--cpu', action="store_true")
    parser_main.add_argument('--collect_buffer_from_run', default=None, type=str)
    parser_main.add_argument('--buffer_identifyer', default="", type=str)
    

    args_main = parser_main.parse_args()
    # catch wrong save_interval
    if args_main.max_episodes_per_pair > args_main.save_interval : print("!!!!!!!!! max_episodes_per_pair > save_interval !!!!!!!!!\nnothing gets saved!\n")
    
    # Start
    print('self-play started with this configuration: ')
    print(args_main)

    config_wandb = vars(args_main).copy()
    # for key in ['notes','tags','wandb']:del config_wandb[key]
    # del config_wandb
    # if args["wandb_resume is not None :
    if args_main.wandb: 
        wandb_run = wandb.init(project="self-play",
            config=config_wandb,
            notes=" - ".join(args_main.agents),
            # resume="must",
            # id=args["wandb_resume
            )
    else: wandb_run = None

    if args_main.wandb: 
        run_name = wandb_run.name
    else:
          run_name = str(np.random.randint(100))
    Path("self-play/buffers/"+run_name).mkdir(parents=True, exist_ok=True)

    # scripted agent
    config_scripted = {"algo": "scripted","use_derivative":False}
    name_scripted = "scripted_agent"
    scripted_agent = ScriptedAgent()

    config_agents = []
    agents = []
    names = []

    print("\nInstantiate Agents\n")
    bootstrap=False
    if (args_main.bootstrap_overwrite):
        if len(args_main.bootstrap_overwrite) == len(args_main.agents):
            print("\n Bootstraping \n")
            bootstrap=True
        else:
            print("bootstrap list doesn't match agent list")
    for i, file in enumerate(args_main.agents):
        names.append(Path(file).stem)
        with open(file, 'r') as f:
            config = json.load(f)
            config_agents.append(config) 
            # Instantiate agents
            if (bootstrap):
                if args_main.bootstrap_overwrite[i] == "None":
                    agents.append(instanciate_agent(config,wandb_run,cpu=args_main.cpu))
                else: agents.append(instanciate_agent(config,wandb_run,args_main.bootstrap_overwrite[i],cpu=args_main.cpu))
            else:
                agents.append(instanciate_agent(config,wandb_run,cpu=args_main.cpu))

    # scripted agent
    if args_main.scripted_agent:
        names.append(name_scripted)
        config_agents.append(config_scripted)  
        agents.append(scripted_agent)   

    # All against one
    if args_main.all_against_one:
        names.append(Path(args_main.all_against_one).stem )
        with open(args_main.all_against_one, 'r') as f:
            config = json.load(f)
            config_agents.append(config) 
            if args_main.all_against_one_bootstrap:
                agents.append(instanciate_agent(config,wandb_run,args_main.all_against_one_bootstrap,cpu=args_main.cpu))
            else:
                agents.append(instanciate_agent(config,wandb_run,cpu=args_main.cpu))
        loner_idx = len(names)-1
        all_agains_one = True
        print("\nAll against one !\n")
    else: 
        all_agains_one = False
        loner_idx = None

    # collecting buffers
    if args_main.collect_buffer_from_run is not None:
        print("\nCollecting buffer from run: ",args_main.collect_buffer_from_run,"\n")
        buffer_path = "self-play/buffers/"+args_main.collect_buffer_from_run
        buffer_files = [f for f in os.listdir(buffer_path) if os.path.isfile(os.path.join(buffer_path, f))]
        buffer_files.sort()
        buffer_files = buffer_files[1:]
        for j,name in enumerate(names):
            found=False
            for i, file in enumerate(buffer_files):
                if name in file and args_main.buffer_identifyer in file:
                    found=True
                    print("for agent ",name)
                    agents[j].load_buffer(buffer_path+"/"+file)
                    print("\n")
            if not found:
                print("no buffer found for agent: ",name,"\n")
        print("done collecting buffer")

    

    # print agent configs
    for i, agent_config in enumerate(config_agents):
        print(names[i])
        print(agent_config)
        print("\n") 

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

    try:
        
        if args_main.wandb:
            tables = [wandb.Table(columns=(names[:i] + names[i+1:])) for i in range(len(agents))]
        else :tables = None
        train(agents, config_agents,names, env, args_main.iter_fit, args_main.max_episodes_per_pair, args_main.max_timesteps, args_main.log_interval,args_main.save_interval,args_main.val_episodes,tables,all_agains_one,loner_idx)
    finally:
        print("closing script")
        # Save replay buffers
        for i,agent in enumerate(agents):
            date_str = datetime.today().strftime('%Y-%m-%dT%H.%M')
            agent.save_buffer("self-play/buffers/"+run_name+"/"+names[i]+"_buffer_end_"+date_str)
        # Save agents
        for i,agent in enumerate(agents):
            date_str = datetime.today().strftime('%Y-%m-%dT%H.%M')
            agent.save_agent_wandb("end", "_", "_", date_str,"sp-"+names[i])

        if wandb_run:
            log_dict = dict()
            for i,table in enumerate(tables):
                log_dict[names[i]] = table
            wandb_run.log(log_dict)
           
            # wandb.log({"my_custom_id" : wandb.plot.line_series(
            #                     xs=[0, 1, 2, 3, 4], 
            #                     ys=[[10, 20, 30, 40, 50], [0.5, 11, 72, 3, 41]],
            #                     keys=["metric Y", "metric Z"],
            #                     title="Two Random Metrics",
            #                     xname="x units")})

