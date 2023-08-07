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

def instanciate_agent(args,wandb_run):
    
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
        

def validate(agents, idx1, idx2,val_episodes,max_timesteps):
    def act_val(obs,pastobs,agents,config_agents,idx):
        if config_agents[idx]["use_derivative"]:    a_s = agents[idx].act(add_derivative(obs,pastobs),eps=0.)
        else :                                      a_s = agents[idx].act(obs,eps=0.)
        if config_agents[idx]["algo"] == "dqn":     a = discrete_to_continous_action(int(a_s))
        else:                                       a = a_s
        return a, a_s
    
    print("Validating...")
    length = []
    rewards = []
    for i_episode in range(1, val_episodes+1):
        ob1, _info = env.reset()
        ob2 = env.obs_agent_two()
        # Incorporate  Acceleration
        past_obs1 = ob1.copy()
        past_obs2 = ob2.copy()
        total_reward=0
        for t in range(max_timesteps):
            
            a1, a1_s = act_val(ob1,past_obs1,agents,config_agents,idx1)
            a2, a2_s = act_val(ob2,past_obs2,agents,config_agents,idx2)

            (ob_new1, reward, done, trunc, _info) = env.step(np.hstack([a1,a2]))
            ob_new2 = env.obs_agent_two()

            reward = env._compute_reward()/10

            total_reward+= reward
            past_obs1 = ob1
            past_obs2 = ob2
            ob1=ob_new1
            ob2=ob_new2
            if done or trunc: break
        rewards.append(total_reward)
        length.append(t)
    win_rate = np.array(rewards)
    draw = np.sum(win_rate == 0) / win_rate.size
    win_rate = (win_rate + 1 ) /2
    win_rate = win_rate.mean().round(3)
    print("\t win rate ",names[idx1], " vs ",names[idx2], ": ",np.round(win_rate,2), " - draws: ",draw)
    return win_rate

    print("\t avg length: ",np.array(length).mean(), ", avg reward: ",np.array(rewards).mean())
    if self.wandb.run is not None:
        wandb.log({"validation_length": np.array(length).mean(), "validation_reward": np.array(rewards).mean()})

def train(agents, config_agents,names, env, iter_fit, max_episodes_per_pair, max_timesteps, log_interval,save_interval,val_episodes,tables):
    
    num_agents = len(agents)
    win_rates = np.empty((num_agents,num_agents,1)).tolist()
    for i in range(num_agents):
        for j in range(num_agents):
            win_rates[i][j].pop()

    pairings = list(itertools.combinations(range(num_agents), 2))
    current_pairing = 0

    # Set up wandb logging metrics
    if args_main.wandb:
        for pair in pairings:
            idx1, idx2 = pair
            wandb.define_metric(names[idx1] +"-"+ names[idx2]+"_step")
            wandb.define_metric(names[idx1]+"_loss", step_metric=names[idx1] +"-"+ names[idx2]+"_step")
            wandb.define_metric(names[idx2]+"_loss", step_metric=names[idx1] +"-"+ names[idx2]+"_step")
            wandb.define_metric(names[idx1] +"-"+ names[idx2] +"_reward", step_metric=names[idx1] +"-"+ names[idx2]+"_step")
            wandb.define_metric(names[idx1] +"-"+ names[idx2] +"length", step_metric=names[idx1] +"-"+ names[idx2]+"_step")
   
    while True: # stop manually
        # randomly get pairing
        # idx1, idx2 = random.sample(range(len(agents)), 2)
        idx1, idx2 = pairings[current_pairing % len(pairings)]
        current_pairing += 1
        wandb_steps = [0]*len(pairings)
        
        # if args_main.visualize: 
        print(names[idx1]," vs ",names[idx2])
        
        # inital validation:
        win_rate = validate(agents, idx1, idx2,val_episodes,max_timesteps)
        win_rates[idx1][idx2].append(win_rate)
        win_rates[idx2][idx1].append(1-win_rate)
        if current_pairing % len(pairings) == 1 and current_pairing != 1:
            if args_main.wandb:
                log_dict = dict()
                for i,table in enumerate(tables):
                    table.add(win_rates[i][:i]+win_rates[i][i+1:])
                    log_dict[names[i]+"win_rate"] = np.array(win_rates[i])[:,-1].mean()[0]
                wandb.log({log_dict})



        # to_torch = lambda x: torch.from_numpy(x.astype(np.float32)).to(self.device)
            # logging variables
        rewards = []
        lengths = []
        losses = []
        timestep = 0

        def act(obs,pastobs,agents,config_agents,idx):
            if config_agents[idx]["use_derivative"]:    a_s = agents[idx].act(add_derivative(obs,pastobs))
            else :                                      a_s = agents[idx].act(obs)
            if config_agents[idx]["algo"] == "dqn":     a = discrete_to_continous_action(int(a_s))
            else:                                       a = a_s
            return a, a_s
        
        def store_transition(agents,config_agents,idx,obs,pastobs,action,reward,next_obs,done):
            if config_agents[idx]["use_derivative"]:    agents[idx].store_transition((add_derivative(obs,pastobs),action,reward,add_derivative(next_obs,pastobs),done))
            else :                                      agents[idx].store_transition((obs,action,reward,next_obs,done))

        # training loop
    #        fill_buffer_timesteps = self._config["buffer_size"] // self._config["filled_buffer_ratio"]
        for i_episode in range(1, max_episodes_per_pair+1):
            # validate
    #        if i_episode % self._config["validation_interval"] == 0 and timestep > fill_buffer_timesteps: self.validate()

        #     buffer_size = 1000000
        #     filled_buffer_ratio = 100
        #     fill_buffer_timesteps = buffer_size // filled_buffer_ratio
            while True:
                ob1, _info = env.reset()
                ob2 = env.obs_agent_two()
                # Incorporate  Acceleration
                past_obs1 = ob1.copy()
                past_obs2 = ob2.copy()
                total_reward=0
                added_transitions = 0
                for t in range(max_timesteps):
                    timestep += 1
                    
                    a1, a1_s = act(ob1,past_obs1,agents,config_agents,idx1)
                    a2, a2_s = act(ob2,past_obs2,agents,config_agents,idx2)

                    (ob_new1, reward, done, trunc, _info) = env.step(np.hstack([a1,a2]))
                    ob_new2 = env.obs_agent_two()
                    if args_main.simple_reward:
                        reward1 = env._compute_reward()
                        reward2 = - reward1
                    else: 
                        reward1 = reward
                        reward2 = env.get_reward_agent_two(env.get_info_agent_two())
                    # if(self._config["dense_reward"]): 
                    #     reward = reward + _info["reward_closeness_to_puck"] + _info["reward_touch_puck"] + _info["reward_puck_direction"]
                    total_reward+= reward
                    
                    if not args_main.visualize:
                        store_transition(agents,config_agents,idx1,ob1,past_obs1,a1_s,reward1,ob_new1,done)
                        store_transition(agents,config_agents,idx2,ob2,past_obs2,a2_s,reward2,ob_new2,done)
                    else:
                        env.render()
                        time.sleep(args_main.sleep)
                        # print(reward, ", real reward: ",env._compute_reward()) #," agent 2: ",env.get_reward_agent_two(env.get_info_agent_two()),
                        
                    
                    added_transitions += 1
                    past_obs1 = ob1
                    past_obs2 = ob2
                    ob1=ob_new1
                    ob2=ob_new2
                
                    if done or trunc: 
                        # if re
                        break
                if not args_main.visualize: break
                # # To fill buffer once before training
                # if(timestep > fill_buffer_timesteps and not args_main.visualize):
                #     break
                # elif timestep == fill_buffer_timesteps:                  
                #     print("Buffer filled")
                #     added_transitions = 1     

            if(args_main.replay_ratio != 0):
                iter_fit = int(added_transitions * args_main.replay_ratio) + 1  

            l1 = agents[idx1].train_innerloop(iter_fit)
            l2 = agents[idx2].train_innerloop(iter_fit)
            # losses.extend(l)

            rewards.append(total_reward)
            lengths.append(t)

            # logging
            if args_main.wandb: 
                wandb_steps[current_pairing % len(pairings)] += 1
                wandb.log({names[idx1]+"_loss": np.array(l1[0]).mean() , names[idx2]+"_loss": np.array(l2[0]).mean() ,
                           names[idx1] +"-"+ names[idx2] +"_reward": total_reward, names[idx1] +"-"+ names[idx2] +"length":t,
                            names[idx1] +"-"+ names[idx2]+"_step":wandb_steps[current_pairing % len(pairings)] })

            # save every 500 episodes
            if i_episode % save_interval == 0:
                agents[idx1].save_agent_wandb(i_episode, rewards, lengths, losses,"sp-"+names[idx1])
                agents[idx2].save_agent_wandb(i_episode, rewards, lengths, losses,"sp-"+names[idx2])

            # logging
            if i_episode % log_interval == 0:
                avg_reward = np.mean(rewards[-log_interval:])
                avg_length = int(np.mean(lengths[-log_interval:]))
                print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, avg_reward))

        # validate after training
        win_rate = validate(agents, idx1, idx2,val_episodes,max_timesteps)
        win_rates[idx1][idx2].append(win_rate)
        win_rates[idx2][idx1].append(1-win_rate)
        if current_pairing % len(pairings) == 1 and current_pairing != 1:
            if args_main.wandb:
                log_dict = dict()
                for i,table in enumerate(tables):
                    table.add(win_rates[i][:i]+win_rates[i][i+1:])
                    log_dict[names[i]+"win_rate"] = np.array(win_rates[i])[:,-1].mean()[0]
                wandb.log({log_dict})
    # if i_episode % log_interval != 0: save_checkpoint(self.state(),self.savepath,"TD3",self.env_name, i_episode, self.wandb_run, self._eps, lr, self.seed,rewards,lengths, losses)
        
    return losses


if __name__ == '__main__':

    # Load agent config from files
    parser_main = argparse.ArgumentParser()
    parser_main.add_argument('--file', nargs='+', help='<Required> Set flag', required=True)
    parser_main.add_argument('--max_episodes_per_pair', default=1000000, type=int)
    parser_main.add_argument('--log_interval', default=20, type=int)
    parser_main.add_argument('--save_interval', default=5000, type=int)
    parser_main.add_argument('--max_timesteps', default=800, type=int)
    parser_main.add_argument('--iter_fit', default=10, type=int)
    parser_main.add_argument('--replay_ratio', default=0., type=float)
    parser_main.add_argument('--notes', default="",type=str)
    parser_main.add_argument('--wandb', action="store_true")
    parser_main.add_argument('--visualize', action="store_true")
    parser_main.add_argument('-s','--sleep', default=0., type=float)
    parser_main.add_argument('--simple_reward', action="store_true")
    parser_main.add_argument('--val_episodes', default=20, type=int)


    args_main = parser_main.parse_args()

    config_wandb = vars(args_main).copy()
    # for key in ['notes','tags','wandb']:del config_wandb[key]
    # del config_wandb
    # if args["wandb_resume is not None :
    if args_main.wandb: 
        wandb_run = wandb.init(project="self-play",
            config=config_wandb,
            notes=" - ".join(args_main.file),
            # resume="must",
            # id=args["wandb_resume
            )
    else: wandb_run = None


    config_agents = []
    agents = []
    names = []
    for file in args_main.file:
        names.append(Path(file).stem)
        with open(file, 'r') as f:
            config = json.load(f)
            config_agents.append(config) 
            # instanciate agents
            agents.append(instanciate_agent(config,wandb_run))

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

#    #weights and biases
    # config_wandb = vars(args_main).copy()
    # # for key in ['notes','tags','wandb']:del config_wandb[key]
    # # del config_wandb
    # # if args["wandb_resume is not None :
    # if args_main.wandb: 
    #     wandb_run = wandb.init(project="self-play " + env_name + " - " +names[0] ,  #" - ".join(names[0])
    #         config=config_wandb,
    #         notes=args_main.notes,
    #         # resume="must",
    #         # id=args["wandb_resume
    #         )
    # else:
    try:

        if args_main.wandb:
            tables = [wandb.Table(columns=(names[:i] + names[i+1:])) for i in range(len(agents))]
        else :tables = None
        train(agents, config_agents,names, env, args_main.iter_fit, args_main.max_episodes_per_pair, args_main.max_timesteps, args_main.log_interval,args_main.save_interval,args_main.val_episodes,tables)
    finally:
        print("closing script")
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
