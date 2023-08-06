import argparse
import json
import gymnasium as gym
import sys
from pathlib import Path
import laserhockey.hockey_env as h_env
from importlib import reload
import wandb
import torch
import numpy as np
import random

def instanciate_agent(args):
    
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

    #weights and biases
    if args["wandb"]   : 
        config_wandb = vars(args).copy()
        for key in ['notes','tags','wandb']:del config_wandb[key]
        del config_wandb
        if args["wandb_resume"] is not None:
            wandb_run = wandb.init(project=env_name + " - " +args["algo"], 
                config=args,
                notes="Self-Play",
                tags="S",
                resume="must",
                id=args["wandb_resume"])
        else:
            wandb_run = wandb.init(project=env_name + " - " +args["algo"], 
                config=args,
                notes="Self-Play",
                tags="S")
    else : wandb_run = None

    #create save path
    Path(args["savepath"]).mkdir(parents=True, exist_ok=True)

    if args["algo"] == "ddpg":
        sys.path.insert(0,'./DDPG')
        from DDPG import DDPGAgent
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

def train(agents, config_agents,names, env, iter_fit, max_episodes_per_pair, max_timesteps, log_interval,save_interval):
        
    # select two agents:
    # idx1, idx2 = random.sample(range(len(agents)), 2)
    idx1, idx2 = 0,1

    # to_torch = lambda x: torch.from_numpy(x.astype(np.float32)).to(self.device)
        # logging variables
    rewards = []
    lengths = []
    losses = []
    timestep = 0

    def add_derivative(obs,pastobs):
        return np.append(obs,(obs-pastobs)[derivative_indices])
    
    def act(obs,pastobs,agents,config_agents,idx):
        if config_agents[idx]["derivative"]:    a = agents[idx].act(add_derivative(obs,pastobs))
        else :                                  a = agents[idx].act(obs)
        return a
    
    def store_transition(agents,config_agents,idx,obs,pastobs,action,reward,next_obs,done):
        if config_agents[idx]["derivative"]:    agents[idx].store_transition(add_derivative(obs,pastobs),action,reward,add_derivative(next_obs,pastobs),done)
        else :                                  agents[idx].store_transition(obs,action,reward,next_obs,done)


    # training loop
#        fill_buffer_timesteps = self._config["buffer_size"] // self._config["filled_buffer_ratio"]
    for i_episode in range(1, max_episodes_per_pair+1):
        # validate
#        if i_episode % self._config["validation_interval"] == 0 and timestep > fill_buffer_timesteps: self.validate()
        ob1, _info = env.reset()
        ob2 = env.obs_agent_two()
        # Incorporate  Acceleration
        past_obs1 = ob1.copy()
        past_obs2 = ob2.copy()
        total_reward=0
        added_transitions = 0

        buffer_size = 1000000
        filled_buffer_ratio = 100
        fill_buffer_timesteps = buffer_size // filled_buffer_ratio
        while True:
            for t in range(max_timesteps):
                timestep += 1
                
                a1 = act(ob1,past_obs1,agents,config_agents,idx1)
                a2 = act(ob2,past_obs2,agents,config_agents,idx2)

                (ob_new1, reward, done, trunc, _info) = env.step(np.hstack([a1,a2]))
                ob_new2 = env.obs_agent_two()
                # if(self._config["dense_reward"]): 
                #     reward = reward + _info["reward_closeness_to_puck"] + _info["reward_touch_puck"] + _info["reward_puck_direction"]
                total_reward+= reward
                
                store_transition(agents,config_agents,idx1,ob1,past_obs1,a1,reward,ob_new1,done)
                store_transition(agents,config_agents,idx2,ob2,past_obs2,a2,-reward,ob_new2,done)
                
                added_transitions += 1
                past_obs1 = ob1
                past_obs2 = ob2
                ob1=ob_new1
                ob2=ob_new2

                if done or trunc: break

            # To fill buffer once before training
            if(timestep > fill_buffer_timesteps):
                break
            elif timestep == fill_buffer_timesteps:                  
                print("Buffer filled")
                added_transitions = 1     

        if(args_main.replay_ratio != 0):
           iter_fit = int(added_transitions * args_main.replay_ratio) + 1  

        l1 = agents[idx1].train_innerloop(iter_fit)
        l2 = agents[idx2].train_innerloop(iter_fit)
        # losses.extend(l)

        rewards.append(total_reward)
        lengths.append(t)

        # logging
        if args_main.wandb: 
            wandb.log({names[idx1]+"_loss": np.array(l1[0]).mean() , names[idx2]+"_loss": np.array(l2[0]).mean() ,names[idx1] +"-"+ names[idx2] +"_reward": total_reward, "length":t })

        # # save every 500 episodes
        # if i_episode % save_interval == 0:
        #     save_checkpoint(self.state(),self.savepath,"TD3",self.env_name, i_episode, self.wandb_run, self._eps, lr, self.seed,rewards,lengths, losses)

        # logging
        if i_episode % log_interval == 0:
            avg_reward = np.mean(rewards[-log_interval:])
            avg_length = int(np.mean(lengths[-log_interval:]))
            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, avg_reward))

    # if i_episode % log_interval != 0: save_checkpoint(self.state(),self.savepath,"TD3",self.env_name, i_episode, self.wandb_run, self._eps, lr, self.seed,rewards,lengths, losses)
        
    return losses


if __name__ == '__main__':

    # Load agent config from files
    parser_main = argparse.ArgumentParser()
    parser_main.add_argument('--file', nargs='+', help='<Required> Set flag', required=True)
    parser_main.add_argument('--max_episodes_per_pair', default=1000000)
    parser_main.add_argument('--log_interval', default=20)
    parser_main.add_argument('--save_interval', default=5000)
    parser_main.add_argument('--max_timesteps', default=300)
    parser_main.add_argument('--iter_fit', default=10)
    parser_main.add_argument('--replay_ratio', default=0.25)
    parser_main.add_argument('--notes', default="",type=str)
    parser_main.add_argument('--wandb', action="store_true")

    args_main = parser_main.parse_args()

    config_agents = []
    agents = []
    names = []
    for file in args_main.file:
        names.append(Path(file).stem)
        with open(file, 'r') as f:
            config = json.load(f)
            config_agents.append(config) 
            # instanciate agents
            agents.append(instanciate_agent(config))

    # print agent configs
    for i, agent_config in enumerate(config_agents):
        print(names[i])
        print(agent_config)
        print("\n") 

    # creating environment
    env_name = config_agents[0].env_name
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
    config_wandb = vars(args_main).copy()
    # for key in ['notes','tags','wandb']:del config_wandb[key]
    # del config_wandb
    # if args["wandb_resume is not None :
    if args_main.wandb: 
        wandb_run = wandb.init(project="self-play: " + env_name + " - " + str(names), 
            config=config_wandb,
            notes=args_main.notes,
            # resume="must",
            # id=args["wandb_resume
            )
    # else:
    
    train(agents, config_agents,names, env, args_main.iter_fit, args_main.max_episodes_per_pair, args_main.max_timesteps, args_main.log_interval,args_main.save_interval)