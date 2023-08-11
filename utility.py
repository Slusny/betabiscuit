from datetime import datetime
import os
import pickle
import torch
import wandb
import numpy as np
import gymnasium as gym
import hockey_env as h_env
from pathlib import Path
import sys

# sys.path.insert(0,'./DQN')
# from DQN import *

def save_statistics(savepath,algo,env_name,i_episode,rewards=None,lengths=None,train_iter=None, losses=None, eps="Nan",lr="Nan",seed="Nan"):
    date_str = datetime.today().strftime('%Y-%m-%dT%H.%M')
    savepath_stats = os.path.join(savepath,f'{algo}_{env_name}_{i_episode}-eps{eps}-t{train_iter}-l{lr}-s{seed}-{date_str}-stat.pkl')
    with open(savepath_stats, 'wb') as f:
        pickle.dump({"rewards" : rewards, "lengths": lengths, "losses": losses}, f)
                
def wandb_save_model(wandb_run,savepath,notes="",episode=0):
    #print("----------- Writing Model to W&B -----------")
    artifact = wandb.Artifact(wandb_run.name + notes + '_model', type='model',metadata={"episode":episode})
    artifact.add_file(savepath)
    wandb_run.log_artifact(artifact)

def save_checkpoint(torch_state,savepath,algo,env_name,i_episode,wandb_run=None,eps="Nan",train_iter="Nan",lr="Nan",seed="Nan",rewards=None,lengths=None, losses=None,notes=""):
    print("########## Saving a checkpoint... ##########")
    date_str = datetime.today().strftime('%Y-%m-%dT%H.%M')
    savepath_checkpoint = os.path.join(savepath,f'{notes}_{algo}_{env_name}_{i_episode}-{date_str}.pth')
    torch.save(torch_state, savepath_checkpoint)
    if wandb_run : wandb_save_model(wandb_run,savepath_checkpoint,notes,i_episode)
    #save_statistics(savepath,algo,env_name,i_episode,rewards,lengths,train_iter, losses, eps,lr,seed)

def restore_from_wandb(str):
    run = wandb.init()
    artifact = run.use_artifact(str, type='model')
    artifact_dir = artifact.download()
    return artifact_dir

def transform_obs(obs,help=False):
    names = ["x pos player one",
            'y pos player one',
            'angle player one',
            'x vel player one',
            'y vel player one',
            'angular vel player one',
            'x player two',
            'y player two',
            'angle player two',
            'y vel player two',
            'y vel player two',
            'angular vel player two',
            'x pos puck',
            'y pos puck',
            'x vel puck',
            'y vel puck',
            'time left player has puck',
            'time left other player has puck'] # 18 with acceleration 24
    limits = np.array([[-3,0], #[4,0] can go behind barrier
                      [-2,2],  #[3,3]
                      [-1,1],  #[-1.25,1.25] can overshoot with momentum
                      [-10,10],
                      [-10,10],
                      [-20,20],
                      [0,3],
                      [-2,2],
                      [-1,1],
                      [-10,10],
                      [-10,10],
                      [-20,20],
                      [-3,3],
                      [-2,2],
                      [-60,60],
                      [-60,60],
                      [0,15],
                      [0,15]])
    limit_range = (limits[:,1] - limits[:,0]).astype(float)
    if help :
        for i in range(18):
            print(names[i]," : ",limits[i])
    else : 
        return ((obs-limits[:,0]) / limit_range)-0.5


def instanciate_agent(args,wandb_run,bootstrap_overwrite):
    
    if bootstrap_overwrite:
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
            args["derivative_indices"] = [3,4,5,9,10,11,14,15]
        else:
            derivative_indices = []
    elif env_name == "hockey-train-shooting":
        # reload(h_env)
        env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)
        if args["use_derivative"]:
            derivative_indices = [3,4,5,9,10,11,14,15]
            args["derivative_indices"] = [3,4,5,9,10,11,14,15]
        else:
            derivative_indices = []
        action_n = 4
    elif env_name == "hockey-train-defense":
        env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE)
        action_n = 4
        if args["use_derivative"]:
            derivative_indices = [3,4,5,9,10,11,14,15]
            args["derivative_indices"] = [3,4,5,9,10,11,14,15]
        else:
            derivative_indices = []
    else:
        env = gym.make(env_name)

    #create save path
    if "savepath" not in args:
        new_path = os.path.join(os.getcwd(),"results")
        print("no save path specified, using default : ",new_path)
        args["savepath"] = new_path
    if "seed" not in args:
        args["seed"] = None
    Path(args["savepath"]).mkdir(parents=True, exist_ok=True)

    if args["algo"] == "ddpg":
        if args["legacy"]:
            action_n = 8
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
                        bootstrap_local=args["bootstrap_local"],
                        )
    elif args["algo"] == "td3":
        sys.path.insert(0,'./TD3')
        from TD3 import TD3Agent
        sys.path.pop(0)
        agent = TD3Agent(env, env_name, action_n, args["seed"], args["savepath"], wandb_run, args)
                        # eps = args["eps"], 
                        # update_target_every = args["update_every"],
                        # # past_states = args["past_states,
                        # derivative = args["use_derivative"],
                        # derivative_indices = derivative_indices,
                        # buffer_size=args["buffer_size"],
                        # discount=args["discount"],
                        # batch_size=args["batch_size"],
                        # learning_rate_actor = args["learning_rate_actor"],
                        # learning_rate_critic=args["learning_rate_critic"],
                        # hidden_sizes_actor=eval(args["hidden_sizes_actor"]),
                        # hidden_sizes_critic=eval(args["hidden_sizes_critic"]),
                        # tau=args["tau"],
                        # policy_noise=args["policy_noise"],
                        # noise_clip=args["noise_clip"],
                        # per=args["per"],
                        # dense_reward=args["dense_reward"],
                        # bootstrap=args["bootstrap"],
                        # HiL=args["hil"],
                        # bc=args["bc"],
                        # bc_lambda=args["bc_lambda"],
                        # cpu=args["cpu"],
                        # replay_ratio=args["replay_ratio"],
                        # batchnorm=args["batchnorm"],
                        # validation_episodes=args["validation_episodes"],
                        # validation_interval=args["validation_interval"],
                        # filled_buffer_ratio=args["filled_buffer_ratio"],
                        # bootstrap_local=args["bootstrap_local"],
                        # )
    elif args["algo"] == "dqn":
        sys.path.insert(0,'./DQN')
        from DQN import DQNAgent
        from DQN import QFunction
        from DQN import DuelingQFunction
        sys.path.pop(0)
        agent = DQNAgent(env, env_name, 12 , args["seed"], args["savepath"], wandb_run, args)
                        # eps = args["eps"], 
                        # update_target_every = args["update_every"],
                        # # past_states = args["past_states,
                        # derivative = args["use_derivative"],
                        # derivative_indices = derivative_indices,
                        # buffer_size=args["buffer_size"],
                        # discount=args["discount"],
                        # batch_size=args["batch_size"],
                        # learning_rate=args["lr"],
                        # hidden_sizes=eval(args["hidden_sizes"]),
                        # hidden_sizes_values=eval(args["hidden_sizes_values"]),
                        # hidden_sizes_advantages=eval(args["hidden_sizes_advantages"]),
                        # tau=args["tau"],
                        # per=args["per"],
                        # dense_reward=args["dense_reward"],
                        # bootstrap=args["bootstrap"],
                        # bc=args["bc"],
                        # bc_lambda=args["bc_lambda"],
                        # cpu=args["cpu"],
                        # replay_ratio=args["replay_ratio"],
                        # dueling=args["dueling"],
                        # double=args["double"],
                        # per_own_impl=args["per_own_impl"],
                        # beta=args["beta"],
                        # alpha=args["alpha"],
                        # alpha_decay=args["alpha_decay"],
                        # beta_growth=args["beta_growth"],
                        # eps_decay=args["eps_decay"],
                        # min_eps=args["min_eps"],
                        # bootstrap_local=args["bootstrap_local"]
                        # )
    return agent, env