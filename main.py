import argparse
import gymnasium as gym
import sys
from pathlib import Path
import laserhockey.hockey_env as h_env
sys.path.insert(0,'./DDPG')
from DDPG import DDPGAgent
from importlib import reload
import wandb

# Available arguments for program
environments_implemented=['pendulum', 'lunarlander', 'hockey', 'hockey-train-defence', "hockey-train-shooting"]
algorithms_implemented = ['ddpg']

# Loggin
log_interval = 20           # print avg reward in the interval
max_timesteps = 2000         # max timesteps in one episode


# Argument Parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', type=str, required=True,
                    dest='env_name', choices=environments_implemented,
                    help="specify environment, choose one of: "+str(environments_implemented))
parser.add_argument('--algo', type=str, required=True,
                    choices=algorithms_implemented,
                    help="specify algorithm, choose one of: " + str(algorithms_implemented))
parser.add_argument('-r', '--run', action='store_true',
                    help='Do you wish to run/infer and not train')

# Training parameters
parser.add_argument('--train_iter', type=int,
                    default=32,
                    help='number of training batches per episode')
parser.add_argument('-l', '--lr', type=float,
                    default=0.0001,
                    help='learning rate for actor/policy')
parser.add_argument('-n', '--eps',action='store',  type=float,
                    dest='eps',default=0.1,
                    help='Policy noise')
parser.add_argument('--maxepisodes', type=int,
                    dest='max_episodes', default=2000,
                    help='number of episodes')
parser.add_argument('--maxtimesteps', type=int,
                    dest='max_timesteps', default=2000,
                    help='max timesteps in one episode')
parser.add_argument('-u', '--update', type=float,
                    dest='update_every',default=100,
                    help='number of episodes between target network updates')
parser.add_argument('-s', '--seed', type=int,
                    default=None,
                    help='random seed')
parser.add_argument('--savepath', type=str,
                    default='results',
                    help='random seed')

# Architecture
parser.add_argument('--hidden_sizes_actor', type=str,
                    default='[128,128]')
parser.add_argument('--hidden_sizes_critic', type=str,
                    default='[128,128,64]')
parser.add_argument('--buffer_size', type=int,
                    default=int(1e6))
parser.add_argument('--discount', type=float,
                    default=0.95)
parser.add_argument('--batch_size', type=int,
                    default=128)
parser.add_argument('--learning_rate_actor', type=float,
                    default=0.00001)
parser.add_argument('--learning_rate_critic', type=float,
                    default=0.0001)


# Logging
parser.add_argument('--loginterval', type=int,
                    dest='log_interval', default=20,
                    help='print avg reward in the interval')
parser.add_argument('--save_interval', type=int,
                    dest='save_interval', default=500,
                    help='when to save a backup of the model')
parser.add_argument('--wandb', action='store_true',
                    help='use weights and biases')
parser.add_argument('--notes', type=str, default="",
                    help='any notes to add in logging')
parser.add_argument('--tags', type=str, default="",
                    help='any tags to add in logging')
args = parser.parse_args()

if __name__ == "__main__":

    # creating environment
    env_name = args.env_name
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

    #weights and biases
    if args.wandb   : 
        config_wandb = vars(args).copy()
        for key in ['notes','tags','wandb']:del config_wandb[key]
        del config_wandb
        wandb_run = wandb.init(project=env_name + " - " +args.algo, 
                               config=args,
                               notes=args.notes,
                               tags=args.tags)
    else            : wandb_run = None

    #create save path
    Path(args.savepath).mkdir(parents=True, exist_ok=True)

    if args.algo == "ddpg":
        agent = DDPGAgent(env, env_name, args.seed, args.savepath, wandb_run,
                        eps = args.eps, 
                        learning_rate_actor = args.lr,
                        update_target_every = args.update_every)
    
    if args.run:
        print("infer")
    else:
        agent.train(args.train_iter, args.max_episodes, args.max_timesteps, args.log_interval,args.save_interval)