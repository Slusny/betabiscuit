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
environments_implemented=['pendulum', 'lunarlander', 'hockey', 'hockey-train-defense', "hockey-train-shooting"]
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

# Training parameters DDPG
training = parser.add_argument_group('training parameters')
training.add_argument('--buffer_size', type=int,
                    default=int(1e6))
training.add_argument('--discount', type=float,
                    default=0.95)
training.add_argument('--batch_size', type=int,
                    default=128)
training.add_argument('--train_iter', type=int,
                    default=32,
                    help='number of training batches per episode')
training.add_argument('-l', '--lr', type=float,
                    default=0.0001,
                    help='learning rate for actor/policy')
training.add_argument('--max_episodes', type=int,
                    dest='max_episodes', default=2000,
                    help='number of episodes')
training.add_argument('--max_timesteps', type=int,
                    dest='max_timesteps', default=2000,
                    help='max timesteps in one episode')
training.add_argument('-u', '--update', type=float,
                    dest='update_every',default=100,
                    help='number of episodes between target network updates')
training.add_argument('-s', '--seed', type=int,
                    default=None,
                    help='random seed')

# Training parameters DDPG
ddpg = parser.add_argument_group('DDPG')
ddpg.add_argument('--learning_rate_actor', type=float,
                    default=0.00001)
ddpg.add_argument('--learning_rate_critic', type=float,
                    default=0.0001)
ddpg.add_argument('-n', '--eps',action='store',  type=float,
                    dest='eps',default=0.1,
                    help='Exploration noise')

# Training parameters TD3
td3 = parser.add_argument_group('TD3')
td3.add_argument('--policy_noise', type=float,
                    default=0.2,
                    help='noise on policy, used for policy smoothing')
td3.add_argument('--noise_clip', type=float,
                    default=0.2,
                    help='noise on policy, used for policy smoothing')
td3.add_argument('--policy_freq', type=float,
                    default=0.2,
                    help='noise on policy, used for policy smoothing')

# Architecture
architecture = parser.add_argument_group('Architecture')
architecture.add_argument('--hidden_sizes_actor', type=str,
                    default='[128,128]')
architecture.add_argument('--hidden_sizes_critic', type=str,
                    default='[128,128,64]')
# architecture.add_argument('--past_states', type=int,
#                     default=1)
architecture.add_argument('--use_derivative',action='store',  type=float,
                    help='calculate the derivative of the state variables. If the velocity is available this will calculate the acceleration')


# Logging
logging = parser.add_argument_group('Logging')
logging.add_argument('--log_interval', type=int,
                    dest='log_interval', default=20,
                    help='print avg reward in the interval')
logging.add_argument('--save_interval', type=int,
                    dest='save_interval', default=500,
                    help='when to save a backup of the model')
logging.add_argument('--savepath', type=str,
                    default='results',
                    help='random seed')
logging.add_argument('--wandb', action='store_true',
                    help='use weights and biases')
logging.add_argument('--notes', type=str, default="",
                    help='any notes to add in logging')
logging.add_argument('--tags', type=str, default="",
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
        env = h_env.HockeyEnv()
        # vx1, vy1, rot1, vx2, vy2, rot2, puck_vx, puck_vy
        derivative_indices = [3,4,5,9,10,11,14,15]
    elif env_name == "hockey-train-shooting":
        # reload(h_env)
        env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)
        derivative_indices = [3,4,5,9,10,11,14,15]
    elif env_name == "hockey-train-defense":
        env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE)
        derivative_indices = [3,4,5,9,10,11,14,15]
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
                        update_target_every = args.update_every,
                        past_states = args.past_states,
                        derivative = args.use_derivative,
                        derivative_indices = derivative_indices)
    
    if args.run:
        print("infer")
    else:
        agent.train(args.train_iter, args.max_episodes, args.max_timesteps, args.log_interval,args.save_interval)