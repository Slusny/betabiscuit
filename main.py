import argparse
import gymnasium as gym
import sys
from pathlib import Path
import laserhockey.hockey_env as h_env
sys.path.insert(0,'./DDPG')
sys.path.insert(0,'./TD3')
sys.path.insert(0,'./DQN')
from DDPG import DDPGAgent
from TD3 import TD3Agent
from DQN import DQNAgent
from importlib import reload
import wandb

# Available arguments for program
environments_implemented=['pendulum', 'lunarlander', 'hockey', 'hockey-train-defense', "hockey-train-shooting"]
algorithms_implemented = ['ddpg','td3', 'dqn']

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
parser.add_argument('--cpu', action='store_true',
                    help='Force to run on cpu')

# Training parameters
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
                    dest='max_timesteps', default=200,
                    help='max timesteps in one episode')
training.add_argument('-u', '--update', type=float,
                    dest='update_every',default=100,
                    help='number of episodes between target network updates. Default 100 for DDPG, 2 for TD3')
training.add_argument('--dense_reward',  action='store_true',
                    help='using a dense reward, composed out of closeness to puck and puck direction')
training.add_argument('-s', '--seed', type=int,
                    default=None,
                    help='random seed')
training.add_argument('--replay_ratio', type=float,
                    default=0.,
                    help='how many gradient updates should be done per replay buffer update. Replaces train_iter')
training.add_argument('-n', '--eps',action='store',  type=float,
                    dest='eps',default=0.1,
                    help='Exploration noise')
training.add_argument( '--eps_decay',action='store',  type=float,
                    default=0.9999, help='Exploration decay')
training.add_argument( '--min_eps',action='store',  type=float,
                    default=0.01, help='minimum exploration noise to decay to')



# Training parameters DQN
dqn = parser.add_argument_group('DQN')
dqn.add_argument('--double', action='store_true',help='use double dqn')
dqn.add_argument('--dueling', action='store_true',help='use dueling dqn')
dqn.add_argument('--per_own_impl', action='store_true',help='use own implementation of prioritized experience replay')
dqn.add_argument('--beta', type=float,
                    default=0.4, help='beta for prioritized experience replay')
dqn.add_argument('--beta_growth', type=float,
                    default=1.0001, help='beta_growth for prioritized experience replay')
dqn.add_argument('--alpha', type=float,
                    default=0.6, help='alpha for prioritized experience replay')
dqn.add_argument('--alpha_decay', type=float,
                    default=1., help='alpha_decay for prioritized experience replay')

# Training parameters DDPG
ddpg = parser.add_argument_group('DDPG')
ddpg.add_argument('--learning_rate_actor', type=float,
                    default=0.0001)
ddpg.add_argument('--learning_rate_critic', type=float,
                    default=0.0001)

# Training parameters TD3
td3 = parser.add_argument_group('TD3')
td3.add_argument('--policy_noise', type=float,
                    default=0.1,                                            # in TD3 paper 0.4
                    help='noise on policy, used for policy smoothing')
td3.add_argument('--noise_clip', type=float,
                    default=0.3,                                            # in TD3 paper 0.5
                    help='range to clip the policy noise')
td3.add_argument('--policy_freq', type=float,
                    default=0.2,
                    help='noise on policy, used for policy smoothing')
td3.add_argument('--tau', type=float,
                    default=0.005,
                    help='weight for convex update of target weights')

# Architecture
architecture = parser.add_argument_group('Architecture')
architecture.add_argument('--hidden_sizes_actor', type=str,
                    default='[128,128]')
architecture.add_argument('--hidden_sizes_critic', type=str,
                    default='[128,128,64]')
# architecture.add_argument('--past_states', type=int,
#                     default=1)
architecture.add_argument('--use_derivative',action='store_true', 
                    help='calculate the derivative of the state variables. If the velocity is available this will calculate the acceleration')
architecture.add_argument('--per', action='store_true',help='use prioritized experience replay')
architecture.add_argument('--bootstrap',action='store', type=str, default=None,
                    help='wandb path ("betabiscuit/project/artifact") to model artifacts')
architecture.add_argument('--hil', action='store_true',help='human in the loop training')
architecture.add_argument('--bc', action='store_true',help='use behavior cloning')
architecture.add_argument('--bc_lambda', action='store', type=float, 
                          default=2.0 ,help='hyperparameter for behavior cloning loss')
                    
architecture.add_argument('--legacy', action='store_true',help='use outdated architecture')


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
logging.add_argument('--wandb_resume', action='store', default=None,
                    type=str, help='use weights and biases')
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
        action_n = env.action_space.shape[0]
    elif env_name == "pendulum":
        env = gym.make("Pendulum-v1")
        action_n = env.action_space.shape[0]
    elif env_name == "hockey":
        env = h_env.HockeyEnv()
        action_n = 4
        # vx1, vy1, rot1, vx2, vy2, rot2, puck_vx, puck_vy
        if args.use_derivative:
            derivative_indices = [3,4,5,9,10,11,14,15]
        else:
            derivative_indices = []
    elif env_name == "hockey-train-shooting":
        # reload(h_env)
        env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)
        if args.use_derivative:
            derivative_indices = [3,4,5,9,10,11,14,15]
        else:
            derivative_indices = []
        action_n = 4
    elif env_name == "hockey-train-defense":
        env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE)
        action_n = 4
        if args.use_derivative:
            derivative_indices = [3,4,5,9,10,11,14,15]
        else:
            derivative_indices = []
    else:
        env = gym.make(env_name)

    #weights and biases
    if args.wandb   : 
        config_wandb = vars(args).copy()
        for key in ['notes','tags','wandb']:del config_wandb[key]
        del config_wandb
        if args.wandb_resume is not None :
            wandb_run = wandb.init(project=env_name + " - " +args.algo, 
                config=args,
                notes=args.notes,
                tags=args.tags,
                resume="must",
                id=args.wandb_resume)
        else:
            wandb_run = wandb.init(project=env_name + " - " +args.algo, 
                config=args,
                notes=args.notes,
                tags=args.tags)
    else : wandb_run = None

    #create save path
    Path(args.savepath).mkdir(parents=True, exist_ok=True)

    if args.algo == "ddpg":
        agent = DDPGAgent(env, env_name, action_n, args.seed, args.savepath, wandb_run,
                        eps = args.eps, 
                        learning_rate_actor = args.lr,
                        update_target_every = args.update_every,
                        # past_states = args.past_states,
                        derivative = args.use_derivative,
                        derivative_indices = derivative_indices,
                        buffer_size=args.buffer_size,
                        discount=args.discount,
                        batch_size=args.batch_size,
                        learning_rate_critics=args.learning_rate_critic,
                        hidden_sizes_actor=eval(args.hidden_sizes_actor),
                        hidden_sizes_critic=eval(args.hidden_sizes_critic),
                        per=args.per,
                        dense_reward=args.dense_reward,
                        bootstrap=args.bootstrap,
                        legacy=args.legacy,
                        bc=args.bc,
                        bc_lambda=args.bc_lambda,
                        cpu=args.cpu,
                        replay_ratio=args.replay_ratio,
                        )
    elif args.algo == "td3":
        agent = TD3Agent(env, env_name, action_n, args.seed, args.savepath, wandb_run,
                        eps = args.eps, 
                        learning_rate_actor = args.learning_rate_actor,
                        update_target_every = args.update_every,
                        # past_states = args.past_states,
                        derivative = args.use_derivative,
                        derivative_indices = derivative_indices,
                        buffer_size=args.buffer_size,
                        discount=args.discount,
                        batch_size=args.batch_size,
                        learning_rate_critic=args.learning_rate_critic,
                        hidden_sizes_actor=eval(args.hidden_sizes_actor),
                        hidden_sizes_critic=eval(args.hidden_sizes_critic),
                        tau=args.tau,
                        policy_noise=args.policy_noise,
                        noise_clip=args.noise_clip,
                        per=args.per,
                        dense_reward=args.dense_reward,
                        bootstrap=args.bootstrap,
                        HiL=args.hil,
                        bc=args.bc,
                        bc_lambda=args.bc_lambda,
                        cpu=args.cpu,
                        replay_ratio=args.replay_ratio
                        )
    elif args.algo == "dqn":
        agent = DQNAgent(env, env_name, 8 , args.seed, args.savepath, wandb_run,
                        eps = args.eps, 
                        update_target_every = args.update_every,
                        # past_states = args.past_states,
                        derivative = args.use_derivative,
                        derivative_indices = derivative_indices,
                        buffer_size=args.buffer_size,
                        discount=args.discount,
                        batch_size=args.batch_size,
                        learning_rate=args.lr,
                        hidden_sizes_actor=eval(args.hidden_sizes_actor),
                        hidden_sizes_critic=eval(args.hidden_sizes_critic),
                        tau=args.tau,
                        per=args.per,
                        dense_reward=args.dense_reward,
                        bootstrap=args.bootstrap,
                        bc=args.bc,
                        bc_lambda=args.bc_lambda,
                        cpu=args.cpu,
                        replay_ratio=args.replay_ratio,
                        dueling=args.dueling,
                        double=args.double,
                        per_own_impl=args.per_own_impl,
                        beta=args.beta,
                        alpha=args.alpha,
                        alpha_decay=args.alpha_decay,
                        beta_growth=args.beta_growth,
                        eps_decay=args.eps_decay,
                        min_eps=args.min_eps,

                        )
    
    if args.run:
        print("infer")
    else:
        agent.train(args.train_iter, args.max_episodes, args.max_timesteps, args.log_interval,args.save_interval)