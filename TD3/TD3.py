import torch
import sys
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import optparse
import pickle
import os
import wandb
import memory as mem
from feedforward import Feedforward
sys.path.append('..')
from utility import save_checkpoint
from cpprb import PrioritizedReplayBuffer, ReplayBuffer

import laserhockey.hockey_env as h_env

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class UnsupportedSpace(Exception):
    """Exception raised when the Sensor or Action space are not compatible

    Attributes:
        message -- explanation of the error
    """
    def __init__(self, message="Unsupported Space"):
        self.message = message
        super().__init__(self.message)

# class QFunction():
#     def __init__(self, observation_dim, action_dim, hidden_sizes=[100,100],
#                  learning_rate = 0.0002):
#         self.Q1 = Feedforward(input_size=observation_dim + action_dim, hidden_sizes=hidden_sizes,
#                                 output_size=1,activation_fun=torch.nn.Tanh()).to(device)
        
#         self.Q2 = Feedforward(input_size=observation_dim + action_dim, hidden_sizes=hidden_sizes,
#                                 output_size=1,activation_fun=torch.nn.Tanh()).to(device)

#         self.optimizerQ1=torch.optim.Adam(self.Q1.parameters(),
#                                         lr=learning_rate,
#                                         eps=0.000001)
        
#         self.optimizerQ2=torch.optim.Adam(self.Q2.parameters(),
#                                         lr=learning_rate,
#                                         eps=0.000001)
#         self.loss1 = nn.MSELoss() #torch.nn.SmoothL1Loss()
#         self.loss2 = nn.MSELoss() # not nessessary

#     def fit(self, observations, actions, targets): # all arguments should be torch tensors
#         self.Q1.train() # put model in training mode
#         self.Q2.train()
#         self.optimizerQ1.zero_grad()
#         self.optimizerQ2.zero_grad()
#         # Forward pass

#         # pred1, pred2 = self.Q_value(observations,actions)
#         pred1 = self.Q1.forward(torch.hstack([observations,actions]))
#         pred2 = self.Q2.forward(torch.hstack([observations,actions]))

#         # Optimize both critics -> combined loss
#         lossQ1 = self.loss1(pred1, targets)
#         lossQ2 = self.loss2(pred2, targets)

#         # Backward pass
#         lossQ1.backward()
#         lossQ2.backward()
#         self.optimizerQ1.step()
#         self.optimizerQ2.step()
#         return lossQ1.item()

#     def Q_value(self, observations, actions):
#         # hstack: concatenation along the first axis for 1-D tensors
#         x = torch.hstack([observations,actions])
#         return (self.Q1.forward(x),
#                 self.Q2.forward(x))
    
#     def Q1_value(self, observations, actions):
#         # hstack: concatenation along the first axis for 1-D tensors
#         x = torch.hstack([observations,actions])
#         return (self.Q1.forward(x))
    
    ##################

class QFunction(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, learning_rate, activation_fun=torch.nn.Tanh(), output_activation=None):
        super(QFunction, self).__init__()
        self.input_size = input_size
        self.hidden_sizes  = hidden_sizes
        self.output_size  = 1
        self.output_activation = output_activation
        layer_sizes = [self.input_size] + self.hidden_sizes
        self.layersQ1 = torch.nn.ModuleList([ torch.nn.Linear(i, o) for i,o in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.layersQ2 = torch.nn.ModuleList([ torch.nn.Linear(i, o) for i,o in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.activationsQ1 = [ activation_fun for l in  self.layersQ1 ]
        self.activationsQ2 = [ activation_fun for l in  self.layersQ2 ]
        self.readoutQ1 = torch.nn.Linear(self.hidden_sizes[-1], self.output_size)
        self.readoutQ2 = torch.nn.Linear(self.hidden_sizes[-1], self.output_size)

        self.optimizer=torch.optim.Adam(self.parameters(),
                                        lr=learning_rate,
                                        eps=0.000001)
        
        self.loss = nn.MSELoss() #torch.nn.SmoothL1Loss()

    def forward(self, input):
        # 
        x = input.clone()
        for layer,activation_fun in zip(self.layersQ1, self.activationsQ1):
            x = activation_fun(layer(x))
        if self.output_activation is not None:
            Q1 = self.output_activation(self.readoutQ1(x))
        else:
            Q1 = self.readoutQ1(x)

        # Q2
        x = input.clone()
        for layer,activation_fun in zip(self.layersQ2, self.activationsQ2):
            x = activation_fun(layer(x))
        if self.output_activation is not None:
            Q2 = self.output_activation(self.readoutQ2(x))
        else:
            Q2 = self.readoutQ2(x)

        return Q1, Q2

    def fit(self, observations, actions, targets): # all arguments should be torch tensors
        self.train() # put model in training mode
        self.optimizer.zero_grad()
        # Forward pass
        pred1, pred2 = self.forward(torch.hstack([observations,actions]))

        # Optimize both critics -> combined loss
        loss = self.loss(pred1, targets) + self.loss(pred2, targets)

        # Backward pass
        loss.backward()
        self.optimizer.step()
        return loss.item()

    # def Q_value(self, observations, actions):
    #     # hstack: concatenation along the first axis for 1-D tensors
    #     x = torch.hstack([observations,actions])
    #     return (self.Q1.forward(x),
    #             self.Q2.forward(x))
    
    # Extra function for Q1 value, saves computation on Q2
    def Q1_value(self, observations, actions):
        # hstack: concatenation along the first axis for 1-D tensors
        x = torch.hstack([observations,actions])
        for layer,activation_fun in zip(self.layersQ1, self.activationsQ1):
            x = activation_fun(layer(x))
        if self.output_activation is not None:
            Q1 = self.output_activation(self.readoutQ(x))
        else:
            Q1 = self.readoutQ1(x)
        return Q1
    
####################################

    #     def __init__(self, input_size, hidden_sizes, output_size, activation_fun=torch.nn.Tanh(), output_activation=None):
    #     super(Feedforward, self).__init__()
    #     self.input_size = input_size
    #     self.hidden_sizes  = hidden_sizes
    #     self.output_size  = output_size
    #     self.output_activation = output_activation
    #     layer_sizes = [self.input_size] + self.hidden_sizes
    #     self.layers = torch.nn.ModuleList([ torch.nn.Linear(i, o) for i,o in zip(layer_sizes[:-1], layer_sizes[1:])])
    #     self.activations = [ activation_fun for l in  self.layers ]
    #     self.readout = torch.nn.Linear(self.hidden_sizes[-1], self.output_size)

    # def forward(self, x):
    #     for layer,activation_fun in zip(self.layers, self.activations):
    #         x = activation_fun(layer(x))
    #     if self.output_activation is not None:
    #         return self.output_activation(self.readout(x))
    #     else:
    #         return self.readout(x)

    # def predict(self, x):
    #     with torch.no_grad():
    #         return self.forward(x).cpu().numpy()
        

        #########################

# Orstein-Uhlenbeck noise
class OUNoise():
    def __init__(self, shape, theta: float = 0.15, dt: float = 1e-2):
        self._shape = shape
        self._theta = theta
        self._dt = dt
        self.noise_prev = np.zeros(self._shape)
        self.reset()

    def __call__(self) -> np.ndarray:
        noise = (
            self.noise_prev
            + self._theta * ( - self.noise_prev) * self._dt
            + np.sqrt(self._dt) * np.random.normal(size=self._shape)
        )
        self.noise_prev = noise
        return noise

    def reset(self) -> None:
        self.noise_prev = np.zeros(self._shape)

class TD3Agent(object):
    """
    Agent implementing Q-learning with NN function approximation.
    """
    def __init__(self, env, env_name, action_n, seed, savepath, wandb_run, bootstrap, **userconfig):

        observation_space = env.observation_space
        action_space = env.action_space
        if not isinstance(observation_space, spaces.box.Box):
            raise UnsupportedSpace('Observation space {} incompatible ' \
                                   'with {}. (Require: Box)'.format(observation_space, self))
        if not isinstance(action_space, spaces.box.Box):
            raise UnsupportedSpace('Action space {} incompatible with {}.' \
                                   ' (Require Box)'.format(action_space, self))

        # Seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.savepath = savepath
        self.seed = seed
        self.env_name = env_name
        self.env = env
        self._observation_space = observation_space
        self._obs_dim=self._observation_space.shape[0]
        self._action_space = action_space
        self._action_n = action_n
        self._config = {
            "eps": 0.1,            # Epsilon: noise strength to add to policy
            "discount": 0.95,
            "buffer_size": int(1e6),
            "batch_size": 128,
            "learning_rate_actor": 0.00001,
            "learning_rate_critic": 0.0001,
            "hidden_sizes_actor": [128,128],
            "hidden_sizes_critic": [128,128,64],
            "update_target_every": 2,
            "use_target_net": True,
            # "past_states": 1,
            "derivative": False,
            "derivative_indices": [],
            "bootstrap": None,
            "tau": 0.005,
            "policy_noise": 0.4, 
            "noise_clip": 0.5,
            "per": False,

        }
        self._config.update(userconfig)
        self._eps = self._config['eps']
        print("Config: ", self._config)

        self.action_noise = OUNoise((self._action_n))

        if self._config["per"]:
            self.buffer = PrioritizedReplayBuffer(self._config["buffer_size"], {
                "obs": {"shape": (self._obs_dim+len(self._config["derivative_indices"]))},
                "act": {"shape": (self._action_n)},
                "rew": {},
                "next_obs": {"shape": (self._obs_dim+len(self._config["derivative_indices"]))},
                "done": {}
                }
            )
        else:
            self.buffer = mem.Memory(max_size=self._config["buffer_size"])



        # Q Network
        self.Q = QFunction(input_size=self._obs_dim+self._action_n+len(self._config["derivative_indices"]),#self._obs_dim*self._config["past_states"],
                           hidden_sizes= self._config["hidden_sizes_critic"],
                           learning_rate = self._config["learning_rate_critic"]).to(device)
        # target Q Network
        self.Q_target = QFunction(input_size=self._obs_dim+self._action_n+len(self._config["derivative_indices"]),#self._obs_dim*self._config["past_states"],
                                  hidden_sizes= self._config["hidden_sizes_critic"],
                                  learning_rate = 0).to(device)

        self.policy = Feedforward(input_size=self._obs_dim+len(self._config["derivative_indices"]),#self._obs_dim*self._config["past_states"],
                                  hidden_sizes= self._config["hidden_sizes_actor"],
                                  output_size=self._action_n,
                                  activation_fun = torch.nn.ReLU(),
                                  output_activation = torch.nn.Tanh()).to(device)
        self.policy_target = Feedforward(input_size=self._obs_dim+len(self._config["derivative_indices"]),#self._obs_dim*self._config["past_states"],
                                         hidden_sizes= self._config["hidden_sizes_actor"],
                                         output_size=self._action_n,
                                         activation_fun = torch.nn.ReLU(),
                                         output_activation = torch.nn.Tanh()).to(device)
        
        # To resume training from a saved model.
        # Models get saved in weights-and-biases and loaded from there.
        if(self._config["bootstrap"] is not None):
            api = wandb.Api()
            art = api.artifact(bootstrap, type='model')
            state = torch.load(art.file())
            self.restore_state(state)
        
        # copy initialized weights
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.policy_target.load_state_dict(self.policy.state_dict())

        self.optimizer=torch.optim.Adam(self.policy.parameters(),
                                        lr=self._config["learning_rate_actor"],
                                        eps=0.000001)
        self.train_iter = 0

        # log gradients to W&B
        self.wandb_run = wandb_run
        if(wandb_run):
            wandb.watch(self.Q, log_freq=100)
            wandb.watch(self.policy, log_freq=100)


    def _copy_nets(self):
        # Full copy
        if self._config["tau"] == 1:
            # self.Q_target.Q1.load_state_dict(self.Q.Q1.state_dict())
            # self.Q_target.Q2.load_state_dict(self.Q.Q2.state_dict())
            self.Q_target.load_state_dict(self.Q.state_dict())
            self.policy_target.load_state_dict(self.policy.state_dict())
        
        # Convex update
        else:
            for param, target_param in zip(self.policy.parameters(), self.policy_target.parameters()): # AAAAAAAAAAAAAAAAAAAAAAAAAAAAH
                target_param.data.copy_(self._config["tau"] * param.data + (1 - self._config["tau"]) * target_param.data)

            # for param, target_param in zip(self.Q.Q1.parameters(), self.Q_target.Q1.parameters()):
            #     target_param.data.copy_(self._config["tau"] * param.data + (1 - self._config["tau"]) * target_param.data)

            # for param, target_param in zip(self.Q.Q2.parameters(), self.Q_target.Q2.parameters()):
            #     target_param.data.copy_(self._config["tau"] * param.data + (1 - self._config["tau"]) * target_param.data)
            for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
                target_param.data.copy_(self._config["tau"] * param.data + (1 - self._config["tau"]) * target_param.data)

    def act(self, observation, eps=None):
        if eps is None:
            eps = self._eps
        #
        observation = torch.from_numpy(observation.astype(np.float32)).to(device) 
        action = (self.policy.predict(observation) + eps*self.action_noise()).clip(-1,1)  # action in -1 to 1 (+ noise)
        # action = self.policy.predict(observation) + eps*self.action_noise()  # action in -1 to 1 (+ noise)
        # action = self._action_space.low + (action + 1.0) / 2.0 * (self._action_space.high - self._action_space.low)
        return action

    def store_transition(self, transition):
        transition = transition
        if self._config["per"]:
            transition["error"] = 0
            self.buffer.add(obs=transition[0], act=transition[1], rew=transition[2], next_obs=transition[3], done=transition[4])
        else:
            self.buffer.add_transition(transition)

    def state(self):
        # return (self.Q.Q1.state_dict(),self.Q.Q2.state_dict(), self.policy.state_dict())
        return (self.Q.state_dict(), self.policy.state_dict())

    def restore_state(self, state):
        # self.Q.Q1.load_state_dict(state[0],strict=False)
        # self.Q.Q2.load_state_dict(state[1],strict=False)
        self.Q.load_state_dict(state[0])
        self.policy.load_state_dict(state[1])
        self._copy_nets()

    def reset(self):
        self.action_noise.reset()
    
    def sample_replaybuffer(self):
        to_torch = lambda x: torch.from_numpy(x.astype(np.float32)).to(device)
        if self._config["per"]:
            self.replay_buffer.sample(self._config['batch_size'])
            s, a, r = data['obs'], data['act'], data['rew']
            #actions_h, rewards = data['intervene'], data['acte']
            s_prime, done, idxs = data['next_obs'], data['done'], data['indexes']
        else:
            data=self.buffer.sample(batch=self._config['batch_size'])
            s = to_torch(np.stack(data[:,0])) # s_t
            a = to_torch(np.stack(data[:,1])) # a_t
            rew = to_torch(np.stack(data[:,2])[:,None]) # rew  (batchsize,1)
            s_prime = to_torch(np.stack(data[:,3])) # s_t+1
            done = to_torch(np.stack(data[:,4])[:,None]) # done signal  (batchsize,1)
            idxs = None
        return s,a,rew,s_prime,done,idxs

    # inner training loop where we fit the Actor and Critic
    def train_innerloop(self, iter_fit=32):
        losses = []
        actor_loss = 0
        
        for i in range(iter_fit):

            # sample from the replay buffer
            s,a,rew,s_prime,done,idxs = self.sample_replaybuffer()
            # Target Policy Smoothing (TD3 paper 5.3)
            # adding noise to the target action smooths the value estimate
            policy_noise = (torch.randn_like(a) * self._config["policy_noise"]).clamp(-self._config["noise_clip"], self._config["noise_clip"])
            next_action = (self.policy_target.forward(s_prime) + policy_noise).clamp(-1,1)
            # next_action = (self.policy_target.forward(s_prime) + policy_noise).clamp(self._action_space.low,self._action_space.high)

            # Clipped Double Q-Learning (TD3 paper 4.2)
            # To combat overestimation bias, we use the minimum of the two Q functions
            q_prime_1, q_prime_2 = self.Q_target(torch.hstack([s_prime,next_action]))
            q_prime = torch.min(q_prime_1, q_prime_2)
            
            # target
            gamma=self._config['discount']
            td_target = rew + gamma * (1.0-done) * q_prime

            # prediction for priotized replay buffer
            if self._config["per"]:
                pred1 = self.Q.Q1_value(s,a)
                priorities = abs(td_target - pred1).detach().cpu().numpy()
                self.replay_buffer.update_priorities(idxs, priorities) 

            # optimize the Q objective ( Critic )
            fit_loss = self.Q.fit(s, a, td_target)

            # Delay Polciy Updates (TD3 paper 5.2)
            if self.train_iter % self._config["update_target_every"] == 0:
                self.optimizer.zero_grad()
                q = self.Q.Q1_value(s, self.policy.forward(s))
                actor_loss = -torch.mean(q)
                actor_loss.backward()
                self.optimizer.step()

                self._copy_nets() # with a update_frequency of 2 and a tau of 0.005 a "full" update is done every 400 steps

            self.train_iter+=1
            losses.append((fit_loss, actor_loss.item()))

        return losses

    # Outer loop where the replay buffer gets filled
    def train(self, iter_fit, max_episodes, max_timesteps,log_interval,save_interval):
        to_torch = lambda x: torch.from_numpy(x.astype(np.float32)).to(device)
         # logging variables
        rewards = []
        lengths = []
        losses = []
        timestep = 0
        lr = self._config['learning_rate_actor']

        # def rollrep(arr,arr2):
        #     # roll and replace: roll the array and replace the first row with arr2
        #     arr = np.roll(arr,axis=0, shift=1)
        #     arr[0,:] = arr2
        #     return arr 

        def add_derivative(obs,pastobs):
            # filter_index = [3,4,5,9,10,11,14,15]
            return np.append(obs,(obs-pastobs)[self._config["derivative_indices"]])
        
        if (self.env_name == "hockey"):
            self.player = h_env.BasicOpponent(weak=False)

        def opponent_action(obs):
            if (self.env_name == "hockey"):
                return self.player.act(obs)
            else:
                return np.array([0,0.,0,0])

        # training loop
        for i_episode in range(1, max_episodes+1):
            ob, _info = self.env.reset()
            # Incorporate  Acceleration
            past_obs = ob.copy()
            # if self._config["acceleration"]: past_obs.append([0]*8)

            # Old way of adding old frames
            # a2 = opponent_action(ob)
            # done = False; trunc = False;
            # past_obs = np.tile(ob,(self._config["past_states"],1)) # past_obs is a stack of past observations of shape (past_states, obs_dim)
            # for past in range(self._config["past_states"]-1):
            #     a = self.act(past_obs.flatten())
            #     (ob_past, reward, done, trunc, _info) = self.env.step(np.hstack([a,a2]))
            #     past_obs = rollrep(past_obs,ob_past)
            #     if done or trunc: break
            self.reset()
            total_reward=0
            for t in range(max_timesteps):
                # if done or trunc: break
                timestep += 1
                if self._config["derivative"]:  a = self.act(add_derivative(ob,past_obs))
                else :                          a = self.act(ob)
                a2 = opponent_action(ob)
                # a = self.act(past_obs.flatten())
                # a2 = opponent_action(past_obs[-1])

                (ob_new, reward, done, trunc, _info) = self.env.step(np.hstack([a,a2]))
                total_reward+= reward
                
                self.store_transition((add_derivative(ob,past_obs), a, reward, add_derivative(ob_new,ob), done))
                past_obs = ob
                ob=ob_new

                # self.store_transition((past_obs.flatten(), a, reward, rollrep(past_obs,ob_new).flatten(), done))
                # past_obs = rollrep(past_obs,ob_new)
                if done or trunc: break

            l = self.train_innerloop(iter_fit)
            losses.extend(l)

            rewards.append(total_reward)
            lengths.append(t)
            if self.wandb_run : 
                loss_mean_innerloop = np.array(l).mean(axis=0)
                wandb.log({"actor_loss": loss_mean_innerloop[1] , "critic_loss": loss_mean_innerloop[0] , "reward": total_reward, "length":t })

            # save every 500 episodes
            if i_episode % save_interval == 0:
                save_checkpoint(self.state(),self.savepath,"TD3",self.env_name, i_episode, self.wandb_run, self._eps, lr, self.seed,rewards,lengths, losses)

            # logging
            if i_episode % log_interval == 0:
                avg_reward = np.mean(rewards[-log_interval:])
                avg_length = int(np.mean(lengths[-log_interval:]))
                print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, avg_reward))

        if i_episode % 500 != 0: save_checkpoint(self.state(),self.savepath,"TD3",self.env_name, i_episode, self.wandb_run, self._eps, lr, self.seed,rewards,lengths, losses)
            
        return losses


def main():
    optParser = optparse.OptionParser()
    optParser.add_option('-e', '--env',action='store', type='string',
                         dest='env_name',default="Pendulum-v1",
                         help='Environment (default %default)')
    optParser.add_option('-n', '--eps',action='store',  type='float',
                         dest='eps',default=0.1,
                         help='Policy noise (default %default)')
    optParser.add_option('-t', '--train',action='store',  type='int',
                         dest='train',default=32,
                         help='number of training batches per episode (default %default)')
    optParser.add_option('-l', '--lr',action='store',  type='float',
                         dest='lr',default=0.0001,
                         help='learning rate for actor/policy (default %default)')
    optParser.add_option('-m', '--maxepisodes',action='store',  type='int',
                         dest='max_episodes',default=2000,
                         help='number of episodes (default %default)')
    optParser.add_option('-u', '--update',action='store',  type='float',
                         dest='update_every',default=100,
                         help='number of episodes between target network updates (default %default)')
    optParser.add_option('-s', '--seed',action='store',  type='int',
                         dest='seed',default=None,
                         help='random seed (default %default)')
    opts, args = optParser.parse_args()
    ############## Hyperparameters ##############
    env_name = opts.env_name
    # creating environment
    # if env_name == "LunarLander-v2":
    #     env = gym.make(env_name, continuous = True)
    # else:
    #     env = gym.make(env_name)

    env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE)

    render = False
    log_interval = 20           # print avg reward in the interval
    max_episodes = opts.max_episodes # max training episodes
    max_timesteps = 2000         # max timesteps in one episode

    train_iter = opts.train      # update networks for given batched after every episode
    eps = opts.eps               # noise of DDPG policy
    lr  = opts.lr                # learning rate of DDPG policy

    #############################################

    random_seed = opts.seed
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    TD3 = TD3Agent(env, env_name, opts.seed, "test", False, eps = eps, learning_rate_actor = lr,
                     update_target_every = opts.update_every)

    # logging variables
    rewards = []
    lengths = []
    losses = []
    timestep = 0

    def save_statistics():
        with open(f"./results/TD3_{env_name}-eps{eps}-t{train_iter}-l{lr}-s{random_seed}-stat.pkl", 'wb') as f:
            pickle.dump({"rewards" : rewards, "lengths": lengths, "eps": eps, "train": train_iter,
                         "lr": lr, "update_every": opts.update_every, "losses": losses}, f)


    testing_loss = TD3.train(train_iter,max_episodes, max_timesteps, log_interval)
    print(testing_loss)



if __name__ == '__main__':
    main()
