import torch
import collections
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
import memory_DDPG as mem
from feedforward_DDPG import Feedforward
sys.path.append('..')
from utility import save_checkpoint
from cpprb import PrioritizedReplayBuffer, ReplayBuffer

import laserhockey.hockey_env as h_env


class UnsupportedSpace(Exception):
    """Exception raised when the Sensor or Action space are not compatible

    Attributes:
        message -- explanation of the error
    """
    def __init__(self, message="Unsupported Space"):
        self.message = message
        super().__init__(self.message)

class QFunction(Feedforward):
    def __init__(self, observation_dim, action_dim, hidden_sizes=[100,100],
                 learning_rate = 0.0002):
        super().__init__(input_size=observation_dim + action_dim, hidden_sizes=hidden_sizes,
                         output_size=1)
        self.optimizer=torch.optim.Adam(self.parameters(),
                                        lr=learning_rate,
                                        eps=0.000001)
        self.loss = torch.nn.SmoothL1Loss()

    def fit(self, observations, actions, targets): # all arguments should be torch tensors
        self.train() # put model in training mode
        self.optimizer.zero_grad()
        # Forward pass

        pred = self.Q_value(observations,actions)
        # Compute Loss
        loss = self.loss(pred, targets)

        # Backward pass
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def Q_value(self, observations, actions):
        # hstack: concatenation along the first axis for 1-D tensors
        return self.forward(torch.hstack([observations,actions]))

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

class DDPGAgent(object):
    """
    Agent implementing Q-learning with NN function approximation.
    """
    def __init__(self, env, env_name, action_n, seed, savepath, wandb_run, **userconfig):

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
            "update_target_every": 100,
            "use_target_net": True,
            # "past_states": 1,
            "derivative": False,
            "derivative_indices": [],
            "bootstrap": None,
            "per": False,
            "dense_reward": False,
            "legacy": False,
            "bc": False,
            "bc_lambda":2.0,
            "cpu": False,
            "replay_ratio": 0,
            "bootstrap_local":False,

        }
        self._config.update(userconfig)
        self._eps = self._config['eps']
        print("Config: ", self._config)
        if self._config["legacy"]:
            self._action_n = 8

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

        if self._config["cpu"]:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Q Network
        self.Q = QFunction(observation_dim=self._obs_dim+len(self._config["derivative_indices"]),#self._obs_dim*self._config["past_states"],
                           action_dim=self._action_n,
                           hidden_sizes= self._config["hidden_sizes_critic"],
                           learning_rate = self._config["learning_rate_critic"]).to(self.device)
        # target Q Network
        self.Q_target = QFunction(observation_dim=self._obs_dim+len(self._config["derivative_indices"]),#self._obs_dim*self._config["past_states"],
                                  action_dim=self._action_n,
                                  hidden_sizes= self._config["hidden_sizes_critic"],
                                  learning_rate = 0).to(self.device)

        self.policy = Feedforward(input_size=self._obs_dim+len(self._config["derivative_indices"]),#self._obs_dim*self._config["past_states"],
                                  hidden_sizes= self._config["hidden_sizes_actor"],
                                  output_size=self._action_n,
                                  activation_fun = torch.nn.ReLU(),
                                  output_activation = torch.nn.Tanh()).to(self.device)
        self.policy_target = Feedforward(input_size=self._obs_dim+len(self._config["derivative_indices"]),#self._obs_dim*self._config["past_states"],
                                         hidden_sizes= self._config["hidden_sizes_actor"],
                                         output_size=self._action_n,
                                         activation_fun = torch.nn.ReLU(),
                                         output_activation = torch.nn.Tanh()).to(self.device)

        # To resume training from a saved model.
        # Models get saved in weights-and-biases and loaded from there.
        if(self._config["bootstrap"] is not None):
            if self._config["bootstrap_local"]:
                state = torch.load(str(self._config["bootstrap"]))
                if isinstance(state[0], collections.OrderedDict):
                    self.restore_state(state)
                else:
                    self.Q = state
            else:
                api = wandb.Api()
                art = api.artifact(self._config["bootstrap"], type='model')
                if self._config["cpu"]:
                    state = torch.load(art.file(),map_location='cpu')
                else:
                    state = torch.load(art.file())
                self.restore_state(state)

        if self._config["bc"]:
            self.teacher = h_env.BasicOpponent(weak=False)
        
        self._copy_nets()

        self.optimizer=torch.optim.Adam(self.policy.parameters(),
                                        lr=self._config["learning_rate_actor"],
                                        eps=0.000001)
        self.train_iter = 0

        # log gradients to W&B
        self.wandb_run = wandb_run
        # if(wandb_run): # log gradients to W&B
        #     wandb.watch(self.Q, log_freq=100)
        #     wandb.watch(self.policy, log_freq=100)


    def _copy_nets(self):
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.policy_target.load_state_dict(self.policy.state_dict())

    def act(self, observation, eps=None):
        if eps is None:
            eps = self._eps
        #
        observation = torch.from_numpy(observation.astype(np.float32)).to(self.device) 
        action = self.policy.predict(observation) + eps*self.action_noise()  # action in -1 to 1 (+ noise)
        action = self._action_space.low + (action + 1.0) / 2.0 * (self._action_space.high - self._action_space.low)
        return action

    def store_transition(self, transition):
        transition = transition
        if self._config["per"]:
            self.buffer.add(obs=transition[0], act=transition[1], rew=transition[2], next_obs=transition[3], done=transition[4])
        else:
            self.buffer.add_transition(transition)

    def state(self):
        return (self.Q.state_dict(), self.policy.state_dict())

    def restore_state(self, state):
        self.Q.load_state_dict(state[0],strict=False)
        self.policy.load_state_dict(state[1],strict=False)
        self._copy_nets()

    def reset(self):
        self.action_noise.reset()

    def sample_replaybuffer(self):
        to_torch = lambda x: torch.from_numpy(x.astype(np.float32)).to(self.device)
        if self._config["per"]:
            data = self.buffer.sample(self._config['batch_size'])
            s, a, rew = data['obs'], data['act'], data['rew']
            #actions_h, rewards = data['intervene'], data['acte']
            s_prime, done, idxs = data['next_obs'], data['done'], data['indexes']
        else:
            data=self.buffer.sample(batch=self._config['batch_size'])
            s = np.stack(data[:,0]) # s_t
            a = np.stack(data[:,1]) # a_t
            rew = np.stack(data[:,2])[:,None] # rew  (batchsize,1)
            s_prime = np.stack(data[:,3]) # s_t+1
            done = np.stack(data[:,4])[:,None] # done signal  (batchsize,1)
            idxs = None
        return to_torch(s),to_torch(a),to_torch(rew),to_torch(s_prime),to_torch(done),idxs

    # inner training loop where we fit the Actor and Critic
    def train_innerloop(self, iter_fit=32):
        to_torch = lambda x: torch.from_numpy(x.astype(np.float32)).to(self.device)
        losses = []
        actor_loss_value = []
        q_loss_value = []
        bc_loss_value = []
        self.train_iter+=1
        if self._config["use_target_net"] and self.train_iter % self._config["update_target_every"] == 0:  # using 100 update interval this means 3200 updates to Q networks
            self._copy_nets()
        for i in range(iter_fit):

            # sample from the replay buffer
            s,a,rew,s_prime,done,idxs = self.sample_replaybuffer()

            if self._config["use_target_net"]:
                q_prime = self.Q_target.Q_value(s_prime, self.policy_target.forward(s_prime))
            else:
                q_prime = self.Q.Q_value(s_prime, self.policy.forward(s_prime))
            # target
            gamma=self._config['discount']
            td_target = rew + gamma * (1.0-done) * q_prime

            # prediction for priotized replay buffer
            if self._config["per"]:
                q = self.Q.Q_value(s, a)
                td_error = abs(td_target - q).detach().cpu().numpy()
                self.buffer.update_priorities(idxs, td_error)

            # optimize the Q objective
            fit_loss = self.Q.fit(s, a, td_target)

            # optimize actor objective
            self.optimizer.zero_grad()
            a_policy = self.policy.forward(s)
            q = self.Q.Q_value(s,a_policy)
            if self._config["bc"]:
                alpha = self._config["bc_lambda"]
                a_teacher = to_torch(np.array([self.teacher.act(s_elem.cpu().numpy()) for s_elem in s ])) # expensive copy back and forth
                if self._config["legacy"]: bc_loss = alpha * nn.functional.mse_loss(a_policy[:,:4],a_teacher)
                else: bc_loss = alpha * nn.functional.mse_loss(a_policy,a_teacher)
                q_loss = - torch.mean(q)
                actor_loss = q_loss + bc_loss
                #logging
                q_loss_value.append( q_loss.item())
                bc_loss_value.append( bc_loss.item())
            else:
                actor_loss = -torch.mean(q)

            actor_loss_value.append( actor_loss.item())
            actor_loss.backward()
            self.optimizer.step()


        if self._config["bc"]:
            return (losses,actor_loss_value,q_loss_value,bc_loss_value)
        else:
            return (losses,actor_loss_value)

    def train(self, iter_fit, max_episodes, max_timesteps,log_interval,save_interval):
        to_torch = lambda x: torch.from_numpy(x.astype(np.float32)).to(self.device)
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
            return np.append(obs,(obs-pastobs)[self._config["derivative_indices"]])
        
        if (self.env_name == "hockey"):
            self.player = h_env.BasicOpponent()

        def opponent_action():
            if (self.env_name == "hockey"):
                obs_agent2 = self.env.obs_agent_two()
                return self.player.act(obs_agent2)
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
            fill_buffer_timesteps = self._config["buffer_size"] // 100
            for t in range(fill_buffer_timesteps):
                # if done or trunc: break
                timestep += 1
                if self._config["derivative"]:  a = self.act(add_derivative(ob,past_obs))
                else :                          a = self.act(ob)
                a2 = opponent_action()
                # a = self.act(past_obs.flatten())
                # a2 = opponent_action(past_obs[-1])

                (ob_new, reward, done, trunc, _info) = self.env.step(np.hstack([a,a2]))
                total_reward+= reward
                
                if self._config["derivative"]:  self.store_transition((add_derivative(ob,past_obs), a, reward, add_derivative(ob_new,ob), done))
                else:                           self.store_transition((ob, a, reward, ob_new, done))
                    
                past_obs = ob
                ob=ob_new

                # self.store_transition((past_obs.flatten(), a, reward, rollrep(past_obs,ob_new).flatten(), done))
                # past_obs = rollrep(past_obs,ob_new)
                if done or trunc: break
            
            fill_buffer_timesteps = max_timesteps

            l = self.train_innerloop(iter_fit)
            losses.extend(l)

            rewards.append(total_reward)
            lengths.append(t)
            if self.wandb_run : 
                if self._config["bc"]:
                    wandb.log({"actor_loss": np.array(l[1]).mean() , "critic_loss": np.array(l[0]).mean() , "reward": total_reward, "length":t, "q_loss": np.array(l[2]).mean(), "bc_loss":np.array(l[3]).mean()})    
                else:
                    wandb.log({"actor_loss": np.array(l[1]).mean() , "critic_loss": np.array(l[0]).mean() , "reward": total_reward, "length":t })

            

            # save every 500 episodes
            if i_episode % save_interval == 0:
                save_checkpoint(self.state(),self.savepath,"DDPG",self.env_name, i_episode, self.wandb_run, self._eps, lr, self.seed,rewards,lengths, losses)

            # logging
            if i_episode % log_interval == 0:
                avg_reward = np.mean(rewards[-log_interval:])
                avg_length = int(np.mean(lengths[-log_interval:]))
                print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, avg_reward))

        if i_episode % 500 != 0: save_checkpoint(self.state(),self.savepath,"DDPG",self.env_name, i_episode, self.wandb_run, self._eps, lr, self.seed,rewards,lengths, losses)
            
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

    ddpg = DDPGAgent(env, env_name, opts.seed, "test", False, eps = eps, learning_rate_actor = lr,
                     update_target_every = opts.update_every)

    # logging variables
    rewards = []
    lengths = []
    losses = []
    timestep = 0

    def save_statistics():
        with open(f"./results/DDPG_{env_name}-eps{eps}-t{train_iter}-l{lr}-s{random_seed}-stat.pkl", 'wb') as f:
            pickle.dump({"rewards" : rewards, "lengths": lengths, "eps": eps, "train": train_iter,
                         "lr": lr, "update_every": opts.update_every, "losses": losses}, f)

    # training loop
    # for i_episode in range(1, max_episodes+1):
    #     ob, _info = env.reset()
    #     ddpg.reset()
    #     total_reward=0
        # for t in range(max_timesteps):
        #     timestep += 1
        #     done = False
        #     a = ddpg.act(ob)
        #     (ob_new, reward, done, trunc, _info) = env.step(a)
        #     total_reward+= reward
        #     ddpg.store_transition((ob, a, reward, ob_new, done))
        #     ob=ob_new
        #     if done or trunc: break

    testing_loss = ddpg.train(train_iter,max_episodes, max_timesteps, log_interval)
    print(testing_loss)


        # rewards.append(total_reward)
        # lengths.append(t)

        # save every 500 episodes
        # if i_episode % 500 == 0:
        #     print("########## Saving a checkpoint... ##########")
        #     torch.save(ddpg.state(), f'./results/DDPG_{env_name}_{i_episode}-eps{eps}-t{train_iter}-l{lr}-s{random_seed}.pth')
        #     save_statistics()

        # logging
        # if i_episode % log_interval == 0:
        #     avg_reward = np.mean(rewards[-log_interval:])
        #     avg_length = int(np.mean(lengths[-log_interval:]))
        #
        #     print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, avg_reward))
    # save_statistics()

if __name__ == '__main__':
    main()
