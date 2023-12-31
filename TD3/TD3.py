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
import collections
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


class QFunction(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, learning_rate, activation_fun=torch.nn.ReLU(), output_activation=None):
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
        
        self.loss = nn.MSELoss() # or use torch.nn.SmoothL1Loss()?

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
    def __init__(self, env, env_name, action_n, seed, savepath, wandb_run, userconfig):

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
            "dense_reward": False,
            "HiL": False,
            "bc": False,
            "bc_lambda":2.0,
            "cpu": False,
            "replay_ratio": 0,
            "batchnorm": False,
            "validation_interval": float("inf"),
            "validation_episodes": 200,
            "filled_buffer_ratio": 100,

        }
        self._config.update(userconfig)
        for key in ["hidden_sizes_actor",'hidden_sizes_critic']:
            if type(self._config[key]) == str:
                self._config[key] = eval(self._config[key])
        self._eps = self._config['eps']
        self._obs_dim=self._observation_space.shape[0] + len(self._config["derivative_indices"])
        print("Config: ", self._config)

        self.action_noise = OUNoise((self._action_n))

        if self._config["per"]:
            self.buffer = PrioritizedReplayBuffer(self._config["buffer_size"], {
                "obs": {"shape": (self._obs_dim)},
                "act": {"shape": (self._action_n)},
                "rew": {},
                "next_obs": {"shape": (self._obs_dim)},
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
        self.Q = QFunction(input_size=self._obs_dim+self._action_n,#self._obs_dim*self._config["past_states"],
                           hidden_sizes= self._config["hidden_sizes_critic"],
                           learning_rate = self._config["learning_rate_critic"]).to(self.device)
        # target Q Network
        self.Q_target = QFunction(input_size=self._obs_dim+self._action_n,#self._obs_dim*self._config["past_states"],
                                  hidden_sizes= self._config["hidden_sizes_critic"],
                                  learning_rate = 0).to(self.device)

        self.policy = Feedforward(input_size=self._obs_dim,#self._obs_dim*self._config["past_states"],
                                  hidden_sizes= self._config["hidden_sizes_actor"],
                                  output_size=self._action_n,
                                  activation_fun = torch.nn.ReLU(),
                                  output_activation = torch.nn.Tanh(),
                                  batchnorm=self._config["batchnorm"]
                                  ).to(self.device)
        self.policy_target = Feedforward(input_size=self._obs_dim,#self._obs_dim*self._config["past_states"],
                                         hidden_sizes= self._config["hidden_sizes_actor"],
                                         output_size=self._action_n,
                                         activation_fun = torch.nn.ReLU(),
                                         output_activation = torch.nn.Tanh(),
                                         batchnorm=self._config["batchnorm"]
                                         ).to(self.device)
        
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
                print("Loading bootstrap model from: ", art.file())
                if self._config["cpu"]:
                    state = torch.load(art.file(),map_location='cpu')
                else:
                    state = torch.load(art.file())
                self.restore_state(state)
        
        if self._config["bc"]:
            self.teacher = h_env.BasicOpponent(weak=False)

        
        # copy initialized weights
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.policy_target.load_state_dict(self.policy.state_dict())

        self.optimizer=torch.optim.Adam(self.policy.parameters(),
                                        lr=self._config["learning_rate_actor"],
                                        eps=0.000001)
        self.train_iter = 0

        # log gradients to W&B
        self.wandb_run = wandb_run
        # if(wandb_run):
        #     wandb.watch(self.Q, log_freq=100)
        #     wandb.watch(self.policy, log_freq=100)

    def save_buffer(self, path):
        print("saving buffer to: ", path, " ...")
        if self._config["per"]:
            self.buffer.save_transitions(path)
        else:
            self.buffer.save(path)

    def load_buffer(self, path):
        print("loading buffer from: ", path, " ...")
        if self._config["per"]:
            self.buffer.load_transitions(path)
        else:
            self.buffer.load(path)

    def _copy_nets(self):
        # Full copy
        if self._config["tau"] == 1:
            self.Q_target.load_state_dict(self.Q.state_dict())
            self.policy_target.load_state_dict(self.policy.state_dict())
        
        # Convex update
        else:
            for param, target_param in zip(self.policy.parameters(), self.policy_target.parameters()): 
                target_param.data.copy_(self._config["tau"] * param.data + (1 - self._config["tau"]) * target_param.data)

            for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
                target_param.data.copy_(self._config["tau"] * param.data + (1 - self._config["tau"]) * target_param.data)

    def act(self, observation, eps=None):
        if eps is None:
            eps = self._eps
        #
        observation = torch.from_numpy(observation.astype(np.float32)).to(self.device) 
        action = (self.policy.predict(observation) + eps*self.action_noise()).clip(-1,1)  # action in -1 to 1 (+ noise)
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
        self.Q.load_state_dict(state[0])
        self.policy.load_state_dict(state[1])
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
        losses = []
        actor_loss_value = []
        q_loss_value = []
        bc_loss_value = []
        to_torch = lambda x: torch.from_numpy(x.astype(np.float32)).to(self.device)
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
            td_target = rew + gamma * (1.0-done) * q_prime.detach()

            # prediction for priotized replay buffer - needs copying back to cpu -> performance hit
            if self._config["per"]:
                pred1 = self.Q.Q1_value(s,a).detach()
                priorities = abs(td_target - pred1).cpu().numpy()
                self.buffer.update_priorities(idxs, priorities) 

            # optimize the Q objective ( Critic )
            fit_loss = self.Q.fit(s, a, td_target)
            losses.append(fit_loss)

            # Delay Polciy Updates (TD3 paper 5.2)
            if self.train_iter % self._config["update_target_every"] == 0:
                self.optimizer.zero_grad()
                a_policy = self.policy.forward(s)
                q = self.Q.Q1_value(s, a_policy)
                if self._config["bc"]:
                    alpha = self._config["bc_lambda"]# * q.mean().detach()
                    a_teacher = to_torch(np.array([self.teacher.act(s_elem.cpu().numpy()) for s_elem in s ])).detach() # expensive copy back and forth
                    bc_loss = alpha *nn.functional.mse_loss(a_policy,a_teacher)
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

                self._copy_nets() # with a update_frequency of 2 and a tau of 0.005 a "full" update is done every 400 steps

            self.train_iter+=1
        if self._config["bc"]:
            return (losses,actor_loss_value,q_loss_value,bc_loss_value)
        else:
            return (losses,actor_loss_value)

    # Outer loop where the replay buffer gets filled
    def train(self, iter_fit, max_episodes, max_timesteps,log_interval,save_interval):
        to_torch = lambda x: torch.from_numpy(x.astype(np.float32)).to(self.device)
         # logging variables
        rewards = []
        lengths = []
        losses = []
        timestep = 0
        lr = self._config['learning_rate_actor']

        def add_derivative(obs,pastobs):
            return np.append(obs,(obs-pastobs)[self._config["derivative_indices"]])
        
        if (self.env_name == "hockey"):
            self.player = h_env.BasicOpponent(weak=False)

        # training loop
        fill_buffer_timesteps = self._config["buffer_size"] // self._config["filled_buffer_ratio"]
        for i_episode in range(1, max_episodes+1):
            # validate
            if i_episode % self._config["validation_interval"] == 0 and timestep > fill_buffer_timesteps: self.validate()

            while True:
                ob, _info = self.env.reset()
                # Incorporate  Acceleration
                past_obs = ob.copy()
                self.reset()
                total_reward=0
                added_transitions = 0
                for t in range(max_timesteps):
                    timestep += 1
                    if self._config["derivative"]:  a = self.act(add_derivative(ob,past_obs))
                    else :                          a = self.act(ob)
                    
                    a2 = self.opponent_action()

                    (ob_new, reward, done, trunc, _info) = self.env.step(np.hstack([a,a2]))
                    if(self._config["dense_reward"]): 
                        reward = reward + _info["reward_closeness_to_puck"] + _info["reward_touch_puck"] + _info["reward_puck_direction"]
                    total_reward+= reward
                    
                    if self._config["derivative"]:  self.store_transition((add_derivative(ob,past_obs), a, reward, add_derivative(ob_new,ob), done))
                    else:                           self.store_transition((ob, a, reward, ob_new, done))
                    
                    added_transitions += 1
                    past_obs = ob
                    ob=ob_new

                    if done or trunc: break
                    
                # To fill buffer once before training
                if(timestep > fill_buffer_timesteps):
                    break
                elif timestep == fill_buffer_timesteps:                  
                    print("Buffer filled")
                    added_transitions = 1

            if(self._config["replay_ratio"] != 0):
                iter_fit = int(added_transitions * self._config["replay_ratio"]) + 1  

            l = self.train_innerloop(iter_fit)
            # losses.extend(l)

            rewards.append(total_reward)
            lengths.append(t)
            if self.wandb_run : 
                if self._config["bc"]:
                    wandb.log({"actor_loss": np.array(l[1]).mean() , "critic_loss": np.array(l[0]).mean(axis=0) , "reward": total_reward, "length":t, "q_loss": np.array(l[2]).mean(), "bc_loss":np.array(l[3]).mean()})    
                else:
                    wandb.log({"actor_loss": np.array(l[1]).mean() , "critic_loss": np.array(l[0]).mean(axis=0) , "reward": total_reward, "length":t })

            # save every 500 episodes
            if i_episode % save_interval == 0:
                save_checkpoint(self.state(),self.savepath,"TD3",self.env_name, i_episode, self.wandb_run, self._eps, lr, self.seed,rewards,lengths, losses)

            # logging
            if i_episode % log_interval == 0:
                avg_reward = np.mean(rewards[-log_interval:])
                avg_length = int(np.mean(lengths[-log_interval:]))
                print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, avg_reward))

        if i_episode % log_interval != 0: save_checkpoint(self.state(),self.savepath,"TD3",self.env_name, i_episode, self.wandb_run, self._eps, lr, self.seed,rewards,lengths, losses)
            
        return losses
    
    def save_agent_wandb(self, i_episode, rewards, lengths, losses,notes=""):
        save_checkpoint(self.state(),self.savepath,"TD3",self.env_name, i_episode, self.wandb_run, self._eps, None, self.seed,rewards,lengths, losses,notes=notes)

    def opponent_action(self):
        if (self.env_name == "hockey"):
            obs_agent2 = self.env.obs_agent_two()
            return self.player.act(obs_agent2)
        else:
            return np.array([0,0.,0,0])
            
    def train_human_in_the_loop(self, iter_fit, max_episodes, max_timesteps,log_interval,save_interval):
        to_torch = lambda x: torch.from_numpy(x.astype(np.float32)).to(self.device)
         # logging variables
        rewards = []
        lengths = []
        losses = []
        timestep = 0
        lr = self._config['learning_rate_actor']

        def add_derivative(obs,pastobs):
            return np.append(obs,(obs-pastobs)[self._config["derivative_indices"]])
        
        if (self.env_name == "hockey"):
            self.player = h_env.BasicOpponent(weak=False)

        # training loop
        for i_episode in range(1, max_episodes+1):
            ob, _info = self.env.reset()
            # Incorporate  Acceleration
            past_obs = ob.copy()
            self.reset()
            total_reward=0

            for t in range(max_timesteps):
                timestep += 1
                if self._config["derivative"]:  a = self.act(add_derivative(ob,past_obs))
                else :                          a = self.act(ob)
                
                a2 = self.opponent_action()
                
                # Human Action
                a_h = input()
                human_scaling = 0.8
                if (a_h == 'w'):
                    a_h = np.array([0,1,0,0])
                if (a_h == 's'):
                    a_h = np.array([0,-1,0,0])
                if (a_h == 'a'):
                    a_h = np.array([-1,0,0,0])
                if (a_h == 'd'):
                    a_h = np.array([1,0,0,0])
                if (a_h == 'r'):
                    a_h = np.array([0,0,1,0])
                if (a_h == 'f'):
                    a_h = np.array([0,0,-1,0])
                a_h = a_h * human_scaling

                (ob_new, reward, done, trunc, _info) = self.env.step(np.hstack([a_h,a2]))
                if(self._config["dense_reward"]): 
                    reward = reward + _info["reward_closeness_to_puck"] + _info["reward_touch_puck"] + _info["reward_puck_direction"]
                total_reward+= reward
                
                self.store_transition((add_derivative(ob,past_obs), a, reward, add_derivative(ob_new,ob), done))
                past_obs = ob
                ob=ob_new

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
    
    def validate(self):
        print("Validating...")
        length = []
        rewards = []
        for i_episode in range(1, self._config['validation_episodes']+1):
            ob, _info = self.env.reset()
            total_reward = 0
            for t in range(500):
                done = False
                a2 = self.opponent_action()
                a = self.act(ob,eps=0.0)
                (ob_new, reward, done, trunc, _info) = self.env.step(np.hstack([a,a2]))
                total_reward+= reward
                ob=ob_new
                if done: 
                    length.append(t)
                    rewards.append(total_reward)
                    break

        print("\t avg length: ",np.array(length).mean(), ", avg reward: ",np.array(rewards).mean())
        if self.wandb.run is not None:
            wandb.log({"validation_length": np.array(length).mean(), "validation_reward": np.array(rewards).mean()})

