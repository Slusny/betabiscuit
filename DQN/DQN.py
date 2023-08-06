# London Bielicke

import gymnasium as gym
from gymnasium import spaces
import sys
import numpy as np
import itertools
import time
import torch
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
import laserhockey.hockey_env as h_env
import wandb
import memory_DQN as mem
from feedforward_DQN import Feedforward
from feedforward_DQN import DuelingDQN
sys.path.append('..')
from utility import save_checkpoint

from cpprb import PrioritizedReplayBuffer, ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)

# for plots
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

# discretize openAI env actions
class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, bins = 5):
        """A wrapper for converting a 1D continuous actions into discrete ones.
        Args:
            env: The environment to apply the wrapper
            bins: number of discrete actions
        """
        assert isinstance(env.action_space, spaces.Box)
        super().__init__(env)
        self.bins = bins
        self.orig_action_space = env.action_space
        self.action_space = spaces.Discrete(self.bins)

    def action(self, action):
        """ discrete actions from low to high in 'bins'
        Args:
            action: The discrete action
        Returns:
            continuous action
        """
        return self.orig_action_space.low + action/(self.bins-1.0)*(self.orig_action_space.high-self.orig_action_space.low)

# Dueling net, probably could've implemented this better
# because this is basically the same as the other class
# other than the inheritance... whoops
class DuelingQFunction(DuelingDQN):
    def __init__(self, input_size, hidden_sizes, hidden_sizes_values, hidden_sizes_advantages, 
                output_size, activation_fun, activation_fun_values, activation_fun_advantages, learning_rate, output_activation):
        super().__init__(input_size, hidden_sizes, hidden_sizes_values, hidden_sizes_advantages, 
                output_size, activation_fun, activation_fun_values, activation_fun_advantages, output_activation)

        self.optimizer=torch.optim.Adam(self.parameters(),
                                        lr=learning_rate,
                                        eps=0.000001)
        self.loss = torch.nn.SmoothL1Loss() # MSELoss()

    def fit(self, observations, actions, targets, bc_lambda, weights = None, bc_teacher=None):
        self.train() # put model in training mode
        self.optimizer.zero_grad()
        # Forward pass
        # acts = torch.from_numpy(actions)
        pred = self.Q_value(observations, actions)

        # only used for PER
        if weights is None:
            weights = torch.ones_like(pred)

        # Compute Loss
        q_loss = self.loss(pred*weights, targets*weights)

        if bc_teacher is not None:
            bc_loss = bc_lambda * torch.nn.functional.mse_loss(pred,bc_teacher)
            loss = q_loss + bc_loss
            # Backward pass
            loss.backward()
            self.optimizer.step()
            return (loss.item(), q_loss.item(), bc_loss.item())
        else:
            loss = q_loss
            # Backward pass
            loss.backward()
            self.optimizer.step()
            return (loss.item())

    def Q_value(self, observations, actions):
        actions = actions.squeeze()
        return self.forward(observations).gather(1, (actions[:,None]).type(torch.int64))

    def maxQ(self, observations):
        return torch.max(self.predict(observations), axis=-1, keepdims=True)

    def greedyAction(self, observations):
        return torch.argmax(self.predict(observations), axis=-1)

class QFunction(Feedforward):
    def __init__(self, input_size, hidden_sizes, output_size, activation_fun, output_activation,learning_rate):
        super().__init__(input_size, hidden_sizes, output_size, activation_fun, output_activation)

        self.optimizer=torch.optim.Adam(self.parameters(),
                                        lr=learning_rate,
                                        eps=0.000001)
        self.loss = torch.nn.SmoothL1Loss() # MSELoss()

    def fit(self, observations, actions, targets, bc_lambda, weights = None, bc_teacher=None):
        self.train() # put model in training mode
        self.optimizer.zero_grad()
        # Forward pass
        # acts = torch.from_numpy(actions)
        pred = self.Q_value(observations, actions)

        # only used for PER
        if weights is None:
            weights = torch.ones_like(pred)

        # Compute Loss
        q_loss = self.loss(pred*weights, targets*weights)

        if bc_teacher is not None:
            bc_loss = bc_lambda * torch.nn.functional.mse_loss(pred,bc_teacher)
            loss = q_loss + bc_loss
            # Backward pass
            loss.backward()
            self.optimizer.step()
            return (loss.item(), q_loss.item(), bc_loss.item())
        else:
            loss = q_loss
            # Backward pass
            loss.backward()
            self.optimizer.step()
            return (loss.item())

    def Q_value(self, observations, actions):
        actions = actions.squeeze()
        return self.forward(observations).gather(1, (actions[:,None]).type(torch.int64))

    def maxQ(self, observations):
        return torch.max(self.predict(observations), axis=-1, keepdims=True)

    def greedyAction(self, observations):
        return torch.argmax(self.predict(observations), axis=-1)

class DQNAgent(object):
    """
    Agent implementing Q-learning with NN function approximation.
    """
    def __init__(self, env, env_name, action_n, seed, savepath, wandb_run, **userconfig):
        
        observation_space = env.observation_space
        # if not isinstance(observation_space, spaces.box.Box):
        #     raise UnsupportedSpace('Observation space {} incompatible ' \
        #                            'with {}. (Require: Box)'.format(observation_space, self))
        # if not isinstance(action_space, spaces.discrete.Discrete):
        #     raise UnsupportedSpace('Action space {} incompatible with {}.' \
        #                            ' (Reqire Discrete.)'.format(action_space, self))

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.savepath = savepath
        self.seed = seed
        self.env_name = env_name
        self.env = env
        self._observation_space = observation_space
        self._action_n = action_n

        if self.env_name == "hockey":
            self.player = h_env.BasicOpponent(weak=False)
            action_map = {}
            for i in range(0,12):
                action_map[tuple(discrete_to_continous_action(i))] = i
            self._action_space = spaces.Discrete(len(action_map))
        else: self._action_space = self.env.action_space

        self._config = {
            "eps": 1,            # Epsilon in epsilon greedy policies
            "eps_decay":.9999,
            "min_eps":.01,
            "discount": 0.95,
            "buffer_size": int(1e5),
            "batch_size": 128,
            "learning_rate": 0.0001,
            "update_target_every": 20,
            "tau": 0.001,            # rate for soft updates
            "use_hard_updates":True,
            "double":False,
            "per_own_impl": False,
            "per": False,
            "dueling":False,
            "wandb": False,
            "beta": 0.4,
            "alpha": 0.6,
            "alpha_decay": 1,
            "beta_growth": 1.0001,
            "derivative_indices": [],
            'use_derivative': False,
            #new
            'replay_ratio': 0.25,
            'cpu': False,
            "bc": False,
            "bc_lambda":2.0,
            "bootstrap":None,
            "dense_reward":False,
            "hidden_sizes": [256,128,128],
            "hidden_sizes_values":[128], 
            "hidden_sizes_advantages": [128], 
            "activation_fun":torch.nn.Tanh(),
            "activation_fun_value":torch.nn.ReLU(),
            "activation_fun_advantage":torch.nn.ReLU(),
        }
        self._config.update(userconfig)
        self._eps = self._config['eps']
        self.tau = self._config["tau"]
        self._obs_dim=self._observation_space.shape[0] + len(self._config["derivative_indices"])
        print("Config: ", self._config)

        # if using PER, memory uses PER class
        if self._config['per_own_impl']:
            self.buffer = mem.PrioritizedReplayBuffer(self._config["buffer_size"], alpha = self._config["alpha"], beta = self._config["beta"], alpha_decay = self._config["alpha_decay"], beta_growth= self._config["beta_growth"])
        elif self._config['per']:
            self.buffer = PrioritizedReplayBuffer(self._config["buffer_size"], {
                "obs": {"shape": (self._obs_dim)},
                "act": {},
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

        # dueling nets alg inherits NN from different class
        # note: target doesn't learn, rather updated with Q weights
        if self._config['dueling']:
            self.Q = DuelingQFunction(input_size=self._obs_dim, 
                hidden_sizes=self._config["hidden_sizes"], 
                hidden_sizes_values=self._config["hidden_sizes_values"], 
                hidden_sizes_advantages=self._config["hidden_sizes_advantages"], 
                output_size=self._action_n, 
                activation_fun=torch.nn.Tanh(),
                activation_fun_values=torch.nn.ReLU(),
                activation_fun_advantages=torch.nn.ReLU(), 
                learning_rate = self._config["learning_rate"],
                output_activation=None).to(self.device)
            self.Q_target = DuelingQFunction(input_size=self._obs_dim, 
                hidden_sizes=self._config["hidden_sizes"], 
                hidden_sizes_values=self._config["hidden_sizes_values"], 
                hidden_sizes_advantages=self._config["hidden_sizes_advantages"], 
                output_size=self._action_n, 
                activation_fun=torch.nn.Tanh(),
                activation_fun_values=torch.nn.ReLU(),
                activation_fun_advantages=torch.nn.ReLU(), 
                learning_rate = 0,
                output_activation=None).to(self.device)
        else:
            self.Q = QFunction(input_size=self._obs_dim, 
                hidden_sizes=self._config["hidden_sizes"], 
                output_size=self._action_n, 
                activation_fun=torch.nn.Tanh(), 
                output_activation=None,
                learning_rate = self._config["learning_rate"]).to(self.device)
            self.Q_target = QFunction(input_size=self._obs_dim, 
                hidden_sizes=self._config["hidden_sizes"], 
                output_size=self._action_n, 
                activation_fun=torch.nn.Tanh(), 
                output_activation=None,
                learning_rate = 0).to(self.device)


        if(self._config["bootstrap"] is not None):
            api = wandb.Api()
            art = api.artifact(self._config["bootstrap"], type='model')
            if self._config["cpu"]:
                state = torch.load(art.file(),map_location='cpu')
            else:
                state = torch.load(art.file())
            self.restore_state(state)

        # init Q' weights = Q weights
        self._update_target_net()
        self.train_iter = 0
        
        if self._config["bc"]:
            self.teacher = h_env.BasicOpponent(weak=False)

         # log gradients to W&B
        self.wandb_run = wandb_run
        if(wandb_run):
            wandb.watch(self.Q, log_freq=100)


    def get_config(self):
        if self._config['per_own_impl']:
            self._config["beta"] = self.buffer.beta
        return self._config

    def _update_target_net(self):
        self.Q_target.load_state_dict(self.Q.state_dict())

    def soft_update_target_net(self):
        if self._config['tau'] == 1.:
            self._update_target_net()
        else:
            for target_param, param in zip(self.Q_target.parameters(), self.Q.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

    def act(self, observation, eps=None):
        to_torch = lambda x: torch.from_numpy(x.astype(np.float32)).to(self.device)
        observation = to_torch(observation)
        if eps is None:
            eps = self._eps
            self._eps = max(self._config["eps_decay"]*eps, self._config["min_eps"])

        # epsilon greedy
        if np.random.random() > eps:
            action = self.Q.greedyAction(observation)
            action = action.cpu().numpy()
            # action = discrete_to_continous_action(action)
        else:
            action = self._action_space.sample()#[:4]

        return action

    def store_transition(self, transition):
        if self._config["per"]:
            self.buffer.add(obs=transition[0], act=transition[1], rew=transition[2], next_obs=transition[3], done=transition[4])
        else:
            self.buffer.add_transition(transition)

    def sample_replaybuffer(self):
        to_torch = lambda x: torch.from_numpy(x.astype(np.float32)).to(self.device)
        if self._config["per"]:
            data = self.buffer.sample(self._config['batch_size'])
            s, a, rew = data['obs'], data['act'], data['rew']
            #actions_h, rewards = data['intervene'], data['acte']
            s_prime, done, idxs = data['next_obs'], data['done'], data['indexes']
            weights = None
        else:
            if isinstance(self.buffer, mem.PrioritizedReplayBuffer):
                data, weights, idxs = self.buffer.sample(self._config['batch_size'])
            else:
                data=self.buffer.sample(batch=self._config['batch_size'])
            s = np.stack(data[:,0]) # s_t
            a = np.stack(data[:,1]) # a_t
            rew = np.stack(data[:,2])[:,None] # rew  (batchsize,1)
            s_prime = np.stack(data[:,3]) # s_t+1
            done = np.stack(data[:,4])[:,None] # done signal  (batchsize,1)
            idxs = None
            weights = None
        return to_torch(s),to_torch(a),to_torch(rew),to_torch(s_prime),to_torch(done),idxs, weights

    def restore_state(self,state):
        self.Q.load_state_dict(state)
        self._update_target_net()

    def reset(self):
        self.action_noise.reset()
    
    def get_teacher_actions(self, s):
        to_torch = lambda x: torch.from_numpy(x.astype(np.float32)).to(self.device)
        return to_torch(np.array([continuous_to_discrete_action(self.teacher.act(s_elem.cpu().numpy())) for s_elem in s ]))

    def train_innerloop(self, iter_fit=32):
        losses = []
        self.train_iter+=1

        for i in range(iter_fit):

            # sample from the replay buffer
            s,a,rew,s_prime,done,idxs, weights = self.sample_replaybuffer()

            if self._config["double"]:
                # get Q values from frozen network for next state and chosen action
                # Q(s',argmax(Q(s',a', theta_i), theta_i_frozen)) (argmax wrt a')
                acts = self.Q.greedyAction(s_prime)
                v_prime = self.Q_target.Q_value(s_prime, acts)
            else:
                # get Q values from frozen network for next state (no argmax, just one max val)
                # Q(s',a', theta_i_frozen))
                v_prime = self.Q_target.maxQ(s_prime)

            # target
            gamma=self._config['discount']
            td_target = rew + gamma * (1.0-done) * v_prime.detach()
            
            if self._config["per"]:
                pred = self.Q.Q_value(s,a).detach()
                td_error = abs(td_target - pred).cpu().numpy()
                self.buffer.update_priorities(idxs, td_error)
            
            if self._config["bc"]:
                teacher = self.get_teacher_actions(s)
            else:
                teacher=None

            if isinstance(self.buffer, mem.PrioritizedReplayBuffer):
                # same pred as fit function for priority update
                # probably also bad implementation whoops
                pred = self.Q.Q_value(s, a).detach()
                # Compute TD error
                td_error = torch.abs(pred - td_target).cpu().numpy()
                self.buffer.update_priorities(idxs, td_error)
                # same as regular replay buffer but with weights
                fit_loss = self.Q.fit(s, a, td_target, self._config["bc_lambda"], weights = weights, bc_teacher=teacher)
            else:
                # optimize the lsq objective
                fit_loss = self.Q.fit(s, a, td_target, self._config["bc_lambda"], weights = None, bc_teacher=teacher)

            self.soft_update_target_net()

            losses.append(fit_loss)

        return (losses, None)

    def train(self,iter_fit, max_episodes, max_timesteps,log_interval,save_interval):
        #train( exploration=False, wandb_track=False, load_model = None, save_model= None, dueling = False, env_name = "hockey", epoch = 1000, discount = .95, hard_updates = True, target_update = 20, beta = .4, tau=.001, eps=1
        
        def add_derivative(obs,pastobs):
            return np.append(obs,(obs-pastobs)[self._config["derivative_indices"]])
        
        def opponent_action():
            if (self.env_name == "hockey"):
                obs_agent2 = self.env.obs_agent_two()
                return self.player.act(obs_agent2)
            else:
                return np.array([0,0.,0,0])

        stats = []
        rewards = []
        lengths = []
        losses = []
        timestep = 0        

        # training loop
        fill_buffer_timesteps = self._config["buffer_size"] // 100
        for i_episode in range(1, max_episodes+1):
            # if i == 1400:
            #     env = gym.make(env_name, render_mode="human")
            #     if isinstance(env.action_space, spaces.Box):
            #         env = DiscreteActionWrapper(env,5)
            #     ac_space = env.action_space
            
            # self.reset()
            total_reward = 0
            ob, _info = self.env.reset()
            past_obs = ob.copy()
            added_transitions = 0
            while True:
                for t in range(max_timesteps):
                    timestep += 1
                    done = False
                    # env.render()
                    if self._config["derivative"]:  a = self.act(add_derivative(ob,past_obs))
                    else :                          a = self.act(ob)
                    
                    if self.env_name == "hockey":
                        # a1 = a
                        a2 = opponent_action()
                        a_step = np.hstack([discrete_to_continous_action(a),a2])
                    (ob_new, reward, done, trunc, _info) = self.env.step(a_step)
                    # reward = _info["winner"]*10
                    if(self._config["dense_reward"]): 
                        reward = reward + _info["reward_closeness_to_puck"] + _info["reward_touch_puck"] + _info["reward_puck_direction"]
                    total_reward+=reward
                    if self._config["derivative"]:  self.store_transition((add_derivative(ob,past_obs), a, reward, add_derivative(ob_new,ob), done))
                    else:                           self.store_transition((ob, a, reward, ob_new, done)) # a True was once saved here
                    
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

            loss = self.train_innerloop(iter_fit)
            rewards.append(total_reward)
            lengths.append(t)
            losses.append(loss)
            stats.append([i_episode,total_reward,t+1])

            if self.wandb_run : 
                if self._config["bc"]:
                    wandb.log({"actor_loss": np.array(loss[0]).mean()  , "reward": total_reward, "length":t, "q_loss": np.array(loss[1]).mean(), "bc_loss":np.array(loss[2]).mean()})    
                else:
                    wandb.log({"actor_loss": np.array(loss[0]).mean() , "reward": total_reward, "length":t })

            # logging
            if i_episode % log_interval == 0:
                avg_reward = np.mean(rewards[-log_interval:])
                avg_length = int(np.mean(lengths[-log_interval:]))
                print('Episode {} \t avg length: {} \t reward: {} \t eps: {}'.format(i_episode, avg_length, avg_reward, self._eps))

            # save every 500 episodes
            if i_episode % save_interval == 0:
                save_checkpoint(self.Q.state_dict(),self.savepath,"DQN",self.env_name, i_episode, self.wandb_run, self._eps, self._config["learning_rate"], self.seed,rewards,lengths, losses)

        # clean up
        self.env.close()
        stats_np = np.asarray(stats)
        losses_np = np.asarray(losses)
    
        # final save
        if i_episode % log_interval != 0: save_checkpoint(self.Q.state_dict(),self.savepath,"DQN",self.env_name, i_episode, self.wandb_run, self._eps, lr, self.seed,rewards,lengths, losses)
        


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

# easy mapping
def continuous_to_discrete_action(continuous_action):
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
        '''
    high_action = abs(continuous_action) > 0.6
    if(high_action[3] == 1): return 7
    if sum(abs(high_action)) == 0: return 0
    # only 2 simultaneous actions allowed
    if sum(abs(high_action)) == 1:
        index = np.argmax(continuous_action)
        if continuous_action[index] > 0: return (index + 1)*2
        else: return (index + 1)*2 -1
        return  + 1
    
# ToDo  
    if sum(abs(high_action)) > 2:
        ordered_actions = np.argsort(abs(continuous_action))[::-1]
        if ordered_actions[0] == 3: return 7


        high_action = abs(continuous_action) > 0.7
        if sum(abs(continuous_action)) == 3: 
            high_action = abs(continuous_action) > 0.8
            if sum(abs(continuous_action)) == 3: 
                high_action = abs(continuous_action) > 0.9
    if sum(abs(continuous_action)) == 1:
        if continuous_action[0] == 1: return 1
        if continuous_action[0] == -1: return 1
        if continuous_action[1] == 1: return 4
        if continuous_action[1] == -1: return 3
        if continuous_action[2] == 1: return 6
        if continuous_action[2] == -1: return 5

    return map[tuple(continuous_action)]




def run(model, env_name="hockey"):
    if env_name == "hockey":
        env = h_env.HockeyEnv()
        action_map = {}
        for i in range(0,12):
            action_map[tuple(discrete_to_continous_action(i))] = i
        ac_space = spaces.Discrete(len(action_map))
        player2 = h_env.BasicOpponent(weak=False)
        obs_agent2 = env.obs_agent_two()
    else:
        env = gym.make(env_name, render_mode="rgb_array")
        if isinstance(env.action_space, spaces.Box):
            env = DiscreteActionWrapper(env,5)
        ac_space = env.action_space

    o_space = env.observation_space

    # q_agent = DQNAgent(o_space, ac_space, discount=0, eps=0,
    #                    use_hard_updates = False, tau = 0, update_target_every= 20, double=False, priority=False,
    #                     wandb = False, load_model = model, dueling = True)

    q_agent.Q.load_state_dict(torch.load(model).state_dict())

    b,_info = env.reset()

    stats = []

    max_episodes=20000
    max_steps=9999999999
    avg_total_reward = 0
    p2 = 0
    p1 = 0
    for i in range(max_episodes):
        total_reward = 0
        ob, _info = env.reset()
        for t in range(max_steps):
            done = False
            env.render()
            a = q_agent.act(ob)
            a_step = a
            if env_name == "hockey":
                a1 = env.discrete_to_continous_action(a)
                a2 = player2.act(obs_agent2)
                # a2 = [0,0.,0,0]
                a_step = np.hstack([a1,a2])
                obs_agent2 = env.obs_agent_two()
            (ob_new, reward, done, trunc, _info) = env.step(a_step)
            # print(_info)
            total_reward+=reward
            ob=ob_new
            if done: break
        stats.append([i,total_reward,t+1])
        avg_total_reward += total_reward

        if _info["winner"] == 1:
            p1 += 1
        if _info["winner"] == -1:
            p2 += 1
        if ((i-1)%200==0):
            print(str(p1) + " vs " + str(p2))
            print("{}: Done after {} steps. Reward: {}".format(i, t+1, _info["winner"]))

    # print(stats)
    env.close()

def main():
    # fig=plt.figure(figsize=(6,3.8))
    # # run(dueling = True, ddqn = True, exploration= True, wandb_track = False)
    # training(dueling = False, ddqn = False, exploration= False, save_model="test_basic", wandb_track =True, hard_updates=False, tau = .001, discount = .95)
    # # run(ddqn = False, exploration= False, wandb_track = False)
    # run(ddqn = False, exploration= True, wandb_track = False)
    # plt.show()

    run("BEST")



if __name__ == '__main__':
    main()


# NOTES:
    # next steps:
    # - cmd swtiches
    # - param testing
    # - dueling nets?
    # - run tests
