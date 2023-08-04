# London Bielicke

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import itertools
import time
import torch
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import laserhockey.hockey_env as h_env
import wandb
import memory as mem
from feedforward import Feedforward
from feedforward import DuelingDQN

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
    def __init__(self, observation_dim, action_dim, hidden_sizes=[100,100],
                 learning_rate = 0.0002):
        super().__init__(input_size=observation_dim,
                         output_size=action_dim)

        self.optimizer=torch.optim.Adam(self.parameters(),
                                        lr=learning_rate,
                                        eps=0.000001)
        self.loss = torch.nn.SmoothL1Loss() # MSELoss()

    def fit(self, observations, actions, targets, weights = None):
        self.train() # put model in training mode
        self.optimizer.zero_grad()
        # Forward pass
        acts = torch.from_numpy(actions)
        pred = self.Q_value(torch.from_numpy(observations).float(), acts)

        # only used for PER
        if weights is None:
            weights = torch.ones_like(pred)

        # Compute Loss
        loss = self.loss(pred*weights, torch.from_numpy(targets).float()*weights)

        # Backward pass
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def Q_value(self, observations, actions):
        return self.forward(observations).gather(1, actions[:,None])

    def maxQ(self, observations):
        return np.max(self.predict(observations), axis=-1, keepdims=True)

    def greedyAction(self, observations):
        return np.argmax(self.predict(observations), axis=-1)

class QFunction(Feedforward):
    def __init__(self, observation_dim, action_dim, hidden_sizes=[100,100],
                 learning_rate = 0.0002):
        super().__init__(input_size=observation_dim, hidden_sizes=hidden_sizes,
                         output_size=action_dim)

        self.optimizer=torch.optim.Adam(self.parameters(),
                                        lr=learning_rate,
                                        eps=0.000001)
        self.loss = torch.nn.SmoothL1Loss() # MSELoss()

    def fit(self, observations, actions, targets, weights = None):
        self.train() # put model in training mode
        self.optimizer.zero_grad()
        # Forward pass
        acts = torch.from_numpy(actions)
        pred = self.Q_value(torch.from_numpy(observations).float(), acts)

        # only used for PER
        if weights is None:
            weights = torch.ones_like(pred)

        # Compute Loss
        loss = self.loss(pred*weights, torch.from_numpy(targets).float()*weights)

        # Backward pass
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def Q_value(self, observations, actions):
        return self.forward(observations).gather(1, actions[:,None])

    def maxQ(self, observations):
        return np.max(self.predict(observations), axis=-1, keepdims=True)

    def greedyAction(self, observations):
        return np.argmax(self.predict(observations), axis=-1)

class DQNAgent(object):
    """
    Agent implementing Q-learning with NN function approximation.
    """
    def __init__(self, observation_space, action_space, **userconfig):

        if not isinstance(observation_space, spaces.box.Box):
            raise UnsupportedSpace('Observation space {} incompatible ' \
                                   'with {}. (Require: Box)'.format(observation_space, self))
        if not isinstance(action_space, spaces.discrete.Discrete):
            raise UnsupportedSpace('Action space {} incompatible with {}.' \
                                   ' (Reqire Discrete.)'.format(action_space, self))

        self._observation_space = observation_space
        self._action_space = action_space
        self._action_n = action_space.n
        self._obs_dim=self._observation_space.shape[0]

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
            'use_derivatives': False,
        }
        self._config.update(userconfig)
        self._eps = self._config['eps']
        self.tau = self._config["tau"]

        # if using PER, memory uses PER class
        if self._config['per_own_impl']:
            self.buffer = mem.PrioritizedReplayBuffer(self._config["buffer_size"], alpha = self._config["alpha"], beta = self._config["beta"], alpha_decay = self._config["alpha_decay"], beta_growth= self._config["beta_growth"])
        elif self._config['per']:
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


        # dueling nets alg inherits NN from different class
        # note: target doesn't learn, rather updated with Q weights
        if self._config['dueling']:
            self.Q = DuelingQFunction(observation_dim=self._observation_space.shape[0],
                               action_dim=self._action_n,
                               learning_rate = self._config["learning_rate"])
            self.Q_target = DuelingQFunction(observation_dim=self._observation_space.shape[0],
                                      action_dim=self._action_n,
                                      learning_rate = 0)
        else:
            self.Q = QFunction(observation_dim=self._observation_space.shape[0],
                               action_dim=self._action_n,
                               learning_rate = self._config["learning_rate"])
            self.Q_target = QFunction(observation_dim=self._observation_space.shape[0],
                                      action_dim=self._action_n,
                                      learning_rate = 0)

        # init Q' weights = Q weights
        self._update_target_net()
        self.train_iter = 0

        if(self._config["wandb"]): # log gradients to W&B
            # start a new wandb run to track this script
            wandb.init(
                # set the wandb project where this run will be logged
                project="dqn-test",
            )
            wandb.watch(self.Q, log_freq=100)


        if(self._config["wandb"]): # log gradients to W&B
            # start a new wandb run to track this script
            wandb.init(
                # set the wandb project where this run will be logged
                project="dqn-test",
            )
            wandb.watch(self.Q, log_freq=100)
            wandb.watch(self.Q_target, log_freq=100)


    def get_config(self):
        if self._config['per_own_impl']:
            self._config["beta"] = self.buffer.beta
        return self._config

    def _update_target_net(self):
        self.Q_target.load_state_dict(self.Q.state_dict())

    def soft_update_target_net(self):
        for target_param, param in zip(self.Q_target.parameters(), self.Q.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

    def act(self, observation, eps=None):
        if eps is None:
            eps = self._eps
            self._eps = max(self._config["eps_decay"]*eps, self._config["min_eps"])


        # epsilon greedy
        if np.random.random() > eps:
            action = self.Q.greedyAction(observation)
        else:
            action = self._action_space.sample()

        return action

    def store_transition(self, transition):
        self.buffer.add_transition(transition)


    def train(self, iter_fit=32):
        losses = []
        self.train_iter+=1

        if self._config["use_hard_updates"] and self.train_iter % self._config["update_target_every"] == 0:
            self._update_target_net()

        for i in range(iter_fit):

            # sample from the replay buffer
            if isinstance(self.buffer, mem.PrioritizedReplayBuffer):
                data, weights, tree_idxs = self.buffer.sample(self._config['batch_size'])
            else:
                data=self.buffer.sample(batch=self._config['batch_size'])

            s = np.stack(data[:,0]) # s_t
            a = np.stack(data[:,1]) # a_t
            rew = np.stack(data[:,2])[:,None] # rew  (batchsize,1)
            s_prime = np.stack(data[:,3]) # s_t+1
            done = np.stack(data[:,4])[:,None] # done signal  (batchsize,1)

            if self._config["double"]:
                # get Q values from frozen network for next state and chosen action
                # Q(s',argmax(Q(s',a', theta_i), theta_i_frozen)) (argmax wrt a')
                acts = torch.from_numpy(self.Q.greedyAction(s_prime))
                v_prime = self.Q_target.Q_value(torch.from_numpy(s_prime).float(), acts).cpu().detach().numpy()
            else:
                # get Q values from frozen network for next state (no argmax, just one max val)
                # Q(s',a', theta_i_frozen))
                v_prime = self.Q_target.maxQ(s_prime)

            # target
            gamma=self._config['discount']
            td_target = rew + gamma * (1.0-done) * v_prime

            if isinstance(self.buffer, mem.PrioritizedReplayBuffer):
                # same pred as fit function for priority update
                # probably also bad implementation whoops
                pred = self.Q.Q_value(torch.from_numpy(s).float(), torch.from_numpy(a)).detach().numpy()
                # Compute TD error
                td_error = np.abs(pred - td_target)
                self.buffer.update_priorities(tree_idxs, td_error)
                # same as regular replay buffer but with weights
                fit_loss = self.Q.fit(s, a, td_target, weights = weights)

            else:
                # optimize the lsq objective
                fit_loss = self.Q.fit(s, a, td_target)

            if not self._config["use_hard_updates"]:
                self.soft_update_target_net()

            losses.append(fit_loss)

        return losses

# added more actions
def discrete_to_continous_action(discrete_action, env):
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
    action_cont = [((discrete_action == 1) | (discrete_action == 8) | (discrete_action == 9)) * -1 + ((discrete_action == 2) | (discrete_action == 10) | (discrete_action == 11)) * 1,  # player x
                   ((discrete_action == 3) | (discrete_action == 8) | (discrete_action == 10)) * -1 + ((discrete_action == 4) | (discrete_action == 9) | (discrete_action == 11)) * 1,  # player y
                   (discrete_action == 5) * -1 + (discrete_action == 6) * 1]  # player angle
    if env.keep_mode:
      action_cont.append(discrete_action == 7)

    return action_cont

# easy mapping
def continuous_to_discrete_action(continuous_action, map):
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

    return map[tuple(continuous_action)]


def training(ddqn=False, exploration=False, wandb_track=False, load_model = None, save_model= None, dueling = False, env_name = "hockey", epoch = 1000, discount = .95, hard_updates = True, target_update = 20, beta = .4, tau=.001, eps=1):

    if env_name == "hockey":
        env = h_env.HockeyEnv()
        action_map = {}
        for i in range(0,12):
            action_map[tuple(discrete_to_continous_action(i, env))] = i
        ac_space = spaces.Discrete(len(action_map))
        player2 = h_env.BasicOpponent(weak=True)
        obs_agent2 = env.obs_agent_two()
    else:
        env = gym.make(env_name, render_mode="rgb_array")
        if isinstance(env.action_space, spaces.Box):
            env = DiscreteActionWrapper(env,5)
        ac_space = env.action_space

    o_space = env.observation_space
    print(ac_space)
    print(o_space)
    print(list(zip(env.observation_space.low, env.observation_space.high)))

    np.random.seed(0)

    stats = []
    losses = []



    q_agent = DQNAgent(o_space, ac_space, discount=discount, eps=eps,
                       use_hard_updates = hard_updates, tau = tau, update_target_every= target_update, double=ddqn, priority=exploration,
                        wandb = wandb_track, load_model = load_model, dueling = dueling, beta = beta)

    if load_model is not None:
        q_agent.Q.load_state_dict(torch.load(load_model).state_dict())
        stats = np.load('test.npy').tolist()

    print(q_agent.get_config())


    ob,_info = env.reset()

    max_episodes=50000
    max_steps=50
    avg_total_reward = 0
    for i in range(max_episodes):
        # if i == 1400:
        #     env = gym.make(env_name, render_mode="human")
        #     if isinstance(env.action_space, spaces.Box):
        #         env = DiscreteActionWrapper(env,5)
        #     ac_space = env.action_space
        total_reward = 0
        ob, _info = env.reset()
        for t in range(max_steps):
            done = False
            # env.render()
            a = q_agent.act(ob)
            a_step = a
            if env_name == "hockey":
                a1 = env.discrete_to_continous_action(a)
                a2 = player2.act(obs_agent2)
                # a2 = [0,0.,0,0]
                a_step = np.hstack([a1,a2])
                obs_agent2 = env.obs_agent_two()
            (ob_new, reward, done, trunc, _info) = env.step(a_step)
            # reward = _info["winner"]*10
            total_reward+=reward
            q_agent.store_transition((ob, a, reward, ob_new, done, True))
            ob=ob_new
            if done: break
        loss = q_agent.train(32)
        losses.extend(loss)
        stats.append([i,total_reward,t+1])
        if wandb_track:
            wandb.log({"loss": loss[len(loss)-1] , "reward": total_reward, "length":t })
        avg_total_reward += total_reward

        if wandb:
            wandb.log({"loss": loss[len(loss)-1] , "reward": total_reward, "length":t })

        if ((i-1)%20==0):
            if wandb_track:
                wandb.log({"loss": loss[len(loss)-1] , "reward": avg_total_reward/100, "length":t })
                avg_total_reward = 0
            print("{}: Done after {} steps. Loss: {}".format(i, t+1, loss[len(loss)-1]))
            print("{}: Done after {} steps. Reward: {}".format(i, t+1, total_reward))

    # print(stats)
    env.close()
    stats_np = np.asarray(stats)
    losses_np = np.asarray(losses)
    # plt.plot(stats_np[:,1], label="return")
    label="run"
    if ddqn:
        label+="_double"
    if exploration:
        label+="_priority"
    # plt.plot(running_mean(stats_np[:,1],100), label=label)
    # plt.legend()

    if save_model is not None:
        np.save('test.npy', stats)
        torch.save(q_agent.Q, save_model)

    print(q_agent.get_config())

def run(model, env_name="hockey"):
    if env_name == "hockey":
        env = h_env.HockeyEnv()
        action_map = {}
        for i in range(0,12):
            action_map[tuple(discrete_to_continous_action(i, env))] = i
        ac_space = spaces.Discrete(len(action_map))
        player2 = h_env.BasicOpponent(weak=False)
        obs_agent2 = env.obs_agent_two()
    else:
        env = gym.make(env_name, render_mode="rgb_array")
        if isinstance(env.action_space, spaces.Box):
            env = DiscreteActionWrapper(env,5)
        ac_space = env.action_space

    o_space = env.observation_space

    np.random.seed(0)

    q_agent = DQNAgent(o_space, ac_space, discount=0, eps=0,
                       use_hard_updates = False, tau = 0, update_target_every= 20, double=False, priority=False,
                        wandb = False, load_model = model, dueling = True)

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
