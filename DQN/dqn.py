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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

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
        # Compute Loss
        if weights is None:
            weights = torch.ones_like(pred)

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
        print("action space", action_space.n)
        self._action_n = action_space.n
        self._config = {
            "eps": 0.3,            # Epsilon in epsilon greedy policies
            "discount": 0.95,
            "buffer_size": int(1e5),
            "batch_size": 128,
            "learning_rate": 0.0002,
            "update_target_every": 20,
            "tau": 0.01,
            "use_hard_updates":True,
            "double":False,
            "priority": False,
            "wandb": False
        }
        self._config.update(userconfig)
        self._eps = self._config['eps']
        self.tau = self._config["tau"]


        if self._config['priority']:
            self.buffer = mem.PrioritizedReplayBuffer(self._config["buffer_size"])
        else:
            self.buffer = mem.Memory(max_size=self._config["buffer_size"])

        # Q Network
        self.Q = QFunction(observation_dim=self._observation_space.shape[0],
                           action_dim=self._action_n,
                           learning_rate = self._config["learning_rate"])


        # Q Network
        self.Q_target = QFunction(observation_dim=self._observation_space.shape[0],
                                  action_dim=self._action_n,
                                  learning_rate = 0)
        self._update_target_net()
        self.train_iter = 0

        if(self._config["wandb"]): # log gradients to W&B
            # start a new wandb run to track this script
            wandb.init(
                # set the wandb project where this run will be logged
                project="dqn-test",
            )
            wandb.watch(self.Q, log_freq=100)


    def _update_target_net(self):
        self.Q_target.load_state_dict(self.Q.state_dict())

    def soft_update_target_net(self):
        for target_param, param in zip(self.Q_target.parameters(), self.Q.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

    def act(self, observation, eps=None):
        if eps is None:
            eps = self._eps
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
                acts = torch.from_numpy(self.Q_target.greedyAction(s_prime))
                v_prime = self.Q.Q_value(torch.from_numpy(s_prime).float(), acts).cpu().detach().numpy()
            else:
                v_prime = self.Q_target.maxQ(s_prime)

            # target
            gamma=self._config['discount']
            td_target = rew + gamma * (1.0-done) * v_prime

            if isinstance(self.buffer, mem.PrioritizedReplayBuffer):

                pred = self.Q.Q_value(torch.from_numpy(s).float(), torch.from_numpy(a)).detach().numpy()
                # Compute TD error
                td_error = np.abs(pred - td_target)
                self.buffer.update_priorities(tree_idxs, td_error)

                fit_loss = self.Q.fit(s, a, td_target, weights = weights)

            else:
                # optimize the lsq objective
                fit_loss = self.Q.fit(s, a, td_target)

            if not self._config["use_hard_updates"]:
                self.soft_update_target_net()

            losses.append(fit_loss)

        return losses


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

def run(ddqn=False, guided=False, exploration=False):

    env_name = 'hockey'
    # env_name = 'CartPole-v0'

    if env_name == "hockey":
        env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)
        action_map = {}
        for i in range(0,12):
            action_map[tuple(discrete_to_continous_action(i, env))] = i
        ac_space = spaces.Discrete(len(action_map))
    else:
        env = gym.make(env_name)
        if isinstance(env.action_space, spaces.Box):
            env = DiscreteActionWrapper(env,5)
        ac_space = env.action_space

    o_space = env.observation_space
    print(ac_space)
    print(o_space)
    print(list(zip(env.observation_space.low, env.observation_space.high)))

    if ddqn:
        hard_updates= False
    else:
        hard_updates = True

    target_update = 20
    tau = 0.1

    q_agent = DQNAgent(o_space, ac_space, discount=0.95, eps=0.1,
                       use_hard_updates = hard_updates, tau = tau, update_target_every= target_update, double=True, priority=True,
                        wandb = True)

    ob,_info = env.reset()

    stats = []
    losses = []

    max_episodes=1000
    max_steps=60
    for i in range(max_episodes):
        # print("Starting a new episode")
        total_reward = 0
        ob, _info = env.reset()
        for t in range(max_steps):
            # if i > 2000:
            #     env.render()
            # done = False

            a = q_agent.act(ob)
            a_step = a
            if env_name == "hockey":
                # if np.random.random() > .1:

                a1 = env.discrete_to_continous_action(a)
                # else:
                #     pos = np.array((ob[0], ob[1]))
                #     puck = np.array((ob[12], ob[13]))
                #
                #     vel = puck - pos
                #
                #     if vel[0] >= .3:
                #         vel[0] = 1
                #     elif vel[0] <= -.3:
                #         vel[0] = -1
                #     else:
                #         vel[0] = 0
                #
                #     if vel[1] >= .3:
                #         vel[1] = 1
                #     elif vel[1] <= -.3:
                #         vel[1] = -1
                #     else:
                #         vel[1] = 0
                #
                #
                #     a1 = [vel[0], vel[1], 0, 0]
                #
                #     a = continuous_to_discrete_action(a1, action_map)

                    # print(continuous_to_discrete_action(a1))
                a2 = [0,0.,0,0]
                a_step = np.hstack([a1,a2])

            (ob_new, reward, done, trunc, _info) = env.step(a_step)

            if env_name == "hockey":
                obs_agent2 = env.obs_agent_two()

            total_reward+= reward
            q_agent.store_transition((ob, a, reward, ob_new, done, True))
            ob=ob_new
            if done: break
        loss = q_agent.train(32)
        losses.extend(loss)
        stats.append([i,total_reward,t+1])
        wandb.log({"loss": loss[len(loss)-1] , "reward": total_reward, "length":t })

        if ((i-1)%20==0):
            print("{}: Done after {} steps. Loss: {}".format(i, t+1, loss[len(loss)-1]))
            print("{}: Done after {} steps. Reward: {}".format(i, t+1, total_reward))

    print(stats)
    stats_np = np.asarray(stats)
    losses_np = np.asarray(losses)
    fig=plt.figure(figsize=(6,3.8))
    plt.plot(stats_np[:,1], label="return")
    plt.plot(running_mean(stats_np[:,1],200), label="smoothed-return")
    plt.legend()
    plt.show()

    print("hello")

    env.close()


def main():
    run(ddqn=True)


if __name__ == '__main__':
    main()
