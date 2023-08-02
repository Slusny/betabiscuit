import numpy as np
import laserhockey.hockey_env as h_env
import gymnasium as gym
from importlib import reload
import time
from utility import transform_obs
import matplotlib.pyplot as plt
# %matplotlib inline
from IPython import display
from datetime import datetime

from cpprb import ReplayBuffer

replay_buffer = ReplayBuffer(100,
                    {"obs": {"shape": (18)},
                    "act": {"shape": (4)},
                    "rew": {},
                    "next_obs": {"shape": (18)},
                    "done": {},
                    "info": {"shape": (3)}
                    }
                )

env = h_env.HockeyEnv()
player1 = h_env.HumanOpponent(env=env, player=1)
player2 = h_env.BasicOpponent()

obs, info = env.reset()

env.render()
time.sleep(1)
for j in range(100000):
    time.sleep(0.3)
    env.render()
    a1 = player1.act(obs) 
    obs_agent2 = env.obs_agent_two()
    a2 = player2.act(obs_agent2)
    obs_next, r, d, _, info = env.step(np.hstack([a1,a2]))   
    replay_buffer.add(obs=obs, act=a1, rew=r, next_obs=obs_next, done=d, info=[info["reward_closeness_to_puck"],info["reward_touch_puck"],info["reward_puck_direction"]])
    obs=obs_next
    if d or _: env.reset()
    if j % 200 == 0:
        print("End this? Press 'q' to quit")
        x = input()
        if x == 'q':
            break
date_str = datetime.today().strftime('%Y-%m-%dT%H.%M')
replay_buffer.save_transitions(f'recorded_dataset_{date_str}')