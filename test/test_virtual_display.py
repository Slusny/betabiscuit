import numpy as np
import laserhockey.hockey_env as h_env
import gymnasium as gym
from importlib import reload
import time
import matplotlib.pyplot as plt
# %matplotlib inline
from IPython import display


env = h_env.HockeyEnv()
player1 = h_env.BasicOpponent()
player2 = h_env.BasicOpponent()
obs_buffer = []
reward_buffer=[]
obs, info = env.reset()
obs_agent2 = env.obs_agent_two()
img = plt.imshow(env.render(mode='rgb_array'))
for k in range(100):
    img.set_data(env.render(mode='rgb_array'))
    display.display(plt.gcf())
    display.clear_output(wait=True)
    a1 = player1.act(obs)
    a2 = player2.act(obs_agent2)
    obs, r, d, _, info = env.step(np.hstack([a1,a2]))    
    obs_buffer.append(obs)
    reward_buffer.append(r)
    obs_agent2 = env.obs_agent_two()
    if d: env.reset()
obs_buffer = np.asarray(obs_buffer)
reward_buffer = np.asarray(reward_buffer)
env.close()