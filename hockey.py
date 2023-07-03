import numpy as np
import laserhockey.hockey_env as h_env
import gymnasium as gym
from importlib import reload
import time

def normal_play():
    env = h_env.HockeyEnv()
    print(env.observation_space)
    print(env.action_space)

    obs, info = env.reset()
    obs_agent2 = env.obs_agent_two()
    _ = env.render()

    obs, info = env.reset()
    obs_agent2 = env.obs_agent_two()

    for _ in range(600):
        env.render(mode="human")
        a1 = np.random.uniform(-1,1,4)
        a2 = np.random.uniform(-1,1,4)
        obs, r, d, t, info = env.step(np.hstack([a1,a2]))
        obs_agent2 = env.obs_agent_two()
        if d: break
    env.close()

    return info

# shooting with guidance
def shooting():
    env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)
    o, info = env.reset()
    _ = env.render()

    for i in range(500):
        env.render()

        a1 = [0,0,0,1]
        if i > 0:
            pos = np.array((obs[0], obs[1]))
            puck = np.array((obs[12], obs[13]))
            vel = puck-pos/(np.sqrt((puck[0]-pos[0])**2 + (puck[1] - pos[1])**2))
            print(vel)
            a1 = [vel[0], vel[1], 0, 0]
            # print(a1)

        # a1 = [0,0,-.1,1] # np.random.uniform(-1,1,4)
        a2 = [0,0.,0,0]
        obs, r, d, _ , info = env.step(np.hstack([a1,a2]))

        obs_agent2 = env.obs_agent_two()
        if d: break

    env.close()

    return info

def defending():
    env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE)
    o, info = env.reset()
    _ = env.render()

    for _ in range(600):
        env.render()
        print(env)
        a1 = [0,0,.1,1] # np.random.uniform(-1,1,3)
        a2 = [0,0.,0,1]
        obs, r, d,_, info = env.step(np.hstack([a1,a2]))
        print(r)
        obs_agent2 = env.obs_agent_two()
        if d: break
    env.close()

    return info

print(shooting())
