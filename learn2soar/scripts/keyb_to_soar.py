#!/usr/bin/env python3
import gym
import rotors_gym_envs.learn_to_albatross_env_v1
from getkey import getkey 
from itertools import product

keys = list("zxcasdqwe")
values = list(product([-1,0,+1],[-1,0,+1]))
keymap = dict(zip(keys, values))

env = gym.make('Albatross-v1')
while True:
    env.reset()
    ep_rew = 0.0
    done = False
    while not done:
        obs, rew, done, _ = env.step(keymap[getkey()])
        print("{0} {1}".format(obs,rew))
        ep_rew += rew
    print("Episode finished with cumulated reward = ", ep_rew)