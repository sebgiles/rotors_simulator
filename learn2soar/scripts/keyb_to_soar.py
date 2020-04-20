#!/usr/bin/env python3
import gym
import rotors_gym_envs.learn_to_soar_env_v3
from getkey import getkey 
from itertools import product

keys = list("zxcasdqwe")
values = list(product([-1,0,+1],[-1,0,+1]))
keymap = dict(zip(keys, values))

env = gym.make('LearnToSoar-v3')

while True:
    env.reset()
    done = False
    while not done:
        _, _, done, _ = env.step(keymap[getkey()])