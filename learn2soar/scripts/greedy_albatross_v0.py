#!/usr/bin/env python3

from stable_baselines import A2C
from stable_baselines.common import make_vec_env
import rospkg 

import rotors_gym_envs.learn_to_albatross_env_v0

env = make_vec_env('Albatross-v0')

l2s_path = rospkg.RosPack().get_path('learn2soar') + "/"

model_filename = l2s_path + "trained_models/albatross_v0.3"

model = A2C.load(model_filename)

while True:
    obs = env.reset()
    ep_rew = 0
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rew, done, _ = env.step(action)
        ep_rew += rew
    print("Episode reward = %f"%(ep_rew))
