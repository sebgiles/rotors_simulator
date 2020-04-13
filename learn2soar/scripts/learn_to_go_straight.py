#!/usr/bin/env python3
import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import A2C

import rotors_gym_envs.learn_to_soar_env_v1

from rospy.exceptions import ROSInterruptException
from rospy.service    import ServiceException
env = make_vec_env('LearnToSoar-v1')

#model = A2C(MlpPolicy, env, tensorboard_log="./l2s/")
model = A2C.load("a2c_soar_autosave", env, tensorboard_log="./l2s/")
model.verbose = 1
try:
    model.learn(total_timesteps=1000000)
except (ROSInterruptException, ServiceException):
    print("Interrupted, Puasing, saving model")
    model.save("a2c_soar_autosave")
    exit()

model.save("a2c_soar_autosave")
