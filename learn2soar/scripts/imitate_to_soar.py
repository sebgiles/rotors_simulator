#!/usr/bin/env python3
import numpy as np
import copy

import rospy
from teleop.msg import TwoTuple
from std_msgs.msg import Float32, Bool

from stable_baselines.common import make_vec_env
from stable_baselines.gail import ExpertDataset
from stable_baselines import A2C, PPO2
from stable_baselines.common.policies import FeedForwardPolicy, MlpPolicy, LstmPolicy

import gym
from gym import spaces
import rotors_gym_envs.learn_to_albatross_env_v0
import time

demo_name = 'demonstrations/seb_run001.npz'

class CustomLSTMPolicy(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=64, reuse=False, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                         net_arch=[ 32, 'lstm', dict(vf=[32,16], pi=[32,16]) ],
                         layer_norm=True, feature_extraction="mlp", **_kwargs)

class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
            #net_arch=[ 32, dict(vf=[32,16], pi=[32,16]) ],
            net_arch=[ dict(vf=[32,16], pi=[6,4]) ],
            feature_extraction="mlp")


def main():

    env = make_vec_env('Albatross-v0')
    # Using only one expert trajectory
    # you can specify `traj_limitation=-1` for using the whole dataset
    dataset = ExpertDataset(expert_path=demo_name, verbose=1,
                            traj_limitation=-1, batch_size=5, train_fraction=0.8)

    model = A2C(CustomLSTMPolicy,           # 1 
        env, 
        verbose = 1, 
        #learning_rate=5e-4, 
        #gamma=0.995, 
        #n_steps=5, 
        #lr_schedule='linear' 
    )


    #model = PPO2(CustomLSTMPolicy, env, verbose=1)
    # Pretrain the PPO2 model
    model.pretrain(dataset, n_epochs=2000, )

    # As an option, you can train the RL agent
    # model.learn(int(1e5))

    while not rospy.is_shutdown():
        obs = env.reset()
        ep_rew = 0
        done = False
        while not done and not rospy.is_shutdown():
            action, _states = model.predict(obs)
            try:
                obs, rew, done, _ = env.step(action)
            except:
                exit()
            ep_rew += rew
        print("Episode reward = %f"%(ep_rew))


if __name__ == '__main__':
    main()
