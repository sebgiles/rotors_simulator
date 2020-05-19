#!/usr/bin/env python3
import numpy as np
import copy

import rospkg 

from stable_baselines.common import make_vec_env
from stable_baselines.gail import ExpertDataset
from stable_baselines import A2C, PPO2
from stable_baselines.common.policies import FeedForwardPolicy, MlpPolicy, LstmPolicy

import gym
from gym import spaces
import rotors_gym_envs.l2s_energy_env_v0
import time

demo_name = 'demonstrations/seb_run006.npz'
l2s_path = rospkg.RosPack().get_path('learn2soar') + "/"

class CustomLSTMPolicy(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=64, reuse=False, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                         net_arch=[ 32, 'lstm', dict(vf=[32,16], pi=[32,16]) ],
                         layer_norm=True, feature_extraction="mlp", **_kwargs)

class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
            #net_arch=[ 32, dict(vf=[32,16], pi=[32,16]) ],
            net_arch=[ dict(vf=[32,16], pi=[8]) ],
            feature_extraction="mlp")


def main():

    env = make_vec_env('l2s-energy-v0')
    # Using only one expert trajectory
    # you can specify `traj_limitation=-1` for using the whole dataset
    dataset = ExpertDataset(expert_path=demo_name, verbose=1,
                            traj_limitation=-1, batch_size=64, train_fraction=0.9)

    model = A2C(CustomPolicy,           # 1 
        env, 
        verbose = 1,
        #learning_rate=5e-4,
        #gamma=0.995,
        #n_steps=5,
        #lr_schedule='linear'
    )


    #model = PPO2(CustomPolicy, env, verbose=1)
    # Pretrain the PPO2 model
    model.pretrain(dataset, n_epochs=3000, )
    model_filename = l2s_path + "trained_models/energy_v0.2"

    model.save(model_filename)

    # As an option, you can train the RL agent
    # model.learn(int(1e5))

    while True:
        obs = env.reset()
        ep_rew = .0
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, rew, done, _ = env.step(action)
            ep_rew += rew
        print("Episode reward = %f"%(ep_rew))


if __name__ == '__main__':
    main()
