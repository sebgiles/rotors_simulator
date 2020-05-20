#!/usr/bin/env python3
import numpy as np
import copy

import rospy
from teleop.msg import TwoTuple
from std_msgs.msg import Float32, Bool

import gym
from gym import spaces
import rotors_gym_envs.l2s_energy_rudd_env_v0
import time
        

class HumanInTheLoop():
    
    def __init__(self):
        self.action = [0.0, 0.0, 0.0]
        self.paused = True
        rospy.Subscriber("/teleop_mouse_cmd", TwoTuple,
                         self._mouse_cb, queue_size=1)
        rospy.Subscriber("/teleop_mouse_pressed", Bool,
                         self._click_cb, queue_size=1)
    

    def get_action(self):
        while self.paused:
            if rospy.is_shutdown(): 
                raise KeyboardInterrupt
            time.sleep(0.1)
        return self.action
            

    def _click_cb(self, msg):
        self.paused = not msg.data


    def _mouse_cb(self, msg):
        self.action = [msg.v, -msg.h, msg.h]

demo_name = 'eugi_energy'

def main():
    env = gym.make('l2s-energy-v0')
    agent = HumanInTheLoop()

    episode_starts = []
    actions = []
    observations = []
    rewards = []
    episode_returns = []

    ep_ret = 0.0
    obs    = None
    done   = True  # To force reset on first step

    while True:  
        if done: 
            if ep_ret >= 100.0:
                print("%.1f"%ep_ret)
                episode_returns += [ep_ret]
                episode_starts  += _episode_starts
                actions         += _actions
                observations    += _observations
                rewards         += _rewards
            else:
                print("You didn't reach rhe goal. Discarding episode...")
            _episode_starts = []
            _actions = []
            _observations = []
            _rewards = []
            ep_ret = 0.0
            while not agent.paused:
                time.sleep(0.1)
            obs = env.reset()
        try:
            act = agent.get_action()
        except KeyboardInterrupt:
            break

        _episode_starts.append(done)
        _observations.append(obs)
        _actions.append(act)

        obs, rew, done, _ = env.step(act)

        _rewards.append(rew)

        ep_ret += rew

    print("Demonstrations concluded.")
    print("Insert comment or just press enter. ['xxx' to discard the run]")
    comment = input()
    if comment=='xxx': 
        print('Discarding.')
        return

    if isinstance(env.observation_space, spaces.Box):
        observations = np.concatenate(observations).reshape((-1,) + env.observation_space.shape)
    elif isinstance(env.observation_space, spaces.Discrete):
        observations = np.array(observations).reshape((-1, 1))

    if isinstance(env.action_space, spaces.Box):
        actions = np.concatenate(actions).reshape((-1,) + env.action_space.shape)
    elif isinstance(env.action_space, spaces.Discrete):
        actions = np.array(actions).reshape((-1, 1))

    rewards         = np.array(rewards)
    episode_starts  = np.array(episode_starts)
    episode_returns = np.array(episode_returns)

    numpy_dict = {
        'actions': actions,
        'obs': observations,
        'rewards': rewards,
        'episode_returns': episode_returns,
        'episode_starts': episode_starts
    }  # type: Dict[str, np.ndarray]

    for key, val in numpy_dict.items():
        print(key, val.shape)  

    import os
    i = 0
    while os.path.exists("demonstrations/%s_run%03d.npz" % (demo_name, i)):
        i += 1
    filename = "demonstrations/%s_run%03d.npz"%(demo_name, i)
    print('Saving to ' + filename)
    np.savez(filename, **numpy_dict)

    if len(comment) > 0:
        with open("demonstrations/comments.txt", 'a+') as file:
            file.write("%s\t%s\n"%(filename, comment))

if __name__ == '__main__':
    main()
