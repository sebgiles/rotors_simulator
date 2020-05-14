#!/usr/bin/env python3
import numpy as np
import copy

import rospy
from teleop.msg import TwoTuple
from std_msgs.msg import Float32, Bool

import gym
from gym import spaces
import rotors_gym_envs.l2s_energy_env_v1
import time
        

class HumanInTheLoop():
    
    def __init__(self):
        self.mouse_cmd = TwoTuple()
        self.paused = True
        rospy.Subscriber("/teleop_mouse_cmd", TwoTuple,
                         self._mouse_cb, queue_size=1)
        rospy.Subscriber("/teleop_mouse_pressed", Bool,
                         self._click_cb, queue_size=1)
    

    def get_action(self, obs):
        while self.paused:
            if rospy.is_shutdown(): 
                raise KeyboardInterrupt
            time.sleep(0.1)
        airspeed_factor = 1 / (max(5,obs[1])/15.0)**2
        airspeed_factor = min(max(airspeed_factor,-1),1)
        act = np.array([self.mouse_cmd.v, -self.mouse_cmd.h]) * airspeed_factor
        return act

    def __call__(self, obs): 
        return self.get_action(obs)


    def _click_cb(self, msg):
        self.paused = not msg.data


    def _mouse_cb(self, msg):
        self.mouse_cmd = msg

demo_name = 'seb'

def main():
    env = gym.make('l2s-energy-v1')
    agent = HumanInTheLoop()

    episode_starts = []
    actions = []
    observations = []
    rewards = []
    episode_returns = []

    ep_ret = None
    obs    = None
    info   = None
    done   = True  # To force reset on first step


    while True: 
        if done: 

            _episode_starts = []
            _actions = []
            _observations = []
            _rewards = []
            ep_ret = 0.0
            obs = env.reset()
            ep_steps = 0

            while not agent.paused:
                time.sleep(0.1)   

        try:
            act = agent(obs)
        except KeyboardInterrupt:
            break

        _episode_starts.append(done)
        _observations.append(obs)
        _actions.append(act)

        obs, rew, done, info = env.step(act)
        ep_steps += 1 
        
        # if ep_steps >= 60: 
        #     done = True
        if agent.paused: 
            done = True


        _rewards.append(rew)
        ep_ret += rew

        if done or ep_steps%6 == 0:
            print("\r", end='')
            print("Reward: %04.0f\t"%ep_ret, end='')
            print("min_airspeed: %04.1f\t"%info['min_airspeed'], end='')

        if done: 
            if agent.paused: 
                print(' - quit', end='')
            if ep_ret >= 0.0 :
                episode_returns += [ep_ret]
                episode_starts  += _episode_starts
                actions         += _actions
                observations    += _observations
                rewards         += _rewards
                print(" - KEEP", end='')
            print()


    if len(episode_returns) < 1: return

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
