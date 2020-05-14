#!/usr/bin/env python3
import numpy as np
import copy

import rospy
from teleop.msg import TwoTuple
from std_msgs.msg import Float32, Bool

import gym
import rotors_gym_envs.l2s_energy_env_v1
import time
class HumanInTheLoop():
    
    def __init__(self):
        self.mouse_cmd = [0.0, 0.0]
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
        pitch, roll = self.mouse_cmd
        action = [pitch, roll]
        return action
            

    def _click_cb(self, msg):
        self.paused = not msg.data


    def _mouse_cb(self, msg):
        self.mouse_cmd = [msg.v, -msg.h]


def main():
    env = gym.make('l2s-energy-v1')
    agent = HumanInTheLoop()
    done = True
    ep_rew = 0
    while True:  
        if done or agent.paused: 
            print(ep_rew)
            obs = env.reset()
            ep_rew = 0
        try:
            act = agent.get_action(obs)
        except KeyboardInterrupt:
            print()
            exit()
        obs, rew, done, _ = env.step(act)
        print(obs)
        ep_rew += rew
        


if __name__ == '__main__':
    main()
