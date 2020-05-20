#!/usr/bin/env python3
import numpy as np
import copy

import rospy
from teleop.msg import TwoTuple
from std_msgs.msg import Float32, Bool

import gym
import rotors_gym_envs.l2s_energy_rudd_env_v0
# import rotors_gym_envs.learn_to_albatross_env_v0
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


def main():
    env = gym.make('l2s-energy-v0')
    agent = HumanInTheLoop()
    done = True
    while True:
        if done or agent.paused: env.reset()
        try:
            act = agent.get_action()
        except KeyboardInterrupt:
            print()
            exit()
        _, _, done, _ = env.step(act)


if __name__ == '__main__':
    main()
