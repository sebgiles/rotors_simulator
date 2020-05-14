#!/usr/bin/env python3
import numpy as np
import copy

import rospy
from teleop.msg import TwoTuple
from std_msgs.msg import Float32, Bool

import gym
import rotors_gym_envs.learn_to_albatross_env_v0
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
            time.sleep(.1)

        z, v, yaw = obs

        roll = 1.0*self.mouse_cmd[1]

        if z < 0.0:
            if roll > .0:
                pitch = .0
            elif yaw < .0:
                pitch = 1.0*roll
            else:
                pitch = -1.0*yaw
        elif z < 15.0:
            if yaw > .0:
                pitch = -1.0*yaw
            else:
                pitch = -1.0*yaw
        elif z > 15.0:
            if yaw > .0:
                pitch = .0
            else:
                pitch = -1.0*yaw

        shear_top = 15.0
        wind_grad = -1.0
        wind = wind_grad * max(min(z, shear_top), 0.0)

        airspeed = v + np.sin(yaw) * wind
        if airspeed < 15:
            pitch = min(0.0, pitch)

        return [pitch, roll]
            

    def _click_cb(self, msg):
        self.paused = not msg.data


    def _mouse_cb(self, msg):
        self.mouse_cmd = [msg.v, -msg.h]


def main():
    env = gym.make('Albatross-v0')
    agent = HumanInTheLoop()
    done = True
    ep_rew = 0
    obs = None
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
