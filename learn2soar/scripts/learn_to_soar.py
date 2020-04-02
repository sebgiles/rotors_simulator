#!/usr/bin/env python
import gym
from gym import wrappers
import time
import numpy
import random
import time

import rotors_gym_envs.learn_to_soar_env_v0

# contains our contoller, using it as a policy to test environment
from controller import Controller


def main():
    env = gym.make('LearnToSoar-v0')

    cntr = Controller()
    cntr.roll_pitch_control_period = 0.05
    total_episodes = 10
    episode_duration = 4.0 #seconds
    steps_per_episode = int(episode_duration/env.time_step)

    for x in range(total_episodes):

        observation, done = env.reset()
        print('==== EPISODE ' + str(x) +' ====')
        for i in range(steps_per_episode):
            if done: break
            cntr.update_sensor_data(observation[0], observation[1])
            cntr.do_low_level_control()
            action = cntr.action
            observation, reward, done, info = env.step(action)
        print("Final state {}".format(cntr.phi))

if __name__ == '__main__':
    
    main()

