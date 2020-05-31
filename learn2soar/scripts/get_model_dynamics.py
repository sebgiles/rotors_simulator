#!/usr/bin/env python3

import numpy as np
from scipy.spatial.transform import Rotation

class get_model_dynamics():

    def __init__(self, prev_observation, observation, action):

        self.time_step  = 0.3
        self.alpha      = 0.0
        self.beta       = 0.0
        self.m          = 2.55
        self.g          = 9.81

        self.prev_z     = prev_observation[0]
        self.prev_yaw   = prev_observation[2]
        self.prev_pitch = prev_observation[3]
        self.prev_roll  = prev_observation[4]
        self.z          = observation[0]
        self.airspeed   = observation[1]
        self.yaw        = observation[2]
        self.pitch      = observation[3]
        self.roll       = observation[4]
        self.d_e        = actions[0]
        self.d_al       = actions[1]
        self.d_ar       = - actions[1]
        self.d_r        = actions[2]

        self.rot = Rotation.from_euler('ZYX', [self.yaw, self.pitch, self.roll])

    def get_wind(self, z):
        wind = np.array([0,0,0], dtype=float)
        shear_top = 15.0
        wind_grad = -1.0
        wind[1] = wind_grad * max(min(z, shear_top), 0.0)
        return wind
    
    def get_ground_speed(self):

        I_v_wind = get_wind(self.z)
        B_v_wind = self.rot.apply(I_v_wind)

        B_v_air = np.zeros((B_v_wind.shape))
        B_v_air[0] = self.airspeed

        return B_v_air + B_v_wind

    def get_ground_angular_velocity(self):
        
        I_w = np.array([(self.yaw   - self.prev_yaw)   /self.time_step,
                        (self.pitch - self.prev_pitch) /self.time_step,
                        (self.roll  - self.prev_roll)  /self.time_step])

        return self.rot.apply(I_w)

    def get_net_cg_forces(self):

        B_F_a1 = np.array([[0.0, -0.009, -0.007],
                           [-0.003, 0.0, 0.08],
                           [0.0, -0.078, 0.0]])
    
        B_F_a2 = np.array([[-0.056, 0.026, -0.001, -0.001, 0.0, 0.0, -0.005],
                           [0.0, -0.296, 0.0, 0.0, 0.0, 0.034, 0.0],
                           [-1.892, 0.0, -0.017, -0.017, -0.027, 0.0, 0.038]])

        action_vec = np.array([self.alpha, self.beta, self.d_al, self.d_ar, self.d_e, self.d_r, 1.0])

        I_F_g = np.array([0.0, 0.0, self.m * self.g])

        B_w = get_ground_angular_velocity()

        B_F_g = self.rot.apply(I_F_g)
        B_F_1 = B_F_a1.dot(B_w) * self.airspeed
        B_F_2 = B_F_a2.dot(action_vec) * (self.airspeed**2)

        return B_F_1 + B_F_2 + B_F_g

    def get_get_next_ground_speed(self):

        B_v = get_ground_velocity()
        B_w = get_ground_angular_velocity()
        B_F = get_net_cg_forces()

        return B_v + self.time_step * (B_F / self.mass + np.cross(B_w,B_v))

    def get_next_altitude(self):
        
        v = get_next_ground_speed()

        return self.z + self.time_step * v[2]

    def get_next_airspeed(self):

        next_v_g = get_next_ground_speed()

        return np.linalg.norm(next_v_g - get_wind(self.z))

    def get_net_torque(self):

        B_M_a1 = np.array([[-0.386, 0.0, 0.039],
                           [0.0, -0.066, 0.0],
                           [0.008, 0.0, -0.046]])

        B_M_a2 = np.array([[0.0, -0.005, 0.017, -0.017, 0.0, 0.001, 0.0],
                           [-0.157, 0.0, 0.001, 0.001, -0.017, 0.0, 0.002],
                           [0.0, 0.16, -0.001, 0.001, 0.0, -0.018, 0.0]])

        action_vec = np.array([self.alpha, self.beta, self.d_al, self.d_ar, self.d_e, self.d_r, 1.0])

        B_w = get_ground_angular_velocity()

        return B_M_a1.dot(B_w) * self.airspeed + B_M_a2.dot(action_vec) + (self.airspeed**2)

    def get_next_pose(self):

        J = np.array([[0.6688, 0.0, 0.0281],
                      [0.0, 0.16235, 0.0],
                      [0.0281, 0.0, 0.69364]])
        
        B_M = get_net_torque()
        B_w = get_ground_angular_velocity()

        next_w = B_w + self.time_step * (np.linalg.inv(J).dot(B_M + np.cross(B_w, J.dot(B_w))))

        next_pose = np.array([self.yaw, self.pitch, self.roll]) + self.time_step * next_w

        return next_pose[0], next_pose[1], next_pose[2]