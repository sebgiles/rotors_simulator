#!/usr/bin/env python3

import numpy as np
from scipy.spatial.transform import Rotation

shear_top = 15.0
wind_grad = -1.0
time_step  = 0.3
mass       = 2.55
g          = 9.81

I_F_g = np.array([0.0, 0.0, mass * g])

B_F_a1_T = np.array([[0.0, -0.009, -0.007],
                    [-0.003, 0.0, 0.08],
                    [0.0, -0.078, 0.0]]).T

B_F_a2_T = np.array([[-0.056, 0.026, -0.001, -0.001, 0.0, 0.0, -0.005],
                    [0.0, -0.296, 0.0, 0.0, 0.0, 0.034, 0.0],
                    [-1.892, 0.0, -0.017, -0.017, -0.027, 0.0, 0.038]]).T

B_M_a1 = np.array([[-0.386, 0.0, 0.039],
                    [0.0, -0.066, 0.0],
                    [0.008, 0.0, -0.046]])

B_M_a2 = np.array([[0.0, -0.005, 0.017, -0.017, 0.0, 0.001, 0.0],
                    [-0.157, 0.0, 0.001, 0.001, -0.017, 0.0, 0.002],
                    [0.0, 0.16, -0.001, 0.001, 0.0, -0.018, 0.0]])


class get_model_dynamics():

    def __init__(self, prev_observation, observation, action):

        self.batch_size = action.shape[0]

        self.prev_yaw   = prev_observation[:,2]
        self.prev_pitch = prev_observation[:,3]
        self.prev_roll  = prev_observation[:,4]
        self.z          = observation[:,0]
        self.airspeed   = observation[:,1]
        self.yaw        = observation[:,2]
        self.pitch      = observation[:,3]
        self.roll       = observation[:,4]

        alpha= 0.0
        beta = 0.0
        d_e  = action[:,0]
        d_al = action[:,1]
        d_ar = - action[:,1]
        d_r  = action[:,2]

        ones = np.ones([self.batch_size])

        self.action_vec = np.stack(
            [ones*alpha, ones*beta, d_al, d_ar, d_e, d_r, ones], axis=1)

        self.rot = [
            Rotation.from_euler('ZYX', [self.yaw[i], self.pitch[i], self.roll[i]]) 
            for i in range(self.batch_size)]


    def get_wind(self, z):
        wind = np.zeros([z.shape[0], 3], dtype=float)
        wind[:, 1] = wind_grad * np.clip(z, 0.0, shear_top)
        return wind
    

    def get_ground_speed(self):

        I_v_wind = self.get_wind(self.z)
        B_v_wind = np.zeros([self.batch_size, 3])
        for i in range(self.batch_size):
            B_v_wind[i] = self.rot[i].apply(I_v_wind[i])

        B_v_air = np.zeros([self.batch_size, 3])
        B_v_air[:,0] = self.airspeed

        return B_v_air + B_v_wind


    def get_ground_angular_velocity(self):

        I_w = np.stack([(self.yaw   - self.prev_yaw),
                         (self.pitch - self.prev_pitch),
                         (self.roll  - self.prev_roll) 
                        ],axis=1) / time_step
        w = np.zeros([self.batch_size, 3])
        for i in range(self.batch_size):
            w[i] = self.rot[i].apply(I_w[i])

        return w


    def get_net_cg_forces(self):

        B_w = self.get_ground_angular_velocity()
        B_F_g = np.zeros([self.batch_size, 3])

        for i in range(self.batch_size):
            B_F_g[i] = self.rot[i].apply(I_F_g)

        B_F_1 = B_w.dot(B_F_a1_T) * self.airspeed[:, np.newaxis]
        B_F_2 = self.action_vec.dot(B_F_a2_T) * (self.airspeed[:, np.newaxis]**2)

        return B_F_1 + B_F_2 + B_F_g


    def get_next_ground_speed(self):
        B_v = self.get_ground_speed()
        B_w = self.get_ground_angular_velocity()
        B_F = self.get_net_cg_forces()
        return B_v + time_step * (B_F / mass + np.cross(B_w,B_v))


    def get_next_altitude(self):
        v = self.get_next_ground_speed()
        return self.z + time_step * v[:,2]


    def get_next_airspeed(self):
        next_v_g = self.get_next_ground_speed()
        return np.linalg.norm(next_v_g - self.get_wind(self.z), axis=1)


    def get_next_pose(self):
        B_w = self.get_ground_angular_velocity()
        next_pose = np.stack([self.yaw, self.pitch, self.roll],axis=1) + time_step * B_w
        return next_pose


    def get_pred_obs(self):
        z = self.get_next_altitude()
        v = self.get_next_airspeed()
        pose = self.get_next_pose()
        return np.hstack([z[:, np.newaxis], v[:, np.newaxis], pose])


    def get_pred_rew(self):
        return None


    def get_pred_done(self):
        return None

