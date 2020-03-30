# -*- coding: utf-8 -*-
import rospy
import numpy as np
import tf

def saturate(val, limit):
    return min(max(val, -limit), limit) # Saturate in [-1, 1]


class Learn2Soar:

    def __init__(self):
        print("started")
        self.tf = tf.TransformerROS()
        self.roll_pitch_control_period = 0.01
        self.alt_control_period = 0.1
        
        self.alt = None
        self.alt_ref = 50
        self.last_alt_err = 0
        self.alt_integrator = 0
        self.theta_ref = 0
        
        self.last_wp_id = 0
        self.psi_integrator = 0       
        self.last_psi_err = 0
        self.phi_ref = 0

        self.prop = None
        self.elev = None
        self.ail = None

        self.theta = None
        self.phi = None


    def update_sensor_data(self, pose, twist):
        self.pose = pose
        self.twist = twist

        
    def do_alt_control(self):
        
        ## waypoint navigation

        wp_int = np.pi / 2 # waypoint interval
        traj_radius = 100

        x = self.pose.position.x
        y = self.pose.position.y
        ang_position = np.arctan2(y,x)

        # get location of next waypoint
        wp_id = np.floor(ang_position/wp_int) + 1
        wp_ang_position = wp_int*wp_id
        wp_x, wp_y = traj_radius*np.array([np.cos(wp_ang_position), np.sin(wp_ang_position)])

        ## altitude control
        if self.pose.position.x <= 100:
            self.alt_ref = 50
        else:
            self.alt_ref = 50

        alt = self.pose.position.z
        alt_err = (self.alt_ref - alt)

        K_p_alt = 0.10
        K_i_alt = 0.05
        K_d_alt = 0.0050

        theta_limit = 0.3*np.pi/2

        self.alt_integrator += alt_err * self.alt_control_period
        if K_i_alt > 0:
            self.alt_integrator = saturate(self.alt_integrator, theta_limit/K_i_alt)  # anti-windup

        alt_derivative = (alt_err - self.last_alt_err) / self.alt_control_period
        self.last_alt_err = alt_err

        self.theta_ref = - (K_p_alt * alt_err + K_i_alt * self.alt_integrator + K_d_alt * alt_derivative)
      
        self.theta_ref = saturate(self.theta_ref, theta_limit)  # limit pitch angle
        

        ## heading control

        psi_ref = np.arctan2(wp_y-y, wp_x-x)

        quat = (self.pose.orientation.x,
                self.pose.orientation.y,
                self.pose.orientation.z,
                self.pose.orientation.w)

        euler = tf.transformations.euler_from_quaternion(quat)
        psi     = euler[2]  # heading

        psi_err = np.mod((psi_ref - psi) + np.pi , 2*np.pi) - np.pi

        K_p_head = 1.5   #0.01
        K_i_head = 0.3   #0.05
        K_d_head = 0.2    #0.05
        phi_limit = 0.7*np.pi/2

        self.psi_integrator += psi_err*self.alt_control_period
        if K_i_head > 0:
            self.psi_integrator = saturate(self.psi_integrator, phi_limit/K_i_head)

        d_psi_err = (psi_err - self.last_psi_err) / self.alt_control_period
        self.last_psi_err = psi_err

        if self.last_wp_id != wp_id:
            d_psi_err = 0
            self.psi_integrator = 0

        phi_ref = -( K_p_head * psi_err + K_d_head * d_psi_err + K_i_head * self.psi_integrator)
        phi_ref = saturate(phi_ref,  phi_limit)

        self.phi_ref = phi_ref

        wp_dist = np.linalg.norm([wp_x-x, wp_y-y])
        radius = np.sqrt(x**2 + y**2)
        
        print('z: {:.1f}\trho: {:.1f}\twp: {:.1f}°\td: {:.1f}\tps_err: {:.1f}\tph_ref: {:.1f}\tth_ref: {:.1f}\t'.format(alt, radius, wp_ang_position*180/np.pi, wp_dist, psi_err, self.phi_ref, self.theta_ref))
        #print('z: {:.1f}\tph_ref: {:.1f}\tth_ref: {:.1f}\t'.format(alt, self.phi_ref, self.theta_ref))
        #print('elev: {:.2f}\tth: {:.2f}\tth_ref: {:.2f}\talt: {:.2f}'.format(self.elev, self.theta, self.theta_ref, alt))

    def do_roll_pitch_control(self):

        pose = self.pose
        quat = (pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quat)

        twist = self.twist
        R = self.tf.fromTranslationRotation((0,0,0), quat).T
        I_v = np.array([twist.linear.x, twist.linear.y, twist.linear.z, 1])
        airSpeed = np.dot(R, I_v)[0]
        if airSpeed < 0.1: 
            airSpeed = 0.1

            
        ## Roll controller

        K_p_r = 200
        ph_lim = 0.8*np.pi/2

        phi_ref = self.phi_ref
        phi     = euler[0]
        phi_err = phi_ref - phi

        ail_pos = 1 / airSpeed**2 * K_p_r * phi_err
        ail_pos = saturate(ail_pos, ph_lim)

        ## Pitch controller 

        K_p_p = 250
        elev_trim = -0.05
        th_lim = 1

        theta_ref = self.theta_ref 
        theta     = euler[1]
        theta_err = theta_ref - theta

        if np.abs( np.abs(phi) - np.pi/2 ) < 0.01:
            elev_pos = elev_trim  # can't control pitch with elevator when sideways
            print("SIDEWAYS!!")
        else:
            elev_pos = 1 / np.cos(phi) * ( 1 / airSpeed**2 * K_p_p * theta_err + elev_trim )

        elev_pos = saturate(elev_pos, th_lim)


        ## Speed controller

        prop = 10000


        ## Useful values to access throughout in the class 
        self.theta = theta
        self.phi = phi
        self.ail  = ail_pos
        self.elev = elev_pos
        self.prop = prop

        # command = [
        #   rudd_pos  
        #   elev_pos 
        #   ail_l_pos 
        #   ail_r_pos   
        #   flap_l1_pos 
        #   flap_l2_pos 
        #   flap_r1_pos 
        #   flap_r2_pos  
        #   prop_ref_0
        # ]

        command = [0, elev_pos, ail_pos, -ail_pos, 0, 0 , 0, 0, prop]

        return command