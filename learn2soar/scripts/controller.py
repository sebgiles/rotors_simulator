# -*- coding: utf-8 -*-
import rospy
import numpy as np
from std_msgs.msg import Float32
import tf

def saturate(val, limit):
    return min(max(val, -limit), limit) # Saturate in [-1, 1]

floor_offset = 10.0

def get_wind(pos):
    z = pos.z - floor_offset
    wind = np.array([0,0,0, 0], dtype=float)
    shear_top = 15.0
    wind_grad = -1.0
    wind[1] = wind_grad * max(min(z, shear_top), 0.0)
    return wind

class Controller:

    def __init__(self):
        self.tf = tf.TransformerROS()
        self.low_level_control_period = 0.05
        self.high_level_control_period = 0.1
        
        self.last_psi_err = 0
        self.last_phi_err = 0
        self.last_theta_err = 0
        self.alt = None
        self.alt_ref = 50
        self.last_alt_err = 0
        self.alt_integrator = 0
        self.theta_ref = 0
        
        self.last_wp_id = 0
        self.psi_integrator = 0       
        self.last_psi = 0
        self.phi_ref = 0

        self.psi_ref = None

        self.last_curv = 0

        self.prop = 0
        self.elev = None
        self.ail = None
        self.rudd = None

        self.theta = None
        self.phi = None
        self.psi = None

        self.airSpeed = 0.01

        self.pose = None
        self.twist = None

    def update_sensor_data(self, pose, twist):
        self.pose = pose
        self.twist = twist


        
    def do_high_level_control(self):
        
        if self.pose is None or self.twist is None: return

        ## waypoint navigation

        wp_int = np.pi / 6. # waypoint interval
        traj_radius = 200

        x = self.pose.position.x
        y = self.pose.position.y
        ang_position = np.arctan2(y,x)

        x = self.pose.position.x
        y = self.pose.position.y

        # get location of next waypoint
        wp_id = np.floor(ang_position/wp_int) + 1
        wp_ang_position = wp_int*wp_id
        wp_x, wp_y = traj_radius*np.array([np.cos(wp_ang_position), np.sin(wp_ang_position)])
        d = np.linalg.norm([wp_x-x, wp_y-y])
        wp_dir = np.arctan2(wp_y-y, wp_x-x)

        pub_x = rospy.Publisher('glider_waypoint_x', Float32, queue_size=1)
        pub_y = rospy.Publisher('glider_waypoint_y', Float32, queue_size=1)
        pub_x.publish(Float32(wp_x))
        pub_y.publish(Float32(wp_y))

        self.alt_ref = 50.

        alt = self.pose.position.z

        alt_err = (self.alt_ref - alt)

        K_p_alt = 0.10
        K_i_alt = 0.05
        K_d_alt = 0.0050

        theta_limit = 0.5*np.pi/2

        self.alt_integrator += alt_err * self.alt_control_period
        if K_i_alt > 0:
            self.alt_integrator = saturate(self.alt_integrator, theta_limit/K_i_alt)  # anti-windup

        alt_derivative = (alt_err - self.last_alt_err) / self.alt_control_period
        self.last_alt_err = alt_err

        self.theta_ref = - (K_p_alt * alt_err + K_i_alt * self.alt_integrator + K_d_alt * alt_derivative)
      
        self.theta_ref = saturate(self.theta_ref, theta_limit)  # limit pitch angle
        

        ## heading control

        quat = (self.pose.orientation.x,
                self.pose.orientation.y,
                self.pose.orientation.z,
                self.pose.orientation.w)

        euler = tf.transformations.euler_from_quaternion(quat)

        psi = euler[2]  # heading

        psi_err = np.mod((wp_dir - psi) + np.pi , 2*np.pi) - np.pi

        if d > 15:
            curv = 2 * psi_err / d
            curv *= 0.8 # sideslip correction
        else:
            curv = self.last_curv
            
        self.last_curv = curv
        g = 9.81
        #K_p_head = 0.8
        if self.airSpeed > 0.01: 
            phi_limit = np.arccos (18 * g / self.airSpeed**2)
        else:
            phi_limit = 0.01

        #psi_dot_ref = K_p_head * psi_err 
        psi_dot_ref = self.airSpeed * curv
        
        phi_ref = - np.arctan2(psi_dot_ref * self.airSpeed, g)

        if np.abs(phi_ref) >= phi_limit:
            print("Banking at the limit")

        phi_ref = saturate(phi_ref,  phi_limit)



        self.phi_ref = phi_ref

        radius = np.sqrt(x**2 + y**2)
        
        print('v: {:.1f}\tpr: {:.1f}\tz: {:.1f}\trho: {:.1f}\td: {:.1f}\tps_err: {:.1f}\tph_err: {:.1f}\tth_ref: {:.1f}\tth: {:.1f}\t'\
              .format(self.airSpeed,self.prop, alt, radius, d, psi_err, self.last_phi_err, self.theta_ref, self.theta))
        #print('z: {:.1f}\tph_ref: {:.1f}\tth_ref: {:.1f}\t'.format(alt, self.phi_ref, self.theta_ref))
        #print('elev: {:.2f}\tth: {:.2f}\tth_ref: {:.2f}\talt: {:.2f}'.format(self.elev, self.theta, self.theta_ref, alt))

    def do_low_level_control(self, dt):

        if self.pose is None or self.twist is None: 
            print("control called too early")
            return False
        
        dt = dt.to_sec()

        if dt < 0.5*self.low_level_control_period:
            return False


        pose = self.pose
        quat = (pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quat)

        twist = self.twist
        R = self.tf.fromTranslationRotation((0,0,0), quat).T

        I_v = np.array([twist.linear.x, twist.linear.y, twist.linear.z, 1])

        wind = get_wind(self.pose.position)

        airSpeed = np.dot(R, I_v - wind)[0] 

        if airSpeed < 0.1: 
            airSpeed = 0.1

        ## Roll controller

        K_p_r = 400.
        K_d_r = 100.
        ail_lim = 0.5*np.pi/2

        phi_ref = self.phi_ref
        phi     = euler[0]
        phi_err = phi_ref - phi

        d_phi_err = (phi_err - self.last_phi_err ) / dt
        self.last_phi_err = phi_err
        ail_pos = 1 / airSpeed**2 * (K_p_r * phi_err + K_d_r * d_phi_err)
        ail_pos = saturate(ail_pos, ail_lim)


        ## Pitch controller 

        K_p_p = 100.
        K_d_p = 1.0
        elev_trim = -0.05
        elev_lim = 0.3*np.pi/2

        theta_ref = self.theta_ref 
        theta     = euler[1]
        theta_err = theta_ref - theta

        d_theta_err = (theta_err - self.last_theta_err ) / dt
        self.last_theta_err = theta_err

        if np.pi/2 - np.abs(phi) < 0.001:
            elev_pos = 0. 
        else:
            elev_pos =  1/np.cos(phi) * (elev_trim +
                            1/airSpeed**2 * (K_p_p * theta_err + K_d_p * d_theta_err))

        elev_pos = saturate(elev_pos, elev_lim)

        ## Yaw controller

        K_p_y = 50.
        K_d_y = 1.
        rudd_lim = 0.5*np.pi/2

        psi_ref = self.psi_ref 
        psi     = euler[2]
        psi_err = psi_ref - psi

        d_psi_err = (psi_err - self.last_psi_err ) / dt
        self.last_psi_err = psi_err
        rudd_pos = 1 / airSpeed**2 * (K_p_y * psi_err + K_d_y * d_psi_err)
        rudd_pos = saturate(rudd_pos, rudd_lim)

        ## Speed controller, very rough, upsgrade to PI if you care
        # k_p_v = 5000
        # v_ref = 18
        # prop_ff = 900
        # self.prop = prop_ff + k_p_v * (v_ref - airSpeed)
        # self.prop = max(0, min(10000, self.prop))

        ## Useful values to access throughout in the class 
        self.theta = theta
        self.phi = phi
        self.psi = psi
        self.airSpeed = airSpeed
        self.ail  = ail_pos
        self.elev = elev_pos
        self.rudd = rudd_pos
        self.alt = pose.position.z

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
        self.action = [ail_pos, elev_pos, rudd_pos, self.prop]
        self.command = [elev_pos, ail_pos, -ail_pos, rudd_pos, self.prop]

        return True
