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
        self.des_alt_ref = 55
        self.theta_ref = 0
        self.alt_integrator = 0
        self.alt_err_old = 0

    def update_sensor_data(self, pose, twist):
        self.pose = pose
        self.twist = twist

    def do_alt_control(self):
        alt = self.pose.position.z

        if self.pose.position.x <= 200:
            
            self.des_alt_ref = 50
        else:
            self.des_alt_ref = 60
        
        alt_ref = self.des_alt_ref

        # PID gains
        K_p_alt = 0.1
        K_i_alt = 0.005
        K_d_alt = 0.5

        err = (alt_ref - alt)

        theta_limit = 0.3 * np.pi/2
 
        self.alt_integrator = self.alt_integrator + err
        alt_derivative = err - self.alt_err_old

        self.theta_ref = - (K_p_alt * err + K_i_alt * self.alt_integrator + K_d_alt * alt_derivative)
        self.theta_ref = saturate(self.theta_ref, theta_limit)

        # update old error e[k-1]
        self.alt_err_old = err

    def do_roll_pitch_control(self):

        pose = self.pose
        twist = self.twist
        x, y, z = pose.position.x, pose.position.y, pose.position.z
        quat = (pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w)

        I_v = np.array([twist.linear.x, twist.linear.y, twist.linear.z, 1])
        R = self.tf.fromTranslationRotation((0,0,0), quat).T

        airSpeed = (R.dot(I_v))[0]
        euler = tf.transformations.euler_from_quaternion(quat)
        phi     = euler[0]
        theta   = euler[1]
        psi     = euler[2]

        K_p_r = 500
        K_p_p = 250
        alt_trim = -0.05

        phi_ref   = 0
        theta_ref = self.theta_ref

        phi_err = phi_ref - phi
        theta_err = theta_ref - theta

        if airSpeed < 0.1: 
            ail_pos = 0
            elev_pos = 0
        else:
            ail_pos  = K_p_r / (airSpeed**2) * phi_err
            elev_pos = (1 / (airSpeed**2)) * (K_p_p  * theta_err) + alt_trim

        limit = 1

        ail_pos = min(max(ail_pos, -limit), limit)
        elev_pos = min(max(elev_pos, -limit), limit)

        des_alt = self.des_alt_ref

        print('x: {:.2f}\tz: {:.2f}\talt: {:.2f}'.format(x, z, des_alt))

        prop = 10000



# rudd_pos  elev_pos ail_l_pos ail_r_pos   flap_l1_pos flap_l2_pos flap_r1_pos flap_r2_pos  prop_ref_0

        command = [0, elev_pos, ail_pos, -ail_pos, 0, 0 ,0 , 0, prop]

        return command