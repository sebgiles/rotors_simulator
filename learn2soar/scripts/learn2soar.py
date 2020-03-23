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
        self.theta_ref = 0.25
        self.alt_integrator = 0

    def update_sensor_data(self, pose, twist):
        self.pose = pose
        self.twist = twist

    def do_alt_control(self):
        alt = self.pose.position.z
        alt_ref = 50

        K_p_alt = 0.2 
        K_i_alt = 0.001

        err = (alt_ref - alt)

        theta_limit = 0.5
        self.alt_integrator = self.alt_integrator + err
        self.alt_integrator = saturate(self.alt_integrator, theta_limit/K_i_alt)

        theta_ref = -( K_p_alt * err + K_i_alt * self.alt_integrator )

        self.theta_ref = saturate(theta_ref, theta_limit) # Saturate in [-1, 1]

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

        K_p_r = 100
        K_p_p = 100

        phi_ref   = +0.3
        theta_ref = self.theta_ref

        if airSpeed < 0.1: 
            ail_pos = 0
            elev_pos = 0
        else:
            ail_pos  = K_p_r / (airSpeed**2) * (phi_ref - phi)
            elev_pos = K_p_p / (airSpeed**2) * (theta_ref - theta)

        limit = 1

        ail_pos = min(max(ail_pos, -limit), limit) # Saturate in [-1, 1]
        elev_pos = min(max(elev_pos, -limit), limit) # Saturate in [-1, 1]


        print('z: \t{:.2f}\tv: \t{:.2f}\tr: \t{:.2f}\ta: \t{:.2f}\t\tp: \t{:.2f}\te: \t{:.2f}'.format(z, airSpeed, phi, ail_pos, theta, elev_pos))

        prop = 10000



# rudd_pos  elev_pos ail_l_pos ail_r_pos   flap_l1_pos flap_l2_pos flap_r1_pos flap_r2_pos  prop_ref_0


        command = [0, elev_pos, ail_pos, -ail_pos, 0, 0 ,0 , 0, prop]

        return command