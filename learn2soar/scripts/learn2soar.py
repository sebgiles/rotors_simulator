import rospy
import numpy as np
import tf


class Learn2Soar:
    def __init__(self):
        print("started")
        self.tf_ = tf.TransformerROS()

    def doControl(self, pose, twist):
        x, y, z = pose.position.x, pose.position.y, pose.position.z
        quat = (pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w)

        I_v = np.array([twist.linear.x, twist.linear.y, twist.linear.z, 1])
        R = self.tf_.fromTranslationRotation((0,0,0), quat)

        airSpeed = (R.dot(I_v))[0]
        euler = tf.transformations.euler_from_quaternion(quat)
        phi     = euler[0]
        theta   = euler[1]
        psi     = euler[2]

        K_p_r = 100
        K_p_p = 100

        phi_ref   = - 0
        theta_ref = - 0.1

        if airSpeed < 0.1: 
            ail_pos = 0
            elev_pos = 0
        else:
            ail_pos  = K_p_r / (airSpeed**2) * (phi_ref - phi)
            elev_pos = K_p_p / (airSpeed**2) * (theta_ref - theta)

        limit = 2

        ail_pos = min(max(ail_pos, -limit), limit) # Saturate in [-1, 1]
        elev_pos = min(max(elev_pos, -limit), limit) # Saturate in [-1, 1]


        print('r: {:.2f}\ta: {:.2f}\t\tp: {:.2f}\te: {:.2f}'.format(phi, ail_pos, theta, elev_pos))

        prop = 0



# rudd_pos  elev_pos ail_l_pos ail_r_pos   flap_l1_pos flap_l2_pos flap_r1_pos flap_r2_pos  prop_ref_0


        command = [0, elev_pos, ail_pos, -ail_pos, 0, 0 ,0 , 0, prop]

        return command