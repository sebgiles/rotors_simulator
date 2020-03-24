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
        self.theta_ref = 0
        self.phi_ref = 0
        self.alt_integrator = 0
        self.psi_integrator = 0       
        self.last_psi_err = 0
        self.last_wp_id = 0

    def update_sensor_data(self, pose, twist):
        self.pose = pose
        self.twist = twist

    def do_alt_control(self):
        
        ## altitude control

        alt = self.pose.position.z
        alt_ref = 50

        K_p_alt = 0.05 
        K_i_alt = 0.001

        alt_err = (alt_ref - alt)

        self.alt_integrator = self.alt_integrator + alt_err
        self.alt_integrator = saturate(self.alt_integrator, np.pi/2/K_i_alt)

        theta_ref = -( K_p_alt * alt_err + K_i_alt * self.alt_integrator )

        self.theta_ref = saturate(theta_ref, 0.8*np.pi/2) # limit pitch angle


        ## waypoint navigation

        wp_int = 0.25*np.pi  # waypoint interval
        traj_radius = 50

        x = self.pose.position.x
        y = self.pose.position.y
        ang_position = np.arctan2(y,x)

        # get location of next waypoint
        wp_id = np.floor(ang_position/wp_int) + 1
        wp_ang_position = wp_int*wp_id
        wp_x, wp_y = traj_radius*np.array([np.cos(wp_ang_position), np.sin(wp_ang_position)])

        psi_ref = np.arctan2(wp_y-y, wp_x-x)

        ## heading control

        quat = (pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w)

        euler = tf.transformations.euler_from_quaternion(quat)
        psi     = euler[2]  # heading

        psi_err = np.mod((psi_ref - psi) + np.pi , 2*np.pi) - np.pi

        self.psi_integrator += psi_err*self.alt_control_period
        self.psi_integrator = saturate(self.psi_integrator, np.pi/2/K_i_head)

        d_psi_err = (psi_err - self.last_psi_err) / self.alt_control_period
        self.last_psi_err = psi_err

        K_p_head = 1.0
        K_i_head = 0.05
        K_d_head = 1.0

        if self.last_wp_id != wp_id:
            d_psi_err = 0
            self.psi_integrator = 0

        phi_ref = -( K_p_head * psi_err + K_d_head * d_psi_err + K_i_head * self.psi_integrator)

        phi_ref = saturate(phi_ref,  0.8*np.pi/2)

        self.phi_ref = phi_ref

        z = self.pose.position.z
        wp_dist = np.linalg.norm([wp_x-x, wp_y-y])
        radius = np.sqrt(x**2 + y**2)

        #print('z: {:.1f}\tpsi_err: {:.2f}\tphi_ref: {:.2f}'.format(z, psi_err, phi_ref))
        print('z: {:.1f}\trho: {:.1f}\twp: {:.1f}\tdist: {:.1f}\tpsi_err: {:.1f}\t'
                .format(z, radius, wp_ang_position, wp_dist, psi_err))


    def do_roll_pitch_control(self):
        pose = self.pose
        twist = self.twist
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

        K_p_r = 200
        K_p_p = 100
        elev_trim = 0.05

        phi_ref   = self.phi_ref
        theta_ref = self.theta_ref 


        if airSpeed < 0.1: 
            ail_pos = 0
            elev_pos = 0
        else:
            ail_pos  = K_p_r /               (airSpeed**2) * (  phi_ref -   phi)
            elev_pos = K_p_p / (np.cos(phi) * airSpeed**2) * (theta_ref - theta) + elev_trim 

        limit = 1

        ail_pos = min(max(ail_pos, -limit), limit) # Saturate in [-1, 1]
        elev_pos = min(max(elev_pos, -limit), limit) # Saturate in [-1, 1]

        #z = self.pose.position.z

        #print('z: \t{:.2f}\tv: \t{:.2f}\tr: \t{:.2f}\ta: \t{:.2f}\t\tp: \t{:.2f}\te: \t{:.2f}'.format(z, airSpeed, phi, ail_pos, theta, elev_pos))

        prop = 100000

        # rudd_pos  
        # elev_pos 
        # ail_l_pos 
        # ail_r_pos   
        # flap_l1_pos 
        # flap_l2_pos 
        # flap_r1_pos 
        # flap_r2_pos  
        # prop_ref_0

        command = [0, elev_pos, ail_pos, -ail_pos, 0, 0 ,0 , 0, prop]

        return command