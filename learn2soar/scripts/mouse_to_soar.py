#!/usr/bin/env python3
import numpy as np
import copy

import rospy
from teleop.msg import TwoTuple
from std_msgs.msg import Float32, Bool
from gazebo_msgs.msg import ModelState
from std_srvs.srv import Empty


cs_topics = [   "/uav_1/elev_pos",      
                "/uav_1/ail_l_pos",     
                "/uav_1/ail_r_pos",      
                "/uav_1/prop_ref_0"
              ]

class FWMouseControl():
    
    cs_pubs = []

    def __init__(self):

        rospy.Subscriber("/teleop_mouse_cmd", TwoTuple,
                         self._mouse_cb, queue_size=1)

        rospy.Subscriber("/teleop_mouse_pressed", Bool,
                         self._click_cb, queue_size=1)


        self.roll_pub  = rospy.Publisher("/l2s/attitude_cmd/roll",  Float32, queue_size=1)
        self.pitch_pub = rospy.Publisher("/l2s/attitude_cmd/pitch", Float32, queue_size=1)
        
        for topic in cs_topics:
            self.cs_pubs.append(rospy.Publisher(topic, Float32, queue_size=1))

        self.state_pub = rospy.Publisher('/gazebo/set_model_state', ModelState,
                                         queue_size=1)
                                        
        self.init_state = ModelState()
        self.init_state.model_name = 'uav_1'
        self.init_state.reference_frame = 'ground_collision_plane'

        rospy.wait_for_service('/gazebo/pause_physics')    
        rospy.wait_for_service('/gazebo/unpause_physics')
        self._unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self._pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)

        rospy.spin()

    def _click_cb(self, msg):
        if msg.data:
            self._send_reset()
            self._unpause()
        else:
            self._pause()

    def _mouse_cb(self, msg):
        command = [.5*msg.v, -0.5*msg.h, +0.2*msg.h, 0]
        for i in range(len(command)):
            self.cs_pubs[i].publish(Float32(command[i]))

    def _send_reset(self):
        # theoretically limits altitude to 30 m if it doesn't gain energy
        E = 50*9.81 

        z = 15
        v = np.sqrt(2 * (E - 9.81 * z))

        yaw = -np.pi/2 

        self.init_state.pose.position.x = 50
        self.init_state.pose.position.y = 100
        self.init_state.pose.position.z = 2 + z
        self.init_state.pose.orientation.x = 0
        self.init_state.pose.orientation.y = 0
        self.init_state.pose.orientation.w = np.cos(yaw/2)
        self.init_state.pose.orientation.z = np.sin(yaw/2)

        self.init_state.twist.linear.x  = v * np.cos(yaw)
        self.init_state.twist.linear.y  = v * np.sin(yaw) 
        self.init_state.twist.linear.z  = 0
        self.init_state.twist.angular.x = 0
        self.init_state.twist.angular.y = 0
        self.init_state.twist.angular.z = 0

        self.state = copy.deepcopy(self.init_state)

        self.last_energy = E
        self.last_x = 0.0
        self.extracted_energy = 0.0
        self.episode_start_time = rospy.Time.now()

        self.state_pub.publish(self.init_state)


def main():
    rospy.init_node('mouse_to_soar')

    FWMouseControl()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
