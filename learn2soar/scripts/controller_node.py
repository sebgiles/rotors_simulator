#!/usr/bin/env python

import rospy
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Float32
from controller import Controller

cs_topics = [  "/uav_1/rudd_pos",      
                "/uav_1/elev_pos",      
                "/uav_1/ail_l_pos",     
                "/uav_1/ail_r_pos",     
                "/uav_1/flap_l1_pos",   
                "/uav_1/flap_l2_pos",   
                "/uav_1/flap_r1_pos",   
                "/uav_1/flap_r2_pos",    
                "/uav_1/prop_ref_0"
              ]

class Learn2SoarROSInterface:

    logic = None
    cs_pubs = []

    def __init__(self):
        self.logic = Controller()
        rospy.init_node('rotors_gazebo')

        rospy.Subscriber("/gazebo/model_states", ModelStates, self.new_sensor_data_callback, queue_size=1)

        for topic in cs_topics:
            self.cs_pubs.append(rospy.Publisher(topic, Float32, queue_size=1))

        rospy.Timer(rospy.Duration(self.logic.roll_pitch_control_period), self.do_control_callback)
        rospy.Timer(rospy.Duration(self.logic.alt_control_period), self.do_alt_control_callback)
        rospy.spin()

    def new_sensor_data_callback(self, msg):
        self.logic.update_sensor_data(msg.pose[-1], msg.twist[-1])

    def do_alt_control_callback(self,event):
        self.logic.do_high_level_control()

    def do_control_callback(self, event):
        self.logic.do_low_level_control()
        command = self.logic.command
        for i in range(len(command)):
            self.cs_pubs[i].publish(Float32(command[i]))

if __name__ == '__main__':
    l2sif = Learn2SoarROSInterface()


