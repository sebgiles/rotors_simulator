#!/usr/bin/env python

import rospy
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Float32
import learn2soar as l2s

cs_topics = [  "/uav_1/rudd_pos",      \
                "/uav_1/elev_pos",      \
                "/uav_1/ail_l_pos",     \
                "/uav_1/ail_r_pos",     \
                "/uav_1/flap_l1_pos",   \
                "/uav_1/flap_l2_pos",   \
                "/uav_1/flap_r1_pos",   \
                "/uav_1/flap_r2_pos",    \
                "/uav_1/prop_ref_0"
              ]

class Learn2SoarROSInterface:

    logic = None
    cs_pubs = []

    def __init__(self):
        self.logic = l2s.Learn2Soar()
        rospy.init_node('rotors_gazebo')
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.callback, queue_size=1 )
        for topic in cs_topics:
            self.cs_pubs.append(rospy.Publisher(topic, Float32, queue_size=1))
        rospy.spin()

    def callback(self, msg):
        command = self.logic.doControl(msg.pose[-1], msg.twist[-1])
        for i in range(len(command)):
            self.cs_pubs[i].publish(Float32(command[i]))

if __name__ == '__main__':
    l2sif = Learn2SoarROSInterface()


