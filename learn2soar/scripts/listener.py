#!/usr/bin/env python

import rospy
from gazebo_msgs.msg import ModelStates

def callback(msg):

    x = msg.pose[12].position.x
    y = msg.pose[12].position.y
    z = msg.pose[12].position.z
    rospy.loginfo('\n x: {}\n y: {}\n altitude: {}'.format(x,y,z))

def main():
    
    rospy.init_node('rotors_gazebo')
    rospy.Subscriber("/gazebo/model_states", ModelStates, callback)
    rospy.spin()

if __name__ == '__main__':
        main()


