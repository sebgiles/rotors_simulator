#!/usr/bin/env python2

import rospy
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Float32
from controller import Controller
import time, sys

cs_topics = [   "/uav_1/elev_pos",      
                "/uav_1/ail_l_pos",     
                "/uav_1/ail_r_pos",
                "/uav_1/rudd_pos",      
                #"/uav_1/prop_ref_0"
              ]

class Learn2SoarROSInterface:

    logic = None
    cs_pubs = []

    def __init__(self):
        self.logic = Controller()
        rospy.init_node('rotors_gazebo')

        rospy.Subscriber("/gazebo/model_states", ModelStates, self.new_sensor_data_callback, queue_size=1)
        rospy.Subscriber("/l2s/attitude_cmd/roll", Float32, self.new_cmd_roll, queue_size=1)
        rospy.Subscriber("/l2s/attitude_cmd/pitch", Float32, self.new_cmd_pitch, queue_size=1)
        rospy.Subscriber("/l2s/attitude_cmd/yaw", Float32, self.new_cmd_yaw, queue_size=1)
        #rospy.Subscriber("/l2s/motor_cmd", Float32, self.new_cmd_motor, queue_size=1)
        
        for topic in cs_topics:
            self.cs_pubs.append(rospy.Publisher(topic, Float32, queue_size=1))

        rospy.Timer(rospy.Duration(self.logic.low_level_control_period), self.do_control_callback)
        #rospy.Timer(rospy.Duration(self.logic.high_level_control_period), self.do_alt_control_callback)

        self.freq_count         = 0
        self.last_ros_time      = rospy.Time.now()
        self.last_print         = time.time()
        self.started = False

    def new_cmd_roll(self, msg):
        self.logic.phi_ref = msg.data

    def new_cmd_pitch(self, msg):
        self.logic.theta_ref = msg.data

    def new_cmd_yaw(self, msg):
        self.logic.psi_ref = msg.data      

    def new_cmd_motor(self, msg):
        self.logic.prop = msg.data  

    def new_sensor_data_callback(self, msg):
        self.logic.update_sensor_data(msg.pose[-1], msg.twist[-1])

    def do_alt_control_callback(self,event):
        self.logic.do_high_level_control()

    def do_control_callback(self, event):
        if not self.started: 
            self.started = True
            return #skip first iteration

        rosnow = event.current_real
        if self.logic.do_low_level_control(rosnow - event.last_real):
            self.freq_count += 1
            command = self.logic.command
            for i in range(len(self.cs_pubs)):
                self.cs_pubs[i].publish(Float32(command[i]))

        wallnow = time.time()
        if (wallnow - self.last_print) > 1.0:
            self.print_actual_freq(rosnow)
            self.last_print = wallnow


    def print_actual_freq(self, rosnow):
        ros_freq  = 1.0*self.freq_count/(rosnow - self.last_ros_time).to_sec()
        print "Low Level Control: {:.0f} Hz [sim time] \r".format(ros_freq),
        sys.stdout.flush()

        self.freq_count = 0
        self.last_ros_time = rosnow



if __name__ == '__main__':
    l2sif = Learn2SoarROSInterface()
    rospy.spin()


