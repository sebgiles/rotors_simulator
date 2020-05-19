import numpy as np
from scipy.spatial.transform import Rotation

import rospy, rospkg, copy
import os, roslaunch

from std_srvs.srv import Empty
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates, ModelState
from gazebo_msgs.srv import SetPhysicsProperties, SetPhysicsPropertiesRequest

import gym
from gym import utils, spaces
from gym.utils import seeding


# registration happens after the LearnToSoarEnv class definition
from gym.envs.registration import register

class LearnToSoarEnv(gym.Env):

    def __init__(self):
        rospy.init_node('gym', anonymous=True)

        self.time_step = 0.5

        self.roll_pub  = rospy.Publisher("/l2s/attitude_cmd/roll",  Float32, queue_size=1)
        self.pitch_pub = rospy.Publisher("/l2s/attitude_cmd/pitch", Float32, queue_size=1)
        self.roll_cmd  = None
        self.pitch_cmd = None

        # to provide observations
        # TODO?: evaluate other topics with lighter messages
        rospy.Subscriber("/gazebo/model_states", ModelStates,
                         self._gazebo_state_callback, queue_size=1)
        self.latest_state_msg = None 
        self.state = None

        # to reset to arbitrary initial pose
        self.state_pub = rospy.Publisher('/gazebo/set_model_state', ModelState,
                                         queue_size=1)
                                        
        self.init_state = ModelState()
        self.init_state.model_name = 'uav_1'
        self.init_state.reference_frame = 'ground_collision_plane'

        #  (z, v, roll, yaw, pitch)
        obs_low  = [0,   0, -np.pi, -np.pi, -np.pi/2]
        obs_high = [80, 50, +np.pi, +np.pi, +np.pi/2]
        self.observation_space = spaces.Box(low  =np.array(obs_low), 
                                            high =np.array(obs_high),
                                            dtype=np.float32)

        self.action_space = spaces.Box(low  = np.array([-1, -1]), 
                                       high = np.array([+1, +1]), 
                                       dtype=int)
        self.reward_range = (-np.inf, np.inf)

        # Variables used to calculate rewards and metrics
        self.extracted_energy   = None
        self.last_energy        = None
        self.last_x             = None
        self.final_airspeed     = None
        self.final_altitude     = None
        self.terminal_energy    = None
        self.positive_power     = None
        self.episode_start_time = None
        self.duration           = None

        # to step simulation
        rospy.wait_for_service('/gazebo/pause_physics')    
        rospy.wait_for_service('/gazebo/unpause_physics')
        self._unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self._pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)

        self._seed()
        self.reset()

        print('==== Environment is ready ====')


    def _gazebo_state_callback(self, msg):
        self.latest_state_msg = msg


    def _freeze(self):
        self._pause()
        if self.latest_state_msg is not None:
            self.state.pose  = self.latest_state_msg.pose[-1]
            self.state.twist = self.latest_state_msg.twist[-1]


    # must only look at self.state
    def _observe(self):
        pose    = self.state.pose
        twist   = self.state.twist

        x = pose.position.x + 400
        y = pose.position.y
        z = pose.position.z - 2

        vx = twist.linear.x
        vy = twist.linear.y
        vz = twist.linear.z  

        vy_wind = -1 * z 

        v = np.linalg.norm([vx,vy,vz])
        airspeed = np.linalg.norm([vx, vy-vy_wind, vz])

        quat = (pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w)

        euler = Rotation.from_quat(quat).as_euler('zyx')
        yaw   = euler[0]
        pitch = euler[1]
        roll  = euler[2]

        K = 0.5 * v**2
        U = 9.81 * z
        E = K + U

        deltaE = E - self.last_energy
        delta_x = x - self.last_x

        self.last_energy = E
        self.last_x = x
        
        self.extracted_energy += max([0, deltaE])

        reward = delta_x

        done = False


        if z < 5:
            done = True
            #reward = -100
            #reward += -E  # punish for crash landing by speed

        elif pitch < - 0.5 and airspeed < 8:
            done = True
            #reward = -100
            #reward += -E  # punish for stalling by altitude 
        
        elif np.abs(yaw) > 0.55 * np.pi:
            done = True
            #reward = -100
            #reward = -E

        elif x > 800:
            done = True

        observation = (z, v, roll, yaw, pitch)

        if done and self.episode_start_time is not None: 
            # update these members so the StableBaselines callaback can get
            # them and add them to the tensorboard log
            now = rospy.Time.now()
            self.duration = now.to_time() - self.episode_start_time.to_time()
            self.final_airspeed = airspeed
            self.final_altitude = z
            self.terminal_energy = E
           #self.extracted_energy = self.extracted_energy
            self.positive_power = self.extracted_energy/self.duration 

        return observation, reward, done


    def _apply_action(self, action):
        roll_step_size = 0.1
        pitch_step_size = 0.1

        self.pitch_cmd += (action[0]-1) * pitch_step_size
        self.roll_cmd  += (action[1]-1) * roll_step_size 
               
        self.roll_pub.publish(Float32(np.arctan(self.roll_cmd)))
        self.pitch_pub.publish(Float32(np.arctan(self.pitch_cmd)))


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    # Resets the state of the environment and returns initial observation
    def reset(self):
        self.pitch_cmd = 0
        self.roll_cmd  = 0
        self.roll_pub.publish(Float32(self.pitch_cmd))
        self.pitch_pub.publish(Float32(self.roll_cmd))

        # theoretically limits altitude to 30 m if it doesn't gain energy
        E = 20*9.81 
        rand_n = 2.0*(self.np_random.rand() - 0.5)
        z = 16 + 2 * rand_n
        v = np.sqrt(2 * (E - 9.81 * z))

        rand_n = 2.0*(self.np_random.rand() - 0.5)
        yaw = 0.5 + 0.250*rand_n

        self.init_state.pose.position.x = - 400 + 0
        self.init_state.pose.position.y = 0
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

        self._pause()
        self.state_pub.publish(self.init_state)
        self.latest_state_msg = None
        observation, _, _  = self._observe()
        return observation


    def step(self, action):
        self._apply_action(action)
        self._unpause()
        # wait as the simulation runs
        rospy.sleep(self.time_step)
        self._freeze()
        observation, reward, done = self._observe()
        info = {}
        return observation, reward, done, info


    def render(self, mode='human'):
        pass


register(
    id='LearnToSoar-v3',
    entry_point=LearnToSoarEnv
)