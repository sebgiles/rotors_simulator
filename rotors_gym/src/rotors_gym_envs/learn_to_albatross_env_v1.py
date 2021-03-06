#!/usr/bin/env python3
import numpy as np
from scipy.spatial.transform import Rotation

import rospy, copy

from std_srvs.srv import Empty
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates, ModelState
from gazebo_msgs.srv import SetPhysicsProperties, SetPhysicsPropertiesRequest

import gym
from gym import utils, spaces
from gym.utils import seeding


# registration happens after the class definition
from gym.envs.registration import register

class AlbatrossEnv(gym.Env):

    env_id = 'Albatross-v1'

    def __init__(self):
        rospy.init_node('gym_albatross', anonymous=True)

        self.time_step = 0.2

        # Units will be warped by tangent(current_angle)
        self.pitch_cmd_step = 0.15
        self.roll_cmd_step = 0.25

        self.wind_floor_z = 10.

        # to step simulation
        rospy.wait_for_service('/gazebo/pause_physics')    
        rospy.wait_for_service('/gazebo/unpause_physics')
        self._unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self._pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)

        # to reset to arbitrary initial pose
        self.state_pub = rospy.Publisher('/gazebo/set_model_state', ModelState,
                                         queue_size=1)
        self.init_state = ModelState()
        self.init_state.model_name = 'uav_1'
        self.init_state.reference_frame = 'ground_collision_plane'

        # to send commannds
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

        # gym.Env overrides:
        #   observation space: (z, v, roll, yaw, pitch)
        obs_low  = [    0.0,    0.0, -np.pi, -np.pi, -np.pi/2]
        obs_high = [ np.Inf, np.Inf, +np.pi, +np.pi, +np.pi/2]
        self.observation_space = spaces.Box(low  =np.array(obs_low), 
                                            high =np.array(obs_high))

        #   action space: (pitch_increment, roll_increment)
        self.action_space = spaces.MultiDiscrete([3,3])

        #   reward range
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

        # Initialize random number generator
        self._seed()

        # Set initial condition and pause simulation
        #self.reset()

        print('==== Environment is ready ====')


    # Resets the state of the environment and returns initial observation
    def reset(self):
        # theoretically limits altitude to 40 m if it doesn't gain energy
        E = 50.0 * 9.81 

        randn = 2.0 * (self.np_random.rand() - 0.5) # number in [-1,1]
        z = 20 + 1.0 * randn
        v = np.sqrt(2.0 * (E - 9.81 * z))
        pitch = 0.0

        randn = 2.0 * (self.np_random.rand() - 0.5) # number in [-1,1]
        yaw = - 0.8 + 0.1 * randn 
        heading = yaw - 0.3
        roll = 0.0

        quat_orient = Rotation.from_euler('zyx', [yaw,pitch,roll])

        self.pitch_cmd = pitch
        self.roll_cmd  = roll 

        self.init_state.pose.position.x = 0.0 - 400.0
        self.init_state.pose.position.y = 0.0 + 400.0
        self.init_state.pose.position.z = self.wind_floor_z + z
        self.init_state.pose.orientation.x = quat_orient.as_quat()[0]
        self.init_state.pose.orientation.y = quat_orient.as_quat()[1]
        self.init_state.pose.orientation.z = quat_orient.as_quat()[2]
        self.init_state.pose.orientation.w = quat_orient.as_quat()[3]

        self.init_state.twist.linear.x  = v * np.cos(heading)
        self.init_state.twist.linear.y  = v * np.sin(heading) 
        self.init_state.twist.linear.z  = 0.0
        self.init_state.twist.angular.x = 0.0
        self.init_state.twist.angular.y = 0.0
        self.init_state.twist.angular.z = 0.0

        self.state = copy.deepcopy(self.init_state)

        self.last_energy = E
        self.last_x = 0.0
        self.extracted_energy = 0.0
        self.episode_start_time = rospy.Time.now()

        self.roll_pub.publish(Float32(self.pitch_cmd))
        self.pitch_pub.publish(Float32(self.roll_cmd))
        self._pause()
        self.state_pub.publish(self.init_state)

        self.latest_state_msg = None
        observation = self._observe(observation_only=True)
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


    # Update ground truth
    def _gazebo_state_callback(self, msg):
        self.latest_state_msg = msg


    # Pause and set state to the last received ground truth message
    # (if ground truth messages are received after pausing they will not be considered to 
    # compute the reward and observation)
    def _freeze(self):
        self._pause()
        if self.latest_state_msg is not None:
            self.state.pose  = self.latest_state_msg.pose[-1]
            self.state.twist = self.latest_state_msg.twist[-1]


    # Compute reward and observations to feed back to the agent
    # (all computations must ignore self.latest_state_msg)
    def _observe(self, observation_only=False):
        pose    = self.state.pose
        twist   = self.state.twist

        x = pose.position.x + 400
        #y = pose.position.y
        z = pose.position.z - self.wind_floor_z

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

        euler = Rotation.from_quat(quat).as_euler('ZYX')
        yaw   = euler[0]
        pitch = euler[1]
        roll  = euler[2]

        observation = np.array([z, v, yaw, pitch, roll])

        if observation_only:
            return observation

        K = 0.5 * v**2
        U = 9.81 * z
        E = K + U

        deltaE = E - self.last_energy
        delta_x = x - self.last_x

        self.last_energy = E
        self.last_x = x
        
        self.extracted_energy += max([0, deltaE])

        #reward = 1 
        reward = delta_x

        if z < -5:
            done = True
            #reward = -100
            #reward += -E  # punish for crash landing by speed

        # elif pitch < - 0.5 and airspeed < 8:
        #     done = True
            #reward = -100
            #reward += -E  # punish for stalling by altitude 
        
        # elif np.abs(yaw) > 0.55 * np.pi:
        #     done = True
        #     #reward = -100
        #     #reward = -E

        # elif x > 800:
        #     done = True
        else:
            done = False

        if done and self.episode_start_time is not None: 
            # update these members so the StableBaselines callback can get
            # them and add them to the tensorboard log
            now = rospy.Time.now()
            self.duration = now.to_time() - self.episode_start_time.to_time()
            self.final_airspeed = airspeed
            self.final_altitude = z
            self.terminal_energy = E
           #self.extracted_energy = self.extracted_energy
            self.positive_power = self.extracted_energy/self.duration 

        return observation, reward, done

    # Send updated roll and pitch setpoints to the low-level controller 
    def _apply_action(self, action):

        self.pitch_cmd += (action[0]-1) * self.pitch_cmd_step
        self.roll_cmd  += (action[1]-1) * self.roll_cmd_step

        # self.pitch_cmd = action[0] * self.pitch_cmd_limit
        # self.roll_cmd  = action[1] * self.roll_cmd_limit
               
        self.pitch_pub.publish(Float32(np.arctan(self.pitch_cmd)))
        self.roll_pub.publish(Float32(np.arctan(self.roll_cmd)))


    def render(self, mode='human'):
        pass


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


register(
    id=AlbatrossEnv.env_id,
    entry_point=AlbatrossEnv
)