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
        
        # Time-step toggle:
        self.time_step = 0.3

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

        self.action_space = spaces.MultiDiscrete([3,3])
        self.reward_range = (-np.inf, np.inf)

        # Variables used to calculate rewards and metrics
        self.initial_energy     = None
        self.step_energy        = None
        self.extracted_energy   = None
        self.positive_power     = None
        self.last_energy        = None
        self.terminal_energy    = None
        self.last_x             = None
        self.final_distance     = None
        self.final_altitude     = None
        self.final_airspeed     = None
        self.episode_start_time = None
        self.duration           = None
        self.last_time          = None

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

        v = np.linalg.norm([vx, vy, vz])
        airspeed = np.linalg.norm([vx, vy - vy_wind, vz])

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

        self.step_energy += deltaE
        self.extracted_energy += max(0, deltaE)

        # Reward with time-step
        now = rospy.Time.now()
        time_step = now.to_time() - self.last_time

        reward = deltaE

        self.last_time = now.to_time()

        done = False

        if z < 0:
            done = True
            reward += -E # punish for crash landing by speed
        
        if z > 20:
            done = True

        observation = (z, v, roll, yaw, pitch)

        if done and self.episode_start_time is not None: 
            # update these members so the StableBaselines callaback can get
            # them and add them to the tensorboard log
            now = rospy.Time.now()
            self.duration = now.to_time() - self.episode_start_time.to_time()
            self.final_airspeed = airspeed
            self.final_distance = x
            self.final_altitude = z
            self.terminal_energy = E
            self.step_energy = self.step_energy
            self.extracted_energy = self.extracted_energy
            self.positive_power = self.extracted_energy/self.duration

            if E >= self.initial_energy:
                reward += E

        return observation, reward, done


    def _apply_action(self, action):
        roll_step_size = 0.075
        pitch_step_size = 0.075

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

        # # theoretically limits altitude to 20 m if it doesn't gain energy
        # E = 20*9.81 
        # rand_n = (self.np_random.rand())

        # z = 10 + 2.5 * rand_n # init alt in [10m, 15m]
        # v = np.sqrt(2 * (E - 9.81 * z))

        # yaw_rand_n = 2.0*(rand_n - 0.5)
        # yaw = 0.5 + 0.25*rand_n # init heading in ~[15°, 45°]

        # Initial conditions in "typical" soaring trajectory:

        randn = self.np_random.rand() - 0.5 # number in [-0.5, 0.5]

        yaw = 2.0 * np.pi * randn            # init yaw in [-pi, pi]
        pitch = -0.25 * np.pi * np.sin(yaw)  # init pitch in [-pi/4, pi/4]

        z = 4 * np.cos(yaw) + 11             # # init alt in [7m, 15m]

        E = 25 * 9.81                        # Energy at top
        v = np.sqrt(2 * (E - 9.81 * z))      # velocity at different altitudes

        quat_orient = Rotation.from_euler('zyx', [yaw,pitch,0.0])

        self.init_state.pose.position.x = 0 - 400
        self.init_state.pose.position.z = z + 2
        self.init_state.pose.position.y = 0
        self.init_state.pose.orientation.x = quat_orient.as_quat()[0]
        self.init_state.pose.orientation.y = quat_orient.as_quat()[1]
        self.init_state.pose.orientation.z = quat_orient.as_quat()[2]
        self.init_state.pose.orientation.w = quat_orient.as_quat()[3]
        self.init_state.twist.linear.x  = v * np.cos(yaw) * np.cos(pitch)
        self.init_state.twist.linear.y  = v * np.sin(yaw) * np.cos(pitch)
        self.init_state.twist.linear.z  = v * np.sin(pitch)

        self.state = copy.deepcopy(self.init_state)

        self.initial_energy = E
        self.last_energy = E
        self.last_x = 0.0
        self.step_energy = 0.0
        self.extracted_energy = 0.0
        self.last_time = 0.0
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
    id='LearnToSoar-v4',
    entry_point=LearnToSoarEnv
)