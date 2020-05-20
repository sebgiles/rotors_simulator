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

    def __init__(self):
        rospy.init_node('gym', anonymous=True)

        self.time_step = 0.3
        self.ail_cmd_limit = 0.5
        self.elev_cmd_limit = 0.5

        self.wind_floor_z = 10

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
        self.ail_l_pub  = rospy.Publisher("/uav_1/ail_l_pos",  Float32, queue_size=1)
        self.ail_r_pub  = rospy.Publisher("/uav_1/ail_r_pos",  Float32, queue_size=1)
        self.elev_pub = rospy.Publisher("/uav_1/elev_pos", Float32, queue_size=1)

        self.ail_cmd  = None
        self.elev_cmd = None

        # to provide observations
        # TODO?: evaluate other topics with lighter messages
        rospy.Subscriber("/gazebo/model_states", ModelStates,
                         self._gazebo_state_callback, queue_size=1)
        self.latest_state_msg = None 
        self.state = None

        # gym.Env overrides:
        #   observation space: (z, v, yaw, pitch, roll)
        obs_low  = [    0.0,    0.0, -np.pi, -np.pi/2, -np.pi]
        obs_high = [ np.Inf, np.Inf, +np.pi, +np.pi/2, +np.pi]
        self.observation_space = spaces.Box(low  =np.array(obs_low), 
                                            high =np.array(obs_high))

        #   action space: (pitch_increment, roll_increment)
        self.action_space = spaces.Box(
            low  = np.array([-1.0, -1.0]), 
            high = np.array([+1.0, +1.0])
            )

        #   reward range
        self.reward_range = (-np.inf, np.inf)

        # Variables used to calculate rewards and metrics
        self.extracted_energy   = None
        self.extracted_energy   = None
        self.mean_energy        = None
        self.max_energy         = None
        self.min_airspeed       = None
        self.last_energy        = None
        self.last_x             = None
        self.last_yaw           = None
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


    def get_wind(self, z):
        wind = np.array([0,0,0], dtype=float)
        shear_top = 15.0
        wind_grad = -1.0
        wind[1] = wind_grad * max(min(z, shear_top), 0.0)
        return wind

    def rand(self): # number in [-1,1]
        return 2 * (self.np_random.rand() - 0.5)

    # Resets the state of the environment and returns initial observation
    def reset(self):
        good_enough = False
        while not good_enough:
            yaw     = self.rand() * np.pi
            pitch   = self.rand() * np.pi/2
            roll    = self.rand() * np.pi
            z       = self.rand() * 10.0 + 7.5
            if abs(roll) > 2.0 and pitch > 0.0:
                good_enough = False
            else:
                good_enough = True

        rot = Rotation.from_euler('ZYX', [yaw,pitch,roll])

        # theoretically limits altitude to 60 m if it doesn't gain energy
        E = 60*9.81 
        
        wind = rot.apply(self.get_wind(z), inverse=True) # wind in glider frame

        airspeed = np.sqrt(2 * (E - 9.81*z) - wind[1]**2 - wind[2]**2) - wind[0]

        v = rot.apply(wind+np.array([airspeed,0,0]), inverse=False)

        self.init_state.pose.position.x = 0
        self.init_state.pose.position.y = 0 + 400
        self.init_state.pose.position.z = self.wind_floor_z + z
        self.init_state.pose.orientation.x = rot.as_quat()[0]
        self.init_state.pose.orientation.y = rot.as_quat()[1]
        self.init_state.pose.orientation.z = rot.as_quat()[2]
        self.init_state.pose.orientation.w = rot.as_quat()[3]

        self.init_state.twist.linear.x  = v[0]
        self.init_state.twist.linear.y  = v[1]
        self.init_state.twist.linear.z  = v[2]

        self.init_state.twist.angular.x = 0
        self.init_state.twist.angular.y = 0
        self.init_state.twist.angular.z = 0

        self.state = copy.deepcopy(self.init_state)

        self.last_energy = E
        self.last_x = 0.0
        self.last_yaw = yaw
        self.total_rotation = 0.0
        self.extracted_energy = 0.0
        self.mean_energy      = 0.0
        self.max_energy         = E
        self.min_airspeed       = 999.0
        self.episode_start_time = rospy.Time.now().to_time()

        self.latest_state_msg = None

        self.state_pub.publish(self.init_state)

        observation, _, _, _ = self.step([0.0,0.0])

        return observation


    def step(self, action):
        self._apply_action(action)
        self._unpause()
        # wait as the simulation runs
        rospy.sleep(self.time_step)
        self._freeze()
        observation, reward, done, info = self._observe()
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

        x = pose.position.x
        y = pose.position.y
        z = pose.position.z - self.wind_floor_z
        vx = twist.linear.x
        vy = twist.linear.y
        vz = twist.linear.z  

        quat = (pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w)

        rot = Rotation.from_quat(quat)
        euler = rot.as_euler('ZYX')

        now = rospy.Time.now().to_time()

        v = np.array([vx,vy,vz])

        air_rel_vel = rot.apply(v - self.get_wind(z), inverse=True)
        airspeed = air_rel_vel[0]

        yaw   = euler[0]
        pitch = euler[1]
        roll  = euler[2]

        observation = np.array([z, airspeed, yaw, pitch, roll])

        if observation_only:
            return observation

        K = 0.5 * np.linalg.norm(v)
        U = 9.81 * z
        E = K + U

        deltaE = E - self.last_energy
        delta_x = x - self.last_x

        delta_yaw = yaw - self.last_yaw
        if abs(delta_yaw) > np.pi:
            delta_yaw = -np.sign(delta_yaw)*(np.abs(delta_yaw) - 2*np.pi)

        self.total_rotation += delta_yaw
        self.last_energy = E
        self.last_x = x
        self.last_yaw = yaw
        
        self.extracted_energy += max([0, deltaE])
        self.max_energy        = max([self.max_energy, E])
        self.min_airspeed      = min([self.min_airspeed, airspeed])
        self.mean_energy      += E*self.time_step

        #reward = 1
        #reward = delta_x
        reward = max([0, deltaE])

        done = False
        if now - self.episode_start_time > 180:
            done = True
        elif z < -8:
            done = True  
            reward = 0 
        elif abs(roll) > 0.75*np.pi and z < 10:
        if abs(roll) > 0.75*np.pi and z < 10:
            done = True
            reward = 0
            

        # "TELEMETRY"
        if done and self.episode_start_time is not None: 
            # update these members so the StableBaselines callback can get
            # them and add them to the tensorboard log
            self.duration = now - self.episode_start_time
            self.duration = now.to_time() - self.episode_start_time.to_time()
            self.final_airspeed = airspeed
            self.final_altitude = z
            self.terminal_energy = E
            #self.extracted_energy = self.extracted_energy
            #self.max_energy         = None
            #self.min_airspeed       = None
            #self.total_rotation = self.total_rotation
            self.mean_energy /= self.duration
            self.positive_power = self.extracted_energy/self.duration 

        info = {
            'total_rotation':  self.total_rotation,
            'max_energy':      self.max_energy,
            'min_airspeed':    self.min_airspeed,
            'mean_energy':     self.mean_energy,
        }

        return observation, reward, done, info


    # Send updated roll and pitch setpoints to the low-level controller 
    def _apply_action(self, action):

        # self.pitch_cmd = action[0]
        # self.roll_cmd  = action[1]

        self.elev_cmd = action[0] * self.elev_cmd_limit
        self.ail_cmd  = action[1] * self.ail_cmd_limit
               
        self.elev_pub.publish(Float32(self.elev_cmd))
        self.ail_l_pub.publish(Float32(+self.ail_cmd))
        self.ail_r_pub.publish(Float32(-self.ail_cmd))


    def render(self, mode='human'):
        pass


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


register(
    id='l2s-energy-v1',
    entry_point=AlbatrossEnv
)