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


        # # Launch the simulation with the given launchfile name
        # rospack = rospkg.RosPack()
        # package_path = rospack.get_path('rotors_gym')
        # launchfile = "learn_to_soar_env.launch"
        # full_launchfile_path = os.path.join(package_path, "launch", launchfile)


        # uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        # roslaunch.configure_logging(uuid)
        # launch = roslaunch.parent.ROSLaunchParent(uuid, [full_launchfile_path])
        # launch.start()

        rospy.init_node('gym', anonymous=True)

        # # configure simulation settings
        # set_physics_svc_name = '/gazebo/set_physics_properties'
        # get_physics_svc_name = '/gazebo/get_physics_properties'
        # rospy.wait_for_service(set_physics_svc_name)
        # rospy.wait_for_service(get_physics_svc_name)
        # self.set_physics = rospy.ServiceProxy(set_physics_svc_name, SetPhysicsProperties)
        # self.get_physics = rospy.ServiceProxy(get_physics_svc_name, Empty)
        # poop = self.get_physics()
        # print(type(poop))
        # print(poop)

        # set_physics_request = SetPhysicsPropertiesRequest()
        # set_physics_request.time_step = 0.004
        # set_physics_request.max_update_rate = 0
        # result = self.set_physics(set_physics_request)

        # to step simulation
        self._unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self._pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)

        self._seed()

        # to provide observations
        # TODO?: evaluate other topics with lighter messages
        rospy.Subscriber("/gazebo/model_states", ModelStates,
                         self._gazebo_state_callback, queue_size=1)

        self.roll_pub = rospy.Publisher("/l2s/attitude_cmd/roll", Float32, queue_size=1)
        self.pitch_pub = rospy.Publisher("/l2s/attitude_cmd/pitch", Float32, queue_size=1)
        self.roll_cmd = 0
        self.pitch_cmd = 0
        # to reset to arbitrary initial pose
        self.state_pub = rospy.Publisher('/gazebo/set_model_state', ModelState,
                                         queue_size=1)

        self.init_state = ModelState()
        self.init_state.model_name = 'uav_1'
        self.init_state.reference_frame = 'ground_collision_plane'

        self.state = copy.deepcopy(self.init_state)
        self.latest_state_msg = None

        self.time_step = 0.2
        self.pausing = False
        self.late_msgs = 0

        #  (alpha, d, v_norm, roll, pitch)

        #  (y, z, v_norm, roll, pitch)
        obs_low  = [-25, 25, -np.pi, -np.pi, -np.pi/2]
        obs_high = [+25, 25, +np.pi, +np.pi, +np.pi/2]
        self.observation_space = spaces.Box(low  =np.array(obs_low), 
                                            high =np.array(obs_high),
                                            dtype=np.float32)

        self.action_space = spaces.MultiDiscrete([3,3])
        self.reward_range = (-np.inf, np.inf)

        rospy.wait_for_service('/gazebo/pause_physics')
        rospy.wait_for_service('/gazebo/unpause_physics')

        print('==== Environment is ready ====')


    def _gazebo_state_callback(self, msg):
        self.latest_state_msg = msg
        if self.pausing:
            self.late_msgs += 1


    def _freeze(self):

        # rosclk_b4 = rospy.Time.now()
        # wallclk_b4 = time.time()
        self.pausing = True
        self.late_msgs = 0
        self._pause()

        # last = rosclk_b4
        
        # while 1:
        #     now = rospy.Time.now()
        #     if now == last: break
        #     last = now
        # if True or now.nsecs != rosclk_b4.nsecs:
        #     print("Simtime to pause:\t{}\tLate msgs:\t{}".format(1e-6*(now-rosclk_b4), self.late_msgs))
        
        # self.late_msgs = 0
        
        # time.sleep(0.04)
        # print("Necro msgs:\t{}".format(self.late_msgs))
    
        if self.latest_state_msg is not None:
            self.state.pose  = self.latest_state_msg.pose[-1]
            self.state.twist = self.latest_state_msg.twist[-1]


    # must only look at self.state
    def _observe(self):
        pose    = self.state.pose
        twist   = self.state.twist

        x = pose.position.x
        y = pose.position.y
        z = pose.position.z - 30

        # vx = twist.linear.x
        vy = twist.linear.y
        vz = twist.linear.z  

        quat = (pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w)

        euler = Rotation.from_quat(quat).as_euler('zyx')
        roll  = euler[0]
        pitch = euler[1]
        yaw   = euler[2]

        u = -np.array([y,z])
        d = np.linalg.norm(u)
        v_norm = np.dot(u/d, [vy, vz])    

        boundary = 29
        reward = 0
        done = False

        if d < 1:
            reward = 10
        else:
            reward = 10*(1-((d-1)/(boundary-1))**0.5)

        if d > boundary: 
            done = True
            reward = -100
        
        if x > 500:
            done = True
            reward = +100

        observation = (y, z, yaw, roll, pitch)

        return observation, reward, done


    def _apply_action(self, action):
        roll_step_size = 0.02
        pitch_step_size = 0.02

        self.pitch_cmd += (action[0]-1) * pitch_step_size
        self.roll_cmd  += (action[1]-1) * roll_step_size 
               
        self.roll_pub.publish(Float32(self.roll_cmd))
        self.pitch_pub.publish(Float32(self.pitch_cmd))


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    # Resets the state of the environment and returns initial observation
    def reset(self):
        self._pause()
        self.pitch_cmd = 0
        self.roll_cmd = 0

        self.roll_pub.publish(Float32(self.pitch_cmd))
        self.pitch_pub.publish(Float32(self.roll_cmd))


        self.init_state.twist.linear.x = 18
        self.init_state.pose.position.x = 0
        self.init_state.pose.position.z = 30 + 30*(self.np_random.rand()-0.5)
        self.init_state.pose.position.y = 0  + 30*(self.np_random.rand()-0.5)
        yaw = self.np_random.rand()-0.5
        self.init_state.pose.orientation.w = np.cos(yaw/2)
        self.init_state.pose.orientation.x = np.sin(yaw/2)

        self.state_pub.publish(self.init_state)
        self.state = copy.deepcopy(self.init_state)
        self.latest_state_msg = None
        observation, _, _  = self._observe()
        return observation


    def step(self, action):

        self._apply_action(action)
        self.pausing = False
        self._unpause()
        # wait as the simulation runs,
        # small annoyance: will continue sleeping if rosmaster dies
        rospy.sleep(self.time_step)
        self._freeze()
        observation, reward, done = self._observe()
        info = {}

        return observation, reward, done, info

    def render(self, mode='human'):
        pass

register(
    id='LearnToSoar-v1',
    entry_point=LearnToSoarEnv
)