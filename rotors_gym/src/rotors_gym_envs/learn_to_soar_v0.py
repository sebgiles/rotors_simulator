import numpy as np
import copy

import rospy
import rospkg, roslaunch, os

from std_srvs.srv import Empty
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates, ModelState

import gym
from gym import utils, spaces
from gym.utils import seeding

# registration happens after the LearnToSoarEnv class definition
from gym.envs.registration import register

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

class LearnToSoarEnv(gym.Env):


    def __init__(self):

        # Launch the simulation with the given launchfile name
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('rotors_gym')
        launchfile = "learn_to_soar_v0.launch"
        full_launchfile_path = os.path.join(package_path, "launch", launchfile)


        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        launch = roslaunch.parent.ROSLaunchParent(uuid, [full_launchfile_path])
        launch.start()
        rospy.loginfo("started")
        rospy.init_node('gym', anonymous=True)

        # to step simulation
        self._unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self._pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)

        self._seed()

        # to provide observations
        # TODO?: evaluate other topics with lighter messages
        rospy.Subscriber("/gazebo/model_states", ModelStates,
                         self._gazebo_state_callback, queue_size=1)

        # to send out actions
        self.cs_pubs = []
        for topic in cs_topics:
            self.cs_pubs.append(rospy.Publisher(topic, Float32, queue_size=1))

        # to reset to arbitrary initial pose
        self.state_pub = rospy.Publisher('/gazebo/set_model_state', ModelState,
                                         queue_size=1)

        self.init_state = ModelState()
        self.init_state.model_name = 'uav_1'
        self.init_state.reference_frame = 'ground_collision_plane'
        self.init_state.twist.linear.x = 20
        self.init_state.pose.position.z = 50

        self.state = copy.deepcopy(self.init_state)
        self.latest_state_msg = None
        self.last_action = None

        self.time_step = 0.05

        # ail, elev, rud, prop
        self.action_space = spaces.Box(low =np.array([-1.0, -1.0, -1.0, 0]),
                                       high=np.array([+1.0, +1.0, +1.0, 10000]),
                                       dtype=np.float32)

        self.reward_range = (-np.inf, np.inf)

        rospy.wait_for_service('/gazebo/pause_physics')
        rospy.wait_for_service('/gazebo/unpause_physics')
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
        observation = (pose, twist)
        reward = 0
        done = False
        return observation, reward, done


    def _apply_action(self, action):
        ail, elev, rudd, prop  = action
        command = [0, elev, ail, -ail, 0, 0, 0, 0, prop]
        for i in range(len(command)):
            self.cs_pubs[i].publish(Float32(command[i]))


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    # Resets the state of the environment and returns initial observation
    def reset(self):
        self._pause()
        self.state_pub.publish(self.init_state)
        self.state = copy.deepcopy(self.init_state)
        self.latest_state_msg = None
        observation, reward, done  = self._observe()
        return observation, done


    def step(self, action):

        self._apply_action(action)
        self._unpause()
        # wait as the simulation runs,
        # small annoyance: will continue sleeping if rosmaster dies
        rospy.sleep(self.time_step)
        self._freeze()
        observation, reward, done = self._observe()
        info = {}

        return observation, reward, done, info


register(
    id='LearnToSoar-v0',
    entry_point=LearnToSoarEnv
)