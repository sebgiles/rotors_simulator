#!/usr/bin/env python3
from stable_baselines.common.policies import FeedForwardPolicy, MlpPolicy, LstmPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import A2C

import rotors_gym_envs.learn_to_albatross_env_v0
from std_msgs.msg import Float32

from rospy.exceptions import ROSInterruptException
from rospy.service    import ServiceException
import rospkg 

from stable_baselines.common.env_checker import check_env
from stable_baselines.common.callbacks import BaseCallback
import tensorflow as tf

l2s_path = rospkg.RosPack().get_path('learn2soar') + "/"

env = make_vec_env('Albatross-v0')

# TODO: hide this ugly piece of code
class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_rollout_end(self) -> bool:
        myenv = env.envs[0]
        summary = tf.Summary(value=[])
        self.locals['writer'].add_summary(summary, self.num_timesteps)
        summary = tf.Summary(value=[
            tf.Summary.Value(tag='env/extracted_energy', simple_value=myenv.extracted_energy),
            tf.Summary.Value(tag='env/positive_power',   simple_value=myenv.positive_power),
            tf.Summary.Value(tag='env/duration',         simple_value=myenv.duration),
            tf.Summary.Value(tag='env/terminal_energy',  simple_value=myenv.terminal_energy),
            tf.Summary.Value(tag='env/final_altitude',   simple_value=myenv.final_altitude),
            tf.Summary.Value(tag='env/final_airspeed',   simple_value=myenv.final_airspeed),
            ])
        self.locals['writer'].add_summary(summary, self.num_timesteps)
        return True



model_filename = l2s_path + "trained_models/a2c_rnn_albatross_v0"
tensorboard_filename = l2s_path + "tb_logs/tb_albatross_rnn_0/"

# Custom MLP policy of three layers of size 128 each
class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[32, 16, 8],
                                                          vf=[32, 16, 8])],
                                                          
                                           feature_extraction="mlp")
class CustomLSTMPolicy(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=16, reuse=False, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                         net_arch=[8,'lstm', dict(vf=[16,8], pi=[8])],
                         layer_norm=True, feature_extraction="mlp", **_kwargs)

# Use #1 to create a new model, #2 to reload the model from the file
#model = A2C(CustomLSTMPolicy,             # 1 
model = A2C.load(model_filename,   # 2
            env, 
            tensorboard_log=tensorboard_filename,
            verbose = 1,
            learning_rate=2e-4,
            gamma=0.99, 
            n_steps=5, 
            lr_schedule='double_middle_drop'
        )
try:
    model.learn(total_timesteps=10000000, callback=TensorboardCallback())
except (ROSInterruptException, ServiceException):
    print("Interrupted, saving model")
    model.save(model_filename)
    exit()

model.save(model_filename)

