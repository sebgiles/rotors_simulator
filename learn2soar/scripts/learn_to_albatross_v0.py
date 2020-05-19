#!/usr/bin/env python3

from stable_baselines import A2C
from stable_baselines.common.policies import FeedForwardPolicy, MlpPolicy, LstmPolicy
from stable_baselines.common import make_vec_env
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.callbacks import BaseCallback
import tensorflow as tf

from rospy.exceptions import ROSInterruptException
from rospy.service    import ServiceException
import rospkg 

import rotors_gym_envs.learn_to_albatross_env_v0

env = make_vec_env('Albatross-v0')

l2s_path = rospkg.RosPack().get_path('learn2soar') + "/"

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


model_filename = l2s_path + "trained_models/pretr_albatross_v0.1"
#model_filename = l2s_path + "trained_models/albatross_v0.5"
tensorboard_filename = l2s_path + "tb_logs/albatross_v0/"

# Custom MLP policy of three layers of size 128 each
class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[ 32, dict(vf=[32,16], pi=[32,16]) ],                                                        
                                           feature_extraction="mlp")

class CustomLSTMPolicy(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=64, reuse=False, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                         net_arch=[ 32, 'lstm', dict(vf=[32,16], pi=[32,16]) ],
                         layer_norm=True, feature_extraction="mlp", **_kwargs)

# Use #1 to create a new model, #2 to reload the model from the file
#model = A2C(CustomPolicy,           # 1 
#model = A2C(CustomLSTMPolicy,           # 1 
model = A2C.load(model_filename,   # 2
            env, 
            tensorboard_log=tensorboard_filename,
            verbose = 1,
            learning_rate=2e-4,
            gamma=0.995, 
            n_steps=5, 
            lr_schedule='linear'
        )

try:
    model.learn(total_timesteps=10000000, callback=TensorboardCallback())
except (ROSInterruptException, ServiceException):
    print("Interrupted, saving model")
    model.save(model_filename)
    exit()

model.save(model_filename)

