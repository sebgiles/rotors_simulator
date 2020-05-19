#!/usr/bin/env python3
from stable_baselines.common.policies import FeedForwardPolicy, MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import A2C

import rotors_gym_envs.learn_to_soar_env_v3
from std_msgs.msg import Float32


from rospy.exceptions import ROSInterruptException
from rospy.service    import ServiceException

from stable_baselines.common.callbacks import BaseCallback
import tensorflow as tf

env = make_vec_env('LearnToSoar-v3')

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

model_filename = "a2c_custom_policy"
tensorboard_filename = "./tb_deeper_policy/"

# Custom MLP policy of three layers of size 128 each
class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[8, 16, 8],
                                                          vf=[8, 16, 8])],
                                           feature_extraction="mlp")
# Use #1 to create a new model, #2 to reload the model from the file
#model = A2C(CustomPolicy,             # 1 
model = A2C.load(model_filename,   # 2
            env, 
            tensorboard_log=tensorboard_filename,
            verbose = 1,
            learning_rate=1e-4,
            gamma=0.99, 
            n_steps=5, 
            vf_coef=0.25, 
            ent_coef=0.01, 
            max_grad_norm=0.5,
            alpha=0.99, 
            epsilon=1e-2, 
            lr_schedule='linear'
        )
try:
    model.learn(total_timesteps=10000000, callback=TensorboardCallback())
except (ROSInterruptException, ServiceException):
    print("Interrupted, saving model")
    model.save(model_filename)
    exit()

model.save(model_filename)
