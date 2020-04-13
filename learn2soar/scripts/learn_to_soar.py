#!/usr/bin/env python3
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import A2C

import rotors_gym_envs.learn_to_soar_env_v2

from rospy.exceptions import ROSInterruptException
from rospy.service    import ServiceException

from stable_baselines.common.callbacks import BaseCallback
import tensorflow as tf

env = make_vec_env('LearnToSoar-v2')

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

model_filename = "a2c_24_soar_autosave"
tensorboard_filename = "./tb_l2s_24/"

# Use #1 to create a new model, #2 to reload the model from the file
model = A2C(MlpPolicy,              # 1 
#model = A2C.load(model_filename,   # 2
            env, 
            tensorboard_log=tensorboard_filename,
            verbose = 1,
            learning_rate=5e-3,
            gamma=0.9, 
            n_steps=2, 
            vf_coef=0.25, 
            ent_coef=0.01, 
            max_grad_norm=0.5,
            alpha=0.99, 
            epsilon=5e-2, 
            lr_schedule='constant'
        )
try:
    model.learn(total_timesteps=1000000, callback=TensorboardCallback())
except (ROSInterruptException, ServiceException):
    print("Interrupted, saving model")
    model.save(model_filename)
    exit()

model.save(model_filename)
