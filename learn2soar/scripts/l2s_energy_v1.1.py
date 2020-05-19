#!/usr/bin/env python3
import numpy as np

import gym

from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines import GAIL

from stable_baselines.common.callbacks import BaseCallback
import tensorflow as tf

from rospy.exceptions import ROSInterruptException
from rospy.service    import ServiceException
import rospkg 

import rotors_gym_envs.l2s_energy_env_v1

env = gym.make('l2s-energy-v1')

l2s_path = rospkg.RosPack().get_path('learn2soar') + "/"

# TODO: hide this ugly piece of code
class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_rollout_end(self) -> bool:
        myenv = env
        summary = tf.Summary(value=[])
        self.locals['writer'].add_summary(summary, self.num_timesteps)
        summary = tf.Summary(value=[
            #tf.Summary.Value(tag='env/extracted_energy', simple_value=myenv.extracted_energy),
            tf.Summary.Value(tag='env/positive_power',   simple_value=myenv.positive_power),
            tf.Summary.Value(tag='env/duration',         simple_value=myenv.duration),
            tf.Summary.Value(tag='env/terminal_energy',  simple_value=myenv.terminal_energy),
            tf.Summary.Value(tag='env/max_energy',       simple_value=myenv.max_energy),
            tf.Summary.Value(tag='env/mean_energy',      simple_value=myenv.mean_energy),
            tf.Summary.Value(tag='env/min_airspeed',     simple_value=myenv.min_airspeed),
            tf.Summary.Value(tag='env/total_rotation',   simple_value=myenv.total_rotation),
            #tf.Summary.Value(tag='env/final_altitude',   simple_value=myenv.final_altitude),
            #tf.Summary.Value(tag='env/final_airspeed',   simple_value=myenv.final_airspeed),
            ])  
        self.locals['writer'].add_summary(summary, self.num_timesteps)
        return True


# Custom MLP policy of two layers of size 16 each
class CustomPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                                net_arch=[32,{'pi':[16], 'vf':[8]}],
                                                feature_extraction="mlp")

model_filename = l2s_path + "trained_models/energy_v1.1"

tensorboard_filename = l2s_path + "tb_logs/energy_v1/"

demo_name = l2s_path + 'demonstrations/seb_run014.npz'
from stable_baselines.gail import ExpertDataset
dataset = ExpertDataset(expert_path=demo_name, verbose=1,
                        traj_limitation=-1, batch_size=64, train_fraction=0.9)

#model = DDPG.load(model_filename+'_pre',   
model = GAIL(   CustomPolicy, 
                env, 
                dataset,
                verbose=1, 
                tensorboard_log=tensorboard_filename,
                n_cpu_tf_sess=8
        )

try:
    model.learn(total_timesteps=500000, callback=TensorboardCallback())
except (ROSInterruptException, ServiceException):
    print("Interrupted, saving model")
    model.save(model_filename)
    exit()

model.save(model_filename)

