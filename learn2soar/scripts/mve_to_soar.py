#!/usr/bin/env python3
import numpy as np

import gym

from mve_policy import MVEPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from mve_ddpg import MVEDDPG

from stable_baselines.common.callbacks import BaseCallback
import tensorflow as tf

from rospy.exceptions import ROSInterruptException
from rospy.service    import ServiceException
import rospkg 

import rotors_gym_envs.l2s_energy_env_v1

env_type = 'energy-v1'
env = gym.make('l2s-'+env_type)

class CustomMVEPolicy(MVEPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomMVEPolicy, self).__init__(*args, **kwargs,
            layer_norm=False,
            feature_extraction="mlp",
            actor_layers=[8],
            critic_layers=[256, 256, 256],
            pred_layers=[256,256,256,256]
            )

l2s_path = rospkg.RosPack().get_path('learn2soar') + "/"

# TODO: hide this ugly piece of code
class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def on_episode_end(self, info) -> bool:
        summary = tf.Summary(value=[])
        self.locals['writer'].add_summary(summary, self.num_timesteps)
        summary = tf.Summary(value=[
            #tf.Summary.Value(tag='env/extracted_energy', simple_value=myenv.extracted_energy),
            tf.Summary.Value(tag='env/positive_power',   simple_value=info['positive_power']),
            tf.Summary.Value(tag='env/duration',         simple_value=info['duration']),
            tf.Summary.Value(tag='env/terminal_energy',  simple_value=info['terminal_energy']),
            tf.Summary.Value(tag='env/max_energy',       simple_value=info['max_energy']),
            tf.Summary.Value(tag='env/mean_energy',      simple_value=info['mean_energy']),
            tf.Summary.Value(tag='env/min_airspeed',     simple_value=info['min_airspeed']),
            tf.Summary.Value(tag='env/avg_rotation',     simple_value=info['avg_rotation']),
            #tf.Summary.Value(tag='env/final_altitude',   simple_value=myenv.final_altitude),
            #tf.Summary.Value(tag='env/final_airspeed',   simple_value=myenv.final_airspeed),
            ])  
        self.locals['writer'].add_summary(summary, self.num_timesteps)
        return True


name = "MVEDDPG.8.256x3.256x4" # <----------------


model_filename = l2s_path + "trained_models/"+env_type+'.'+name
tensorboard_filename = l2s_path + "tb_logs/"+env_type+'easy/'

# the noise objects for DDPG
n_actions = env.action_space.shape[-1]

param_noise = AdaptiveParamNoiseSpec(
    initial_stddev=0.06, 
    desired_action_stddev=0.1, 
    adoption_coefficient=1.01
    )
action_noise = None
#action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.1) * np.ones(n_actions))

model = MVEDDPG.load(model_filename+".int",   
#model = MVEDDPG(   CustomMVEPolicy, 
    env, 
    param_noise=param_noise, 
    action_noise=action_noise,
    actor_lr=1e-7,
    critic_lr=5e-4,
    pred_lr=5e-5,
    gamma=0.98,
    nb_train_steps=35,
    batch_size=256,
    nb_rollout_steps=70,
    verbose=1, 
    tensorboard_log=tensorboard_filename,
    n_cpu_tf_sess=8,
    buffer_size=30000,
    h=4,
    tau = 0.1
    )

pretrain_dynamics = False
if pretrain_dynamics:
    data_name = l2s_path + 'demonstrations/seb_run000.nrm.npz'
    from pred_dataset import ExperienceDataset
    dataset = ExperienceDataset(data_path=data_name, verbose=1,
                            traj_limitation=-1, batch_size=200, train_fraction=0.9, sequential_preprocessing=True)
    model.pretrain_predictor(dataset, n_epochs=250, learning_rate=1e-3, )
    model.save(model_filename+'.predyn')

pretrain_actor = False
if pretrain_actor:

    demo_name = l2s_path + 'demonstrations/seb_run014.nrm.npz'
    from stable_baselines.gail import ExpertDataset
    dataset = ExpertDataset(expert_path=demo_name, verbose=1,
                            traj_limitation=-1, batch_size=128, train_fraction=0.9, sequential_preprocessing=True)
    model.pretrain(dataset, n_epochs=1000, learning_rate=1e-4)
    model.save(model_filename+'.pre')

    try:
        # show off 10 episodes from pretraining only
        for _ in range(5):
                obs = env.reset()
                ep_rew = .0
                done = False
                while not done:
                    action, _states = model.predict(obs)
                    obs, rew, done, _ = env.step(action)
                    ep_rew += rew
                print("Episode reward = %f"%(ep_rew))
    except (ROSInterruptException, ServiceException):
        print("Interrupted, moving on to training")


try:
    model.learn(
        total_timesteps=1000000, 
        callback=TensorboardCallback(),
        tb_log_name=name,
        )
except (ROSInterruptException, ServiceException):
    print("Interrupted, saving model to "+model_filename+'.int')
    model.save(model_filename+'.int')
    model.save(model_filename+'.int', save_buffer=True)
    exit()

model.save(model_filename+'.1M')
model.save(model_filename+'.1M', save_buffer=True)

