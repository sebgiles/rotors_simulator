from functools import reduce
import os
import time
from collections import deque
import pickle
import warnings

import gym
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
from mpi4py import MPI

from stable_baselines import logger
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.mpi_adam import MpiAdam
from stable_baselines.common.buffers import ReplayBuffer
from stable_baselines.common.math_util import unscale_action, scale_action
from stable_baselines.common.mpi_running_mean_std import RunningMeanStd
from mve_policy import MVEPolicy


def normalize(tensor, stats):
    """
    normalize a tensor using a running mean and std

    :param tensor: (TensorFlow Tensor) the input tensor
    :param stats: (RunningMeanStd) the running mean and std of the input to normalize
    :return: (TensorFlow Tensor) the normalized tensor
    """
    if stats is None:
        return tensor
    return (tensor - stats.mean) / stats.std


def denormalize(tensor, stats):
    """
    denormalize a tensor using a running mean and std

    :param tensor: (TensorFlow Tensor) the normalized tensor
    :param stats: (RunningMeanStd) the running mean and std of the input to normalize
    :return: (TensorFlow Tensor) the restored tensor
    """
    if stats is None:
        return tensor
    return tensor * stats.std + stats.mean


def reduce_std(tensor, axis=None, keepdims=False):
    """
    get the standard deviation of a Tensor

    :param tensor: (TensorFlow Tensor) the input tensor
    :param axis: (int or [int]) the axis to itterate the std over
    :param keepdims: (bool) keep the other dimensions the same
    :return: (TensorFlow Tensor) the std of the tensor
    """
    return tf.sqrt(reduce_var(tensor, axis=axis, keepdims=keepdims))


def reduce_var(tensor, axis=None, keepdims=False):
    """
    get the variance of a Tensor

    :param tensor: (TensorFlow Tensor) the input tensor
    :param axis: (int or [int]) the axis to itterate the variance over
    :param keepdims: (bool) keep the other dimensions the same
    :return: (TensorFlow Tensor) the variance of the tensor
    """
    tensor_mean = tf.reduce_mean(tensor, axis=axis, keepdims=True)
    devs_squared = tf.square(tensor - tensor_mean)
    return tf.reduce_mean(devs_squared, axis=axis, keepdims=keepdims)


def get_target_updates(_vars, target_vars, tau, verbose=0):
    """
    get target update operations

    :param _vars: ([TensorFlow Tensor]) the initial variables
    :param target_vars: ([TensorFlow Tensor]) the target variables
    :param tau: (float) the soft update coefficient (keep old values, between 0 and 1)
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :return: (TensorFlow Operation, TensorFlow Operation) initial update, soft update
    """
    if verbose >= 2:
        logger.info('setting up target updates ...')
    soft_updates = []
    init_updates = []
    assert len(_vars) == len(target_vars)
    for var, target_var in zip(_vars, target_vars):
        if verbose >= 2:
            logger.info('  {} <- {}'.format(target_var.name, var.name))
        init_updates.append(tf.assign(target_var, var))
        soft_updates.append(tf.assign(target_var, (1. - tau) * target_var + tau * var))
    assert len(init_updates) == len(_vars)
    assert len(soft_updates) == len(_vars)
    return tf.group(*init_updates), tf.group(*soft_updates)


def get_perturbable_vars(scope):
    """
    Get the trainable variables that can be perturbed when using
    parameter noise.

    :param scope: (str) tensorflow scope of the variables
    :return: ([tf.Variables])
    """
    return [var for var in tf_util.get_trainable_vars(scope) if 'LayerNorm' not in var.name]


def get_perturbed_actor_updates(actor, perturbed_actor, param_noise_stddev, verbose=0):
    """
    Get the actor update, with noise.

    :param actor: (str) the actor
    :param perturbed_actor: (str) the pertubed actor
    :param param_noise_stddev: (float) the std of the parameter noise
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :return: (TensorFlow Operation) the update function
    """
    assert len(tf_util.get_globals_vars(actor)) == len(tf_util.get_globals_vars(perturbed_actor))
    assert len(get_perturbable_vars(actor)) == len(get_perturbable_vars(perturbed_actor))

    updates = []
    for var, perturbed_var in zip(tf_util.get_globals_vars(actor), tf_util.get_globals_vars(perturbed_actor)):
        if var in get_perturbable_vars(actor):
            if verbose >= 2:
                logger.info('  {} <- {} + noise'.format(perturbed_var.name, var.name))
            # Add Gaussian noise to the parameter
            updates.append(tf.assign(perturbed_var,
                                     var + tf.random_normal(tf.shape(var), mean=0., stddev=param_noise_stddev)))
        else:
            if verbose >= 2:
                logger.info('  {} <- {}'.format(perturbed_var.name, var.name))
            updates.append(tf.assign(perturbed_var, var))
    assert len(updates) == len(tf_util.get_globals_vars(actor))
    return tf.group(*updates)


class MVEDDPG(OffPolicyRLModel):
    """
    Deep Deterministic Policy Gradient (DDPG) model

    DDPG: https://arxiv.org/pdf/1509.02971.pdf

    :param policy: (DDPGPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, LnMlpPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) the discount factor
    :param memory_policy: (ReplayBuffer) the replay buffer
        (if None, default to baselines.deepq.replay_buffer.ReplayBuffer)

        .. deprecated:: 2.6.0
            This parameter will be removed in a future version

    :param eval_env: (Gym Environment) the evaluation environment (can be None)
    :param nb_train_steps: (int) the number of training steps
    :param nb_rollout_steps: (int) the number of rollout steps
    :param nb_eval_steps: (int) the number of evaluation steps
    :param param_noise: (AdaptiveParamNoiseSpec) the parameter noise type (can be None)
    :param action_noise: (ActionNoise) the action noise type (can be None)
    :param param_noise_adaption_interval: (int) apply param noise every N steps
    :param tau: (float) the soft update coefficient (keep old values, between 0 and 1)
    :param normalize_returns: (bool) should the critic output be normalized
    :param normalize_observations: (bool) should the observation be normalized
    :param batch_size: (int) the size of the batch for learning the policy
    :param observation_range: (tuple) the bounding values for the observation
    :param return_range: (tuple) the bounding values for the critic output
    :param critic_l2_reg: (float) l2 regularizer coefficient
    :param actor_lr: (float) the actor learning rate
    :param critic_lr: (float) the critic learning rate
    :param clip_norm: (float) clip the gradients (disabled if None)
    :param reward_scale: (float) the value the reward should be scaled by
    :param render: (bool) enable rendering of the environment
    :param render_eval: (bool) enable rendering of the evaluation environment
    :param memory_limit: (int) the max number of transitions to store, size of the replay buffer

        .. deprecated:: 2.6.0
            Use `buffer_size` instead.

    :param buffer_size: (int) the max number of transitions to store, size of the replay buffer
    :param random_exploration: (float) Probability of taking a random action (as in an epsilon-greedy strategy)
        This is not needed for DDPG normally but can help exploring when using HER + DDPG.
        This hack was present in the original OpenAI Baselines repo (DDPG + HER)
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    :param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.
    :param mve_horizon: (int) The number of steps to be imagined by the  
    """
    def __init__(self, policy, env, gamma=0.99, memory_policy=None, eval_env=None, nb_train_steps=50,
                 nb_rollout_steps=100, nb_eval_steps=100, param_noise=None, action_noise=None,
                 normalize_observations=False, tau=0.001, batch_size=128, param_noise_adaption_interval=50,
                 normalize_returns=False, observation_range=(-np.inf, np.inf), critic_l2_reg=0.,
                 return_range=(-np.inf, np.inf), actor_lr=1e-4, critic_lr=1e-3, pred_lr=1e-3, h=15, clip_norm=None, reward_scale=1.,
                 render=False, render_eval=False, memory_limit=None, buffer_size=50000, random_exploration=0.0,
                 verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None,
                 full_tensorboard_log=False, seed=None, n_cpu_tf_sess=1):

        super(MVEDDPG, self).__init__(policy=policy, env=env, replay_buffer=None,
                                   verbose=verbose, policy_base=MVEDDPG,
                                   requires_vec_env=False, policy_kwargs=policy_kwargs,
                                   seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)

        # Parameters.
        self.gamma = gamma
        self.tau = tau

        # TODO: remove this param in v3.x.x
        if memory_policy is not None:
            warnings.warn("memory_policy will be removed in a future version (v3.x.x) "
                          "it is now ignored and replaced with ReplayBuffer", DeprecationWarning)

        if memory_limit is not None:
            warnings.warn("memory_limit will be removed in a future version (v3.x.x) "
                          "use buffer_size instead", DeprecationWarning)
            buffer_size = memory_limit

        self.normalize_observations = normalize_observations
        self.normalize_returns = normalize_returns
        self.action_noise = action_noise
        self.param_noise = param_noise
        self.return_range = return_range
        self.observation_range = observation_range
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.pred_lr = pred_lr
        self.clip_norm = clip_norm
        self.reward_scale = reward_scale
        self.batch_size = batch_size
        self.critic_l2_reg = critic_l2_reg
        self.eval_env = eval_env
        self.render = render
        self.render_eval = render_eval
        self.nb_eval_steps = nb_eval_steps
        self.param_noise_adaption_interval = param_noise_adaption_interval
        self.nb_train_steps = nb_train_steps
        self.nb_rollout_steps = nb_rollout_steps
        self.memory_limit = memory_limit
        self.buffer_size = buffer_size
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log
        self.random_exploration = random_exploration
        self.h = h

        # init
        self.graph = None
        self.stats_sample = None
        self.replay_buffer = None
        self.policy_tf = None
        self.target_init_updates = None
        self.target_soft_updates = None
        self.critic_loss = None
        self.critic_grads = None
        self.critic_optimizer = None
        self.sess = None
        self.stats_ops = None
        self.stats_names = None
        self.perturbed_actor_tf = None
        self.perturb_policy_ops = None
        self.perturb_adaptive_policy_ops = None
        self.adaptive_policy_distance = None
        self.actor_loss = None
        self.actor_grads = None
        self.actor_optimizer = None
        self.old_std = None
        self.old_mean = None
        self.renormalize_q_outputs_op = None
        self.obs_rms = None
        self.ret_rms = None
        self.target_policy = None
        self.pred_loss = None
        self.pred_grads = None
        self.pred_optimizer = None
        self.actor_tf = None
        self.target_actor = None
        self.normalized_critic_tf = None
        self.critic_tf = None
        self.normalized_critic_with_actor_tf = None
        self.critic_with_actor_tf = None
        self.target_q = None
        self.obs_train = None
        self.action_train_ph = None
        self.obs_target = None
        self.action_target = None
        self.obs_noise = None
        self.action_noise_ph = None
        self.obs_adapt_noise = None
        self.action_adapt_noise = None
        self.terminals_ph = None
        self.rewards = None
        self.actions = None
        self.critic_target = None
        self.param_noise_stddev = None
        self.param_noise_actor = None
        self.adaptive_param_noise_actor = None
        self.params = None
        self.summary = None
        self.tb_seen_steps = None
        self.pred_rew = None
        self.pred_obs = None
        self.target_params = None
        self.obs_rms_params = None
        self.ret_rms_params = None
        self.pred_delta_normed = None
        self.pred_rew_normed = None
        self.pred_obs_error_vec = None
        self.pred_done = None

        if _init_setup_model:
            self.setup_model()

    def _get_pretrain_placeholders(self):
        policy = self.policy_tf
        # Rescale
        deterministic_action = unscale_action(self.action_space, self.actor_tf)
        return policy.obs_ph, self.actions, deterministic_action

    def setup_model(self):
        with SetVerbosity(self.verbose):

            assert isinstance(self.action_space, gym.spaces.Box), \
                "Error: DDPG cannot output a {} action space, only spaces.Box is supported.".format(self.action_space)
            assert issubclass(self.policy, MVEPolicy), "Error: the input policy for the DDPG model must be " \
                                                        "an instance of DDPGPolicy."

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.set_random_seed(self.seed)
                self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)

                self.replay_buffer = ReplayBuffer(self.buffer_size)

                with tf.variable_scope("input", reuse=False):
                    # Observation normalization.
                    if self.normalize_observations:
                        with tf.variable_scope('obs_rms'):
                            self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
                    else:
                        self.obs_rms = None

                    # Return normalization.
                    if self.normalize_returns:
                        with tf.variable_scope('ret_rms'):
                            self.ret_rms = RunningMeanStd()
                    else:
                        self.ret_rms = None

                    self.policy_tf = self.policy(self.sess, self.observation_space, self.action_space, 1, 1, None,
                                                 **self.policy_kwargs)

                    # Create target networks.
                    self.target_policy = self.policy(self.sess, self.observation_space, self.action_space, 1, 1, None,
                                                     **self.policy_kwargs)
                    self.obs_target = self.target_policy.obs_ph
                    self.action_target = self.target_policy.action_ph

                    normalized_obs = tf.clip_by_value(normalize(self.policy_tf.processed_obs, self.obs_rms),
                                                       self.observation_range[0], self.observation_range[1])
                    normalized_target_obs = tf.clip_by_value(normalize(self.target_policy.processed_obs, self.obs_rms),
                                                       self.observation_range[0], self.observation_range[1])

                    if self.param_noise is not None:
                        # Configure perturbed actor.
                        self.param_noise_actor = self.policy(self.sess, self.observation_space, self.action_space, 1, 1,
                                                             None, **self.policy_kwargs)
                        self.obs_noise = self.param_noise_actor.obs_ph
                        self.action_noise_ph = self.param_noise_actor.action_ph

                        # Configure separate copy for stddev adoption.
                        self.adaptive_param_noise_actor = self.policy(self.sess, self.observation_space,
                                                                      self.action_space, 1, 1, None,
                                                                      **self.policy_kwargs)
                        self.obs_adapt_noise = self.adaptive_param_noise_actor.obs_ph
                        self.action_adapt_noise = self.adaptive_param_noise_actor.action_ph

                    # Inputs.
                    self.obs_train = self.policy_tf.obs_ph
                    self.action_train_ph = self.policy_tf.action_ph
                    self.terminals_ph = tf.placeholder(tf.float32, shape=(None, 1), name='terminals')
                    self.rewards = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
                    self.actions = tf.placeholder(tf.float32, shape=(None,) + self.action_space.shape, name='actions')
                    self.critic_target = tf.placeholder(tf.float32, shape=(None, 1), name='critic_target')
                    self.param_noise_stddev = tf.placeholder(tf.float32, shape=(), name='param_noise_stddev')

                # Create networks and core TF parts that are shared across setup parts.
                with tf.variable_scope("model", reuse=False):
                    self.actor_tf = self.policy_tf.make_actor(normalized_obs)
                    self.normalized_critic_tf = self.policy_tf.make_critic(normalized_obs, self.actions)
                    self.normalized_critic_with_actor_tf = self.policy_tf.make_critic(normalized_obs,
                                                                                      self.actor_tf,
                                                                                      reuse=True)
                # Noise setup
                if self.param_noise is not None:
                    self._setup_param_noise(normalized_obs)

                with tf.variable_scope("target", reuse=False):
                    self.target_actor = self.target_policy.make_actor(normalized_target_obs)
                    critic_target = self.target_policy.make_critic(normalized_target_obs, self.target_actor)

                with tf.variable_scope("predictor", reuse=False):
                    self.pred_delta_normed, self.pred_rew_normed, self.pred_done = self.target_policy.make_predictor(self.target_policy.obs_ph, self.target_policy.action_ph)
                    pred_delta    = self.pred_delta_normed * np.array([ 0.27281991, 
                                                                        0.09118623, 
                                                                        0.39521678, 
                                                                        0.1420702,  
                                                                        0.35845451])
                    temp          = self.target_policy.next_obs_ph - self.obs_target
                    true_delta    = tf.concat([temp[:,:2], (temp[:,2:5] + 1.0) % 2.0 - 1.0],1)
                    temp          = self.obs_target + pred_delta
                    self.pred_obs = tf.concat([temp[:,:2], (temp[:,2:5] + 1.0) % 2.0 - 1.0],1)
                    self.pred_rew = (self.pred_rew_normed * 20.66) + 12.06

                    rrms = lambda x_hat, x : tf.sqrt(tf.reduce_mean(tf.square(x_hat-x)))
                    
                    tf.summary.scalar('z_error',     rrms(pred_delta[:,0], true_delta[:,0]))
                    tf.summary.scalar('v_error',     rrms(pred_delta[:,1], true_delta[:,1]))
                    tf.summary.scalar('yaw_error',   rrms(pred_delta[:,2], true_delta[:,2]))
                    tf.summary.scalar('pitch_error', rrms(pred_delta[:,3], true_delta[:,3]))
                    tf.summary.scalar('roll_error',  rrms(pred_delta[:,4], true_delta[:,4]))
                    tf.summary.scalar('rew_error',   rrms(self.pred_rew, self.rewards))
                    tf.summary.scalar('term_error',  tf.reduce_mean(tf.abs(self.terminals_ph-self.pred_done)))

                with tf.variable_scope("loss", reuse=False):
                    self.critic_tf = denormalize(
                        tf.clip_by_value(self.normalized_critic_tf, self.return_range[0], self.return_range[1]),
                        self.ret_rms)

                    self.critic_with_actor_tf = denormalize(
                        tf.clip_by_value(self.normalized_critic_with_actor_tf,
                                         self.return_range[0], self.return_range[1]),
                        self.ret_rms)

                    self.target_q = denormalize(critic_target, self.ret_rms)

                    tf.summary.scalar('critic_target', tf.reduce_mean(self.critic_target))
                    if self.full_tensorboard_log:
                        tf.summary.histogram('critic_target', self.critic_target)

                    self._setup_stats()
                    self._setup_target_network_updates()

                with tf.variable_scope("input_info", reuse=False):
                    tf.summary.scalar('rewards', tf.reduce_mean(self.rewards))
                    tf.summary.scalar('param_noise_stddev', tf.reduce_mean(self.param_noise_stddev))

                    if self.full_tensorboard_log:
                        tf.summary.histogram('rewards', self.rewards)
                        tf.summary.histogram('param_noise_stddev', self.param_noise_stddev)
                        if len(self.observation_space.shape) == 3 and self.observation_space.shape[0] in [1, 3, 4]:
                            tf.summary.image('observation', self.obs_train)
                        else:
                            tf.summary.histogram('observation', self.obs_train)

                with tf.variable_scope("Adam_mpi", reuse=False):
                    self._setup_actor_optimizer()
                    self._setup_critic_optimizer()
                    self._setup_pred_optimizer()
                    tf.summary.scalar('actor_loss', self.actor_loss)
                    tf.summary.scalar('critic_loss', self.critic_loss)
                    tf.summary.scalar('pred_loss', self.pred_loss)

                self.params = tf_util.get_trainable_vars("model") \
                    + tf_util.get_trainable_vars('noise/') + tf_util.get_trainable_vars('noise_adapt/')\
                    + tf_util.get_trainable_vars('predictor/')

                self.target_params = tf_util.get_trainable_vars("target")
                self.obs_rms_params = [var for var in tf.global_variables()
                                       if "obs_rms" in var.name]
                self.ret_rms_params = [var for var in tf.global_variables()
                                       if "ret_rms" in var.name]

                with self.sess.as_default():
                    self._initialize(self.sess)

                self.summary = tf.summary.merge_all()

    def _setup_target_network_updates(self):
        """
        set the target update operations
        """
        init_updates, soft_updates = get_target_updates(tf_util.get_trainable_vars('model/'),
                                                        tf_util.get_trainable_vars('target/'), self.tau,
                                                        self.verbose)
        self.target_init_updates = init_updates
        self.target_soft_updates = soft_updates

    def _setup_param_noise(self, normalized_obs):
        """
        Setup the parameter noise operations

        :param normalized_obs: (TensorFlow Tensor) the normalized observation
        """
        assert self.param_noise is not None

        with tf.variable_scope("noise", reuse=False):
            self.perturbed_actor_tf = self.param_noise_actor.make_actor(normalized_obs)

        with tf.variable_scope("noise_adapt", reuse=False):
            adaptive_actor_tf = self.adaptive_param_noise_actor.make_actor(normalized_obs)

        with tf.variable_scope("noise_update_func", reuse=False):
            if self.verbose >= 2:
                logger.info('setting up param noise')
            self.perturb_policy_ops = get_perturbed_actor_updates('model/pi/', 'noise/pi/', self.param_noise_stddev,
                                                                  verbose=self.verbose)

            self.perturb_adaptive_policy_ops = get_perturbed_actor_updates('model/pi/', 'noise_adapt/pi/',
                                                                           self.param_noise_stddev,
                                                                           verbose=self.verbose)
            self.adaptive_policy_distance = tf.sqrt(tf.reduce_mean(tf.square(self.actor_tf - adaptive_actor_tf)))

    def _setup_actor_optimizer(self):
        """
        setup the optimizer for the actor
        """
        if self.verbose >= 2:
            logger.info('setting up actor optimizer')
        self.actor_loss = -tf.reduce_mean(self.critic_with_actor_tf)
        actor_shapes = [var.get_shape().as_list() for var in tf_util.get_trainable_vars('model/pi/')]
        actor_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in actor_shapes])
        if self.verbose >= 2:
            logger.info('  actor shapes: {}'.format(actor_shapes))
            logger.info('  actor params: {}'.format(actor_nb_params))
        self.actor_grads = tf_util.flatgrad(self.actor_loss, tf_util.get_trainable_vars('model/pi/'),
                                            clip_norm=self.clip_norm)
        self.actor_optimizer = MpiAdam(var_list=tf_util.get_trainable_vars('model/pi/'), beta1=0.9, beta2=0.999,
                                       epsilon=1e-08)

    def _setup_pred_optimizer(self):
        """
        setup the optimizer for the predictor
        """
        if self.verbose >= 2:
            logger.info('setting up actor optimizer')
        temp = self.target_policy.next_obs_ph - self.target_policy.obs_ph
        true_delta = tf.concat([ 
                temp[:,:2],
                (temp[:,2:5] + 1.0) % 2.0 - 1.0,
            ],1)
        true_delta_normed = true_delta / np.array([0.27281991, 0.09118623, 0.39521678, 0.1420702,  0.35845451])
        true_rew_normed = (self.rewards - 12.06)/20.66
        self.pred_loss =  tf.reduce_mean(tf.square(tf.concat([
            self.pred_delta_normed - true_delta_normed, 
            self.pred_rew_normed - true_rew_normed, 
            ], 1))) + tf.losses.log_loss(self.terminals_ph, self.pred_done)

        pred_shapes = [var.get_shape().as_list() for var in tf_util.get_trainable_vars('predictor/pred/')]
        pred_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in pred_shapes])
        if self.verbose >= 2:
            logger.info('  pred shapes: {}'.format(pred_shapes))
            logger.info('  pred params: {}'.format(pred_nb_params))
        self.pred_grads = tf_util.flatgrad(self.pred_loss, tf_util.get_trainable_vars('predictor/pred/'),
                                            clip_norm=self.clip_norm)
        self.pred_optimizer = MpiAdam(var_list=tf_util.get_trainable_vars('predictor/pred/'), beta1=0.9, beta2=0.999,
                                       epsilon=1e-08)

    def _setup_critic_optimizer(self):
        """
        setup the optimizer for the critic
        """
        if self.verbose >= 2:
            logger.info('setting up critic optimizer')
        normalized_critic_target_tf = tf.clip_by_value(normalize(self.critic_target, self.ret_rms),
                                                       self.return_range[0], self.return_range[1])
        self.critic_loss = tf.reduce_mean(tf.square(self.normalized_critic_tf - normalized_critic_target_tf))
        if self.critic_l2_reg > 0.:
            critic_reg_vars = [var for var in tf_util.get_trainable_vars('model/qf/')
                               if 'bias' not in var.name and 'qf_output' not in var.name and 'b' not in var.name]
            if self.verbose >= 2:
                for var in critic_reg_vars:
                    logger.info('  regularizing: {}'.format(var.name))
                logger.info('  applying l2 regularization with {}'.format(self.critic_l2_reg))
            critic_reg = tc.layers.apply_regularization(
                tc.layers.l2_regularizer(self.critic_l2_reg),
                weights_list=critic_reg_vars
            )
            self.critic_loss += critic_reg
        critic_shapes = [var.get_shape().as_list() for var in tf_util.get_trainable_vars('model/qf/')]
        critic_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in critic_shapes])
        if self.verbose >= 2:
            logger.info('  critic shapes: {}'.format(critic_shapes))
            logger.info('  critic params: {}'.format(critic_nb_params))
        self.critic_grads = tf_util.flatgrad(self.critic_loss, tf_util.get_trainable_vars('model/qf/'),
                                             clip_norm=self.clip_norm)
        self.critic_optimizer = MpiAdam(var_list=tf_util.get_trainable_vars('model/qf/'), beta1=0.9, beta2=0.999,
                                        epsilon=1e-08)


    def _setup_stats(self):
        """
        Setup the stat logger for DDPG.
        """
        ops = [
            tf.reduce_mean(self.critic_tf),
            reduce_std(self.critic_tf),
            tf.reduce_mean(self.critic_with_actor_tf),
            reduce_std(self.critic_with_actor_tf),
            tf.reduce_mean(self.actor_tf),
            reduce_std(self.actor_tf)
        ]
        names = [
            'reference_Q_mean',
            'reference_Q_std',
            'reference_actor_Q_mean',
            'reference_actor_Q_std',
            'reference_action_mean',
            'reference_action_std'
        ]

        if self.normalize_returns:
            ops += [self.ret_rms.mean, self.ret_rms.std]
            names += ['ret_rms_mean', 'ret_rms_std']

        if self.normalize_observations:
            ops += [tf.reduce_mean(self.obs_rms.mean), tf.reduce_mean(self.obs_rms.std)]
            names += ['obs_rms_mean', 'obs_rms_std']

        if self.param_noise:
            ops += [tf.reduce_mean(self.perturbed_actor_tf), reduce_std(self.perturbed_actor_tf)]
            names += ['reference_perturbed_action_mean', 'reference_perturbed_action_std']

        self.stats_ops = ops
        self.stats_names = names

    def _policy(self, obs, apply_noise=True, compute_q=True):
        """
        Get the actions and critic output, from a given observation

        :param obs: ([float] or [int]) the observation
        :param apply_noise: (bool) enable the noise
        :param compute_q: (bool) compute the critic output
        :return: ([float], float) the action and critic value
        """
        obs = np.array(obs).reshape((-1,) + self.observation_space.shape)
        feed_dict = {self.obs_train: obs}
        if self.param_noise is not None and apply_noise:
            actor_tf = self.perturbed_actor_tf
            feed_dict[self.obs_noise] = obs
        else:
            actor_tf = self.actor_tf

        if compute_q:
            action, q_value = self.sess.run([actor_tf, self.critic_with_actor_tf], feed_dict=feed_dict)
        else:
            action = self.sess.run(actor_tf, feed_dict=feed_dict)
            q_value = None

        action = action.flatten()
        if self.action_noise is not None and apply_noise:
            noise = self.action_noise()
            action += noise
        action = np.clip(action, -1, 1)
        return action, q_value

    def _store_transition(self, obs, action, reward, next_obs, done):
        """
        Store a transition in the replay buffer

        :param obs: ([float] or [int]) the last observation
        :param action: ([float]) the action
        :param reward: (float] the reward
        :param next_obs: ([float] or [int]) the current observation
        :param done: (bool) Whether the episode is over
        """
        reward *= self.reward_scale
        self.replay_buffer.add(obs, action, reward, next_obs, float(done))
        if self.normalize_observations:
            self.obs_rms.update(np.array([obs]))

    def _train_step(self, step, writer, log=False):
        """
        run a step of training from batch

        :param step: (int) the current step iteration
        :param writer: (TensorFlow Summary.writer) the writer for tensorboard
        :param log: (bool) whether or not to log to metadata
        :return: (float, float) critic loss, actor loss
        """
        # Get a batch
        obs, actions, rewards, next_obs, terminals = self.replay_buffer.sample(batch_size=self.batch_size,
                                                                               env=self._vec_normalize_env)
        # Reshape to match previous behavior and placeholder shape
        rewards = rewards.reshape(-1, 1)
        terminals = terminals.reshape(-1, 1)

        h = self.h
        imagined_states         = np.zeros([self.batch_size, h+2, self.observation_space.shape[0]])
        imagined_rewards        = np.zeros([self.batch_size, h+2])
        imagined_actions        = np.zeros([self.batch_size, h+1, self.action_space.shape[0]])
        imagined_states[:,0]  = obs
        imagined_rewards[:,0] = rewards[:,0]
        imagined_actions[:,0] = actions
        imagined_states[:,1]  = next_obs
        prev_done = np.copy(terminals)
        for i in range(1,h+1):
            next_action = self.sess.run(self.target_actor, feed_dict={
                self.obs_target: imagined_states[:,i],
            })
            pred_obs, pred_rew, new_pred_done = self.sess.run([self.pred_obs, self.pred_rew, self.pred_done], feed_dict={
                self.obs_target: imagined_states[:,i],
                self.target_policy.action_ph: next_action,
            })

            # keep last state/action in rows which previously reached a termination
            imagined_actions[:,i] =  next_action*(1-prev_done) + prev_done*imagined_actions[:,i-1]
            imagined_states[:,i+1] =    pred_obs*(1-prev_done) + prev_done*imagined_states[:,i]
            # freeze rew to 0 in rows which reach a termination
            imagined_rewards[:,i] = (pred_rew*(1-prev_done))[:,0]  
            # predicted states after terminations are terminations
            prev_done = np.max([prev_done, new_pred_done],axis=0)

        terminal_q = self.sess.run(self.target_q, feed_dict={
            self.obs_target: imagined_states[:,h+1],
        })

        imagined_rewards[:,h+1] = (terminal_q*(1-prev_done))[:,0]

        gamma_powers = np.tril(self.gamma**np.matmul(np.tri(h+2,k=-1),np.tri(h+2)))[:,:-1]
        
        target_q = np.reshape(np.matmul(imagined_rewards, gamma_powers), [self.batch_size*(h+1), 1])
        imagined_actions = np.reshape(imagined_actions, [self.batch_size*(h+1), 2])
        imagined_states = np.reshape(imagined_states[:,:-1], [self.batch_size*(h+1), 5])

        # Get all gradients and perform a synced update.
        ops = [self.actor_grads, self.actor_loss, self.critic_grads, self.critic_loss, self.pred_grads]
        td_map = {
            self.obs_train: imagined_states, # inputs to actor, critic, and critic_with_actor
            self.obs_target: obs, #inputs to critic_target and to target_actor 
            self.target_policy.action_ph: actions,
            self.target_policy.next_obs_ph: next_obs,
            self.actions: imagined_actions,             # inputs to the critic  <<< ***
            self.action_train_ph: actions,              # reference for actor loss
            self.terminals_ph: terminals,
            self.rewards: rewards,
            self.critic_target: target_q,      # compared to the critic to evaluate loss
            self.param_noise_stddev: 0 if self.param_noise is None else self.param_noise.current_stddev
        }
        if writer is not None:
            # run loss backprop with summary if the step_id was not already logged (can happen with the right
            # parameters as the step value is only an estimate)
            if self.full_tensorboard_log and log and step not in self.tb_seen_steps:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, actor_grads, actor_loss, critic_grads, critic_loss, pred_grads = \
                    self.sess.run([self.summary] + ops, td_map, options=run_options, run_metadata=run_metadata)

                writer.add_run_metadata(run_metadata, 'step%d' % step)
                self.tb_seen_steps.append(step)
            else:
                summary, actor_grads, actor_loss, critic_grads, critic_loss, pred_grads = self.sess.run([self.summary] + ops,
                                                                                            td_map)
            writer.add_summary(summary, step)
        else:
            actor_grads, actor_loss, critic_grads, critic_loss, pred_grads = self.sess.run(ops, td_map)

        self.actor_optimizer.update(actor_grads, learning_rate=self.actor_lr)
        self.critic_optimizer.update(critic_grads, learning_rate=self.critic_lr)
        self.pred_optimizer.update(pred_grads, learning_rate=self.pred_lr)

        return critic_loss, actor_loss

    def _initialize(self, sess):
        """
        initialize the model parameters and optimizers

        :param sess: (TensorFlow Session) the current TensorFlow session
        """
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        self.actor_optimizer.sync()
        self.critic_optimizer.sync()
        self.sess.run(self.target_init_updates)

    def _update_target_net(self):
        """
        run target soft update operation
        """
        self.sess.run(self.target_soft_updates)

    def _get_stats(self):
        """
        Get the mean and standard deviation of the model's inputs and outputs

        :return: (dict) the means and stds
        """
        if self.stats_sample is None:
            # Get a sample and keep that fixed for all further computations.
            # This allows us to estimate the change in value for the same set of inputs.
            obs, actions, rewards, next_obs, terminals = self.replay_buffer.sample(batch_size=self.batch_size,
                                                                                   env=self._vec_normalize_env)
            self.stats_sample = {
                'obs': obs,
                'actions': actions,
                'rewards': rewards,
                'next_obs': next_obs,
                'terminals': terminals
            }

        feed_dict = {
            self.actions: self.stats_sample['actions']
        }

        for placeholder in [self.action_train_ph, self.action_target, self.action_adapt_noise, self.action_noise_ph]:
            if placeholder is not None:
                feed_dict[placeholder] = self.stats_sample['actions']

        for placeholder in [self.obs_train, self.obs_target, self.obs_adapt_noise, self.obs_noise]:
            if placeholder is not None:
                feed_dict[placeholder] = self.stats_sample['obs']

        values = self.sess.run(self.stats_ops, feed_dict=feed_dict)

        names = self.stats_names[:]
        assert len(names) == len(values)
        stats = dict(zip(names, values))

        if self.param_noise is not None:
            stats = {**stats, **self.param_noise.get_stats()}

        return stats

    def _adapt_param_noise(self):
        """
        calculate the adaptation for the parameter noise

        :return: (float) the mean distance for the parameter noise
        """
        if self.param_noise is None:
            return 0.

        # Perturb a separate copy of the policy to adjust the scale for the next "real" perturbation.
        obs, *_ = self.replay_buffer.sample(batch_size=self.batch_size, env=self._vec_normalize_env)
        self.sess.run(self.perturb_adaptive_policy_ops, feed_dict={
            self.param_noise_stddev: self.param_noise.current_stddev,
        })
        distance = self.sess.run(self.adaptive_policy_distance, feed_dict={
            self.obs_adapt_noise: obs, self.obs_train: obs,
            self.param_noise_stddev: self.param_noise.current_stddev,
        })

        mean_distance = MPI.COMM_WORLD.allreduce(distance, op=MPI.SUM) / MPI.COMM_WORLD.Get_size()
        self.param_noise.adapt(mean_distance)
        return mean_distance

    def _reset(self):
        """
        Reset internal state after an episode is complete.
        """
        if self.action_noise is not None:
            self.action_noise.reset()
        if self.param_noise is not None:
            self.sess.run(self.perturb_policy_ops, feed_dict={
                self.param_noise_stddev: self.param_noise.current_stddev,
            })

    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="DDPG",
              reset_num_timesteps=True, replay_wrapper=None):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        callback = self._init_callback(callback)

        if replay_wrapper is not None:
            self.replay_buffer = replay_wrapper(self.replay_buffer)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn()

            # a list for tensorboard logging, to prevent logging with the same step number, if it already occured
            self.tb_seen_steps = []

            rank = MPI.COMM_WORLD.Get_rank()

            if self.verbose >= 2:
                logger.log('Using agent with the following configuration:')
                logger.log(str(self.__dict__.items()))

            eval_episode_rewards_history = deque(maxlen=100)
            episode_rewards_history = deque(maxlen=100)
            episode_successes = []


            with self.sess.as_default(), self.graph.as_default():
                # Prepare everything.
                self._reset()
                obs = self.env.reset()
                # Retrieve unnormalized observation for saving into the buffer
                if self._vec_normalize_env is not None:
                    obs_ = self._vec_normalize_env.get_original_obs().squeeze()
                eval_obs = None
                if self.eval_env is not None:
                    eval_obs = self.eval_env.reset()
                episode_reward = 0.
                episode_step = 0
                episodes = 0
                step = 0
                total_steps = 0

                start_time = time.time()

                epoch_episode_rewards = []
                epoch_episode_steps = []
                epoch_actor_losses = []
                epoch_critic_losses = []
                epoch_adaptive_distances = []
                eval_episode_rewards = []
                eval_qs = []
                epoch_actions = []
                epoch_qs = []
                epoch_episodes = 0
                epoch = 0

                callback.on_training_start(locals(), globals())

                while True:
                    for _ in range(log_interval):
                        callback.on_rollout_start()
                        # Perform rollouts.
                        for _ in range(self.nb_rollout_steps):

                            if total_steps >= total_timesteps:
                                callback.on_training_end()
                                return self

                            # Predict next action.
                            action, q_value = self._policy(obs, apply_noise=True, compute_q=True)
                            assert action.shape == self.env.action_space.shape

                            # Execute next action.
                            if rank == 0 and self.render:
                                self.env.render()

                            # Randomly sample actions from a uniform distribution
                            # with a probability self.random_exploration (used in HER + DDPG)
                            if np.random.rand() < self.random_exploration:
                                # actions sampled from action space are from range specific to the environment
                                # but algorithm operates on tanh-squashed actions therefore simple scaling is used
                                unscaled_action = self.action_space.sample()
                                action = scale_action(self.action_space, unscaled_action)
                            else:
                                # inferred actions need to be transformed to environment action_space before stepping
                                unscaled_action = unscale_action(self.action_space, action)

                            new_obs, reward, done, info = self.env.step(unscaled_action)

                            self.num_timesteps += 1

                            if callback.on_step() is False:
                                callback.on_training_end()
                                return self

                            step += 1
                            total_steps += 1
                            if rank == 0 and self.render:
                                self.env.render()

                            # Book-keeping.
                            epoch_actions.append(action)
                            epoch_qs.append(q_value)

                            # Store only the unnormalized version
                            if self._vec_normalize_env is not None:
                                new_obs_ = self._vec_normalize_env.get_original_obs().squeeze()
                                reward_ = self._vec_normalize_env.get_original_reward().squeeze()
                            else:
                                # Avoid changing the original ones
                                obs_, new_obs_, reward_ = obs, new_obs, reward

                            self._store_transition(obs_, action, reward_, new_obs_, done)
                            obs = new_obs
                            # Save the unnormalized observation
                            if self._vec_normalize_env is not None:
                                obs_ = new_obs_

                            episode_reward += reward_
                            episode_step += 1

                            if writer is not None:
                                ep_rew = np.array([reward_]).reshape((1, -1))
                                ep_done = np.array([done]).reshape((1, -1))
                                tf_util.total_episode_reward_logger(self.episode_reward, ep_rew, ep_done,
                                                                    writer, self.num_timesteps)

                            if done:
                                # Episode done.
                                epoch_episode_rewards.append(episode_reward)
                                episode_rewards_history.append(episode_reward)
                                epoch_episode_steps.append(episode_step)
                                episode_reward = 0.
                                episode_step = 0
                                epoch_episodes += 1
                                episodes += 1

                                maybe_is_success = info.get('is_success')
                                if maybe_is_success is not None:
                                    episode_successes.append(float(maybe_is_success))

                                callback.on_episode_end(info)

                                self._reset()
                                if not isinstance(self.env, VecEnv):
                                    obs = self.env.reset()

                        callback.on_rollout_end()
                        # Train.
                        epoch_actor_losses = []
                        epoch_critic_losses = []
                        epoch_adaptive_distances = []
                        for t_train in range(self.nb_train_steps):
                            # Not enough samples in the replay buffer
                            if not self.replay_buffer.can_sample(self.batch_size):
                                break

                            # Adapt param noise, if necessary.
                            if len(self.replay_buffer) >= self.batch_size and \
                                    t_train % self.param_noise_adaption_interval == 0:
                                distance = self._adapt_param_noise()
                                epoch_adaptive_distances.append(distance)

                            # weird equation to deal with the fact the nb_train_steps will be different
                            # to nb_rollout_steps
                            step = (int(t_train * (self.nb_rollout_steps / self.nb_train_steps)) +
                                    self.num_timesteps - self.nb_rollout_steps)

                            critic_loss, actor_loss = self._train_step(step, writer, log=t_train == 0)
                            epoch_critic_losses.append(critic_loss)
                            epoch_actor_losses.append(actor_loss)
                            self._update_target_net()

                        # Evaluate.
                        eval_episode_rewards = []
                        eval_qs = []
                        if self.eval_env is not None:
                            eval_episode_reward = 0.
                            for _ in range(self.nb_eval_steps):
                                if total_steps >= total_timesteps:
                                    return self

                                eval_action, eval_q = self._policy(eval_obs, apply_noise=False, compute_q=True)
                                unscaled_action = unscale_action(self.action_space, eval_action)
                                eval_obs, eval_r, eval_done, _ = self.eval_env.step(unscaled_action)
                                if self.render_eval:
                                    self.eval_env.render()
                                eval_episode_reward += eval_r

                                eval_qs.append(eval_q)
                                if eval_done:
                                    if not isinstance(self.env, VecEnv):
                                        eval_obs = self.eval_env.reset()
                                    eval_episode_rewards.append(eval_episode_reward)
                                    eval_episode_rewards_history.append(eval_episode_reward)
                                    eval_episode_reward = 0.

                    mpi_size = MPI.COMM_WORLD.Get_size()
                    # Log stats.
                    # XXX shouldn't call np.mean on variable length lists
                    duration = time.time() - start_time
                    stats = self._get_stats()
                    combined_stats = stats.copy()
                    combined_stats['rollout/return'] = np.mean(epoch_episode_rewards)
                    combined_stats['rollout/return_history'] = np.mean(episode_rewards_history)
                    combined_stats['rollout/episode_steps'] = np.mean(epoch_episode_steps)
                    combined_stats['rollout/actions_mean'] = np.mean(epoch_actions)
                    combined_stats['rollout/Q_mean'] = np.mean(epoch_qs)
                    combined_stats['train/loss_actor'] = np.mean(epoch_actor_losses)
                    combined_stats['train/loss_critic'] = np.mean(epoch_critic_losses)
                    if len(epoch_adaptive_distances) != 0:
                        combined_stats['train/param_noise_distance'] = np.mean(epoch_adaptive_distances)
                    combined_stats['total/duration'] = duration
                    combined_stats['total/steps_per_second'] = float(step) / float(duration)
                    combined_stats['total/episodes'] = episodes
                    combined_stats['rollout/episodes'] = epoch_episodes
                    combined_stats['rollout/actions_std'] = np.std(epoch_actions)
                    # Evaluation statistics.
                    if self.eval_env is not None:
                        combined_stats['eval/return'] = np.mean(eval_episode_rewards)
                        combined_stats['eval/return_history'] = np.mean(eval_episode_rewards_history)
                        combined_stats['eval/Q'] = np.mean(eval_qs)
                        combined_stats['eval/episodes'] = len(eval_episode_rewards)

                    def as_scalar(scalar):
                        """
                        check and return the input if it is a scalar, otherwise raise ValueError

                        :param scalar: (Any) the object to check
                        :return: (Number) the scalar if x is a scalar
                        """
                        if isinstance(scalar, np.ndarray):
                            assert scalar.size == 1
                            return scalar[0]
                        elif np.isscalar(scalar):
                            return scalar
                        else:
                            raise ValueError('expected scalar, got %s' % scalar)

                    combined_stats_sums = MPI.COMM_WORLD.allreduce(
                        np.array([as_scalar(x) for x in combined_stats.values()]))
                    combined_stats = {k: v / mpi_size for (k, v) in zip(combined_stats.keys(), combined_stats_sums)}

                    # Total statistics.
                    combined_stats['total/epochs'] = epoch + 1
                    combined_stats['total/steps'] = step

                    for key in sorted(combined_stats.keys()):
                        logger.record_tabular(key, combined_stats[key])
                    if len(episode_successes) > 0:
                        logger.logkv("success rate", np.mean(episode_successes[-100:]))
                    logger.dump_tabular()
                    logger.info('')
                    logdir = logger.get_dir()
                    if rank == 0 and logdir:
                        if hasattr(self.env, 'get_state'):
                            with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as file_handler:
                                pickle.dump(self.env.get_state(), file_handler)
                        if self.eval_env and hasattr(self.eval_env, 'get_state'):
                            with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as file_handler:
                                pickle.dump(self.eval_env.get_state(), file_handler)

    def predict(self, observation, state=None, mask=None, deterministic=True):
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        actions, _, = self._policy(observation, apply_noise=not deterministic, compute_q=False)
        actions = actions.reshape((-1,) + self.action_space.shape)  # reshape to the correct action shape
        actions = unscale_action(self.action_space, actions)  # scale the output for the prediction

        if not vectorized_env:
            actions = actions[0]

        return actions, None

    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        _ = np.array(observation)

        if actions is not None:
            raise ValueError("Error: DDPG does not have action probabilities.")

        # here there are no action probabilities, as DDPG does not use a probability distribution
        warnings.warn("Warning: action probability is meaningless for DDPG. Returning None")
        return None

    def get_parameter_list(self):
        return (self.params +
                self.target_params +
                self.obs_rms_params +
                self.ret_rms_params)

    def save(self, save_path, cloudpickle=False, save_buffer=False):
        data = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "nb_eval_steps": self.nb_eval_steps,
            "param_noise_adaption_interval": self.param_noise_adaption_interval,
            "nb_train_steps": self.nb_train_steps,
            "nb_rollout_steps": self.nb_rollout_steps,
            "verbose": self.verbose,
            "param_noise": self.param_noise,
            "action_noise": self.action_noise,
            "gamma": self.gamma,
            "tau": self.tau,
            "normalize_returns": self.normalize_returns,
            "normalize_observations": self.normalize_observations,
            "batch_size": self.batch_size,
            "observation_range": self.observation_range,
            "return_range": self.return_range,
            "critic_l2_reg": self.critic_l2_reg,
            "actor_lr": self.actor_lr,
            "critic_lr": self.critic_lr,
            "clip_norm": self.clip_norm,
            "reward_scale": self.reward_scale,
            "memory_limit": self.memory_limit,
            "buffer_size": self.buffer_size,
            "random_exploration": self.random_exploration,
            "policy": self.policy,
            "n_envs": self.n_envs,
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs,
        }

        if save_buffer:
            data["replay_buffer"] = self.replay_buffer

        params_to_save = self.get_parameters()

        self._save_to_file(save_path,
                           data=data,
                           params=params_to_save,
                           cloudpickle=cloudpickle)

    @classmethod
    def load(cls, load_path, env=None, custom_objects=None, **kwargs):
        data, params = cls._load_from_file(load_path, custom_objects=custom_objects)

        if 'policy_kwargs' in kwargs and kwargs['policy_kwargs'] != data['policy_kwargs']:
            raise ValueError("The specified policy kwargs do not equal the stored policy kwargs. "
                             "Stored kwargs: {}, specified kwargs: {}".format(data['policy_kwargs'],
                                                                              kwargs['policy_kwargs']))

        model = cls(None, env, _init_setup_model=False)
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model.set_env(env)
        model.setup_model()
        # Patch for version < v2.6.0, duplicated keys where saved
        if len(params) > len(model.get_parameter_list()):
            n_params = len(model.params)
            n_target_params = len(model.target_params)
            n_normalisation_params = len(model.obs_rms_params) + len(model.ret_rms_params)
            # Check that the issue is the one from
            # https://github.com/hill-a/stable-baselines/issues/363
            assert len(params) == 2 * (n_params + n_target_params) + n_normalisation_params,\
                "The number of parameter saved differs from the number of parameters"\
                " that should be loaded: {}!={}".format(len(params), len(model.get_parameter_list()))
            # Remove duplicates
            params_ = params[:n_params + n_target_params]
            if n_normalisation_params > 0:
                params_ += params[-n_normalisation_params:]
            params = params_
        model.load_parameters(params)

        return model


    def pretrain_predictor(self, dataset, n_epochs=10, learning_rate=1e-4,
                 adam_epsilon=1e-8, val_interval=None):
        """
        Pretrain a model using behavior cloning:
        supervised learning given an expert dataset.

        NOTE: only Box and Discrete spaces are supported for now.

        :param dataset: (ExpertDataset) Dataset manager
        :param n_epochs: (int) Number of iterations on the training set
        :param learning_rate: (float) Learning rate
        :param adam_epsilon: (float) the epsilon value for the adam optimizer
        :param val_interval: (int) Report training and validation losses every n epochs.
            By default, every 10th of the maximum number of epochs.
        :return: (BaseRLModel) the pretrained model
        """
        continuous_actions = isinstance(self.action_space, gym.spaces.Box)

        assert continuous_actions, 'Only Discrete and Box action spaces are supported'

        # Validate the model every 10% of the total number of iteration
        if val_interval is None:
            # Prevent modulo by zero
            if n_epochs < 10:
                val_interval = 1
            else:
                val_interval = 10

        with self.graph.as_default():
            with tf.variable_scope('pretrain_predictor'):

                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=adam_epsilon)
                optim_op = optimizer.minimize(self.pred_loss, var_list=self.params)

            self.sess.run(tf.global_variables_initializer())

        if self.verbose > 0:
            print("Fitting predictive model...")

        for epoch_idx in range(int(n_epochs)):
            train_loss = 0.0
            # Full pass on the training set
            for _ in range(len(dataset.train_loader)):
                obs, actions, next_obs, rewards, done = dataset.get_next_batch('train')
                feed_dict = {
                    self.target_policy.obs_ph: obs,
                    self.target_policy.action_ph: actions,
                    self.target_policy.next_obs_ph: next_obs,
                    self.rewards: rewards,
                    self.terminals_ph: done
                }

                train_loss_, _ = self.sess.run([self.pred_loss, optim_op], feed_dict)
                train_loss += train_loss_

            train_loss /= len(dataset.train_loader)

            if self.verbose > 0 and (epoch_idx + 1) % val_interval == 0:
                val_loss = 0.0
                # Full pass on the validation set
                for _ in range(len(dataset.val_loader)):
                    obs, actions, next_obs, rewards, done = dataset.get_next_batch('val')
                    feed_dict = {self.target_policy.obs_ph: obs,
                    self.target_policy.action_ph: actions,
                    self.target_policy.next_obs_ph: next_obs,
                    self.rewards: rewards,
                    self.terminals_ph: done # assume the recorded dataset does not include reset transitions
                    }
                    val_loss_, = self.sess.run([self.pred_loss], feed_dict)
                    val_loss += val_loss_

                val_loss /= len(dataset.val_loader)
                if self.verbose > 0:
                    print("==== Training progress {:.2f}% ====".format(100 * (epoch_idx + 1) / n_epochs))
                    print('Epoch {}'.format(epoch_idx + 1))
                    print("Training loss: {:.6f}, Validation loss: {:.6f}".format(train_loss, val_loss))
                    print()
            # Free memory
            del obs, actions, next_obs
        if self.verbose > 0:
            print("Pretraining done.")
        return self
