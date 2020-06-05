from stable_baselines.ddpg.policies import *

class MVEPolicy(DDPGPolicy):
    """
    Policy object that implements a DDPG-like actor critic, using a feed forward neural network.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) The size of the Neural network for the policy (if None, default to [64, 64])
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param layer_norm: (bool) enable layer normalisation
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, actor_layers=None, critic_layers=None,
                 cnn_extractor=nature_cnn, feature_extraction="cnn",
                 layer_norm=False, act_fun=tf.nn.relu, pred_layers=[256, 256], **kwargs):
        super(MVEPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                                scale=(feature_extraction == "cnn"))

        self._kwargs_check(feature_extraction, kwargs)
        self.layer_norm = layer_norm
        self.feature_extraction = feature_extraction
        self.cnn_kwargs = kwargs
        self.cnn_extractor = cnn_extractor
        self.reuse = reuse
        self._qvalue = None
        if actor_layers is None:
            actor_layers = [64, 64]        
        if critic_layers is None:
            critic_layers = [64, 64]
        self.actor_layers = actor_layers
        self.critic_layers = critic_layers
        self.pred_layers = pred_layers

        assert len(critic_layers) >= 1, "Error: must have at least one hidden layer for the critic."

        self.activ = act_fun
        self.n_batch = n_batch
        self.next_obs_ph = tf.placeholder(shape=(n_batch,) + ob_space.shape, dtype=ob_space.dtype, name='next_obs')


    def make_predictor(self, obs=None, actions=None, reuse=False, scope="pred"):

        with tf.variable_scope(scope, reuse=reuse):
            ins = tf.concat([obs,actions], 1)
            pred_h = tf.layers.flatten(ins)
            for i, layer_size in enumerate(self.pred_layers):
                pred_h = tf.layers.dense(pred_h, layer_size, name='fc' + str(i), use_bias=True)
                pred_h = tf.nn.relu(pred_h)
            out = tf.layers.dense(pred_h, self.ob_space.shape[0]+1+1, use_bias=True, name=scope,
                                                     kernel_initializer=tf.random_uniform_initializer(minval=-3e-3,
                                                                                                      maxval=3e-3))
            delta_pred = out[:,:-2]
            rew_pred = out[:,-2:-1]
            term_pred = tf.sigmoid(out[:,-1:], name='termination_output')
            # please fit to normalised ground truth

        return delta_pred, rew_pred, term_pred
    

    def make_actor(self, obs=None, reuse=False, scope="pi"):
        if obs is None:
            obs = self.processed_obs

        with tf.variable_scope(scope, reuse=reuse):
            if self.feature_extraction == "cnn":
                pi_h = self.cnn_extractor(obs, **self.cnn_kwargs)
            else:
                pi_h = tf.layers.flatten(obs)
            for i, layer_size in enumerate(self.actor_layers):
                pi_h = tf.layers.dense(pi_h, layer_size, name='fc' + str(i))
                if self.layer_norm:
                    pi_h = tf.contrib.layers.layer_norm(pi_h, center=True, scale=True, use_bias=True)
                pi_h = self.activ(pi_h)
            self.policy = tf.nn.tanh(tf.layers.dense(pi_h, self.ac_space.shape[0], use_bias=True, name=scope,
                                                     kernel_initializer=tf.random_uniform_initializer(minval=-3e-3,
                                                                                                      maxval=3e-3)))
        return self.policy


    def make_critic(self, obs=None, action=None, reuse=False, scope="qf"):
        if obs is None:
            obs = self.processed_obs
        if action is None:
            action = self.action_ph

        with tf.variable_scope(scope, reuse=reuse):
            if self.feature_extraction == "cnn":
                qf_h = self.cnn_extractor(obs, **self.cnn_kwargs)
            else:
                qf_h = tf.layers.flatten(obs)
            for i, layer_size in enumerate(self.critic_layers):
                qf_h = tf.layers.dense(qf_h, layer_size, name='fc' + str(i), use_bias=True)
                if self.layer_norm:
                    qf_h = tf.contrib.layers.layer_norm(qf_h, center=True, scale=True)
                qf_h = self.activ(qf_h)
                if i == 0:
                    qf_h = tf.concat([qf_h, action], axis=-1)

            # the name attribute is used in pop-art normalization
            qvalue_fn = tf.layers.dense(qf_h, 1, name='qf_output',
                                        kernel_initializer=tf.random_uniform_initializer(minval=-3e-3,
                                                                                         maxval=3e-3))

            self.qvalue_fn = qvalue_fn
            self._qvalue = qvalue_fn[:, 0]
        return self.qvalue_fn

    def step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy, {self.obs_ph: obs})

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy, {self.obs_ph: obs})

    def value(self, obs, action, state=None, mask=None):
        return self.sess.run(self._qvalue, {self.obs_ph: obs, self.action_ph: action})