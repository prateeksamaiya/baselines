import tensorflow as tf
from baselines.her.util import store_args, nn , features, flat_process_input,flat_process_input_np
# from baselines.her.model import features


class ActorCritic:
    @store_args
    def __init__(self, inputs_tf,penulti_linear,feature_size,hidden, layers,other_obs_size,n_concat_images,is_rgb,is_depth,is_other,is_pred_depth,
    rgb_stats=None,depth_stats=None,g_stats=None,**kwargs):
        """The actor-critic network and related training code.

        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), and the action (u)
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions
            max_u (float): the maximum magnitude of actions; action outputs will be scaled
                accordingly
            o_stats (baselines.her.Normalizer): normalizer for observations
            g_stats (baselines.her.Normalizer): normalizer for goals
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
        """
        self.o_tf = inputs_tf['o']
        self.g_tf = inputs_tf['g']
        self.u_tf = inputs_tf['u']


        flat_obs = flat_process_input(self.o_tf,is_rgb,is_depth,is_other,other_obs_size,self.dim_image,self.n_concat_images)
        
        pi_inputs=[]
        Q_inputs=[]

        # Prepare inputs for actor and critic.
        if is_rgb:
            rgb_img = self.rgb_stats.normalize(flat_obs['rgb'])
            self.rgb_img = tf.reshape(rgb_img,[-1,self.dim_image,self.dim_image,9])
        if is_depth and is_pred_depth:
            depth_img = self.depth_stats.normalize(flat_obs['depth'])
            self.depth_img = tf.reshape(depth_img,[-1,self.dim_image,self.dim_image,3])
        if is_other:
            self.other = self.other_stats.normalize(flat_obs['other'])
           

        g = self.g_stats.normalize(self.g_tf)

        if is_depth and is_pred_depth:
            self.depth_vec = self.pred_depth_vec

        with tf.variable_scope('pi'):

            # Networks.
            if is_rgb:
                with tf.variable_scope('rgb'):
                    self.rgb_vec = features(self.rgb_img,self.penulti_linear,feature_size=self.feature_size)
                    pi_inputs.append(self.rgb_vec)
                    Q_inputs.append(tf.stop_gradient(self.rgb_vec))

            if is_depth and is_pred_depth:
                pi_inputs.append(self.depth_vec)
                Q_inputs.append(tf.stop_gradient(self.depth_vec))

            if is_other:
                pi_inputs.append(self.other)
                Q_inputs.append(self.other)
            
            self.input_pi = tf.concat(axis=1, values=pi_inputs+[g])  # for actor

            self.pi_tf = self.max_u * tf.tanh(nn(self.input_pi, [self.hidden] * self.layers + [self.dimu]))

        with tf.variable_scope('Q'):
            # for policy training
            # input_Q = tf.concat(axis=1, values=[tf.stop_gradient(self.rgb_vec),tf.stop_gradient(self.depth_vec),self.other, g, self.pi_tf / self.max_u]) #stop gradient used
            input_Q = tf.concat(axis=1, values=Q_inputs+[g, self.pi_tf / self.max_u]) #stop gradient used
            # # print("input_q_shape_before",input_Q.get_shape())
            # input_Q = tf.concat(axis=1, values=[o, g, self.pi_tf / self.max_u])
            self.Q_pi_tf = nn(input_Q, [self.hidden] * self.layers + [1])               
            # self.Q_pi_tf = nn(input_Q, [self.hidden] * self.layers + [1])               
            # for critic training
            # input_Q = tf.concat(axis=1, values=[o, g, self.u_tf / self.max_u])
            input_Q = tf.concat(axis=1, values=Q_inputs+[g, self.u_tf / self.max_u]) #stop gradient used
            # print("input_q_shape_later",input_Q.get_shape())
            self._input_Q = input_Q  # exposed for tests
            self.Q_tf = nn(input_Q, [self.hidden] * self.layers + [1], reuse=True)

