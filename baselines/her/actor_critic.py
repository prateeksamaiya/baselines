import tensorflow as tf
from baselines.her.util import store_args, nn , features, flat_process_input,flat_process_input_np
# from baselines.her.model import features


class ActorCritic:
    @store_args
    def __init__(self, inputs_tf, dimo,dim_rgb,dim_depth,dim_other,dimg, dimu, max_u,g_stats,penulti_linear,feature_size,hidden, layers,n_concat_images,**kwargs):
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


        self.rgb_tf,self.depth_tf,self.other_tf = flat_process_input(self.o_tf,size=self.dim_image,n_concat_images=self.n_concat_images)

        # Prepare inputs for actor and critic.
        rgb_img = self.rgb_stats.normalize(self.rgb_tf)
        depth_img = self.depth_stats.normalize(self.depth_tf)
        self.other = self.other_stats.normalize(self.other_tf)
        g = self.g_stats.normalize(self.g_tf)

      
        
        self.rgb_img = tf.reshape(rgb_img,[-1,self.dim_image,self.dim_image,9])
        self.depth_img = tf.reshape(depth_img,[-1,self.dim_image,self.dim_image,3])


        with tf.variable_scope('pi'):

            # Networks.
            with tf.variable_scope('rgb'):
                # print("actor_critic_rgb..............",tf.get_variable_scope().name)
                self.rgb_vec = features(self.rgb_img,self.penulti_linear,feature_size=self.feature_size)
            
            with tf.variable_scope('depth'):
                # print("actor_critic_depth..............",tf.get_variable_scope().name)
                self.depth_vec = features(self.depth_img,self.penulti_linear,feature_size=self.feature_size)
            
            self.input_pi = tf.concat(axis=1, values=[self.rgb_vec,self.depth_vec,self.other, g])  # for actor

            self.pi_tf = self.max_u * tf.tanh(nn(self.input_pi, [self.hidden] * self.layers + [self.dimu]))

        with tf.variable_scope('Q'):
            # for policy training
            input_Q = tf.concat(axis=1, values=[tf.stop_gradient(self.rgb_vec),tf.stop_gradient(self.depth_vec),self.other, g, self.pi_tf / self.max_u]) #stop gradient used
            # # print("input_q_shape_before",input_Q.get_shape())
            # input_Q = tf.concat(axis=1, values=[o, g, self.pi_tf / self.max_u])
            self.Q_pi_tf = nn(input_Q, [self.hidden] * self.layers + [1])               
            # self.Q_pi_tf = nn(input_Q, [self.hidden] * self.layers + [1])               
            # for critic training
            # input_Q = tf.concat(axis=1, values=[o, g, self.u_tf / self.max_u])
            input_Q = tf.concat(axis=1, values=[tf.stop_gradient(self.rgb_vec),tf.stop_gradient(self.depth_vec),self.other,g, self.u_tf / self.max_u]) #stop gradient used
            # print("input_q_shape_later",input_Q.get_shape())
            self._input_Q = input_Q  # exposed for tests
            self.Q_tf = nn(input_Q, [self.hidden] * self.layers + [1], reuse=True)
