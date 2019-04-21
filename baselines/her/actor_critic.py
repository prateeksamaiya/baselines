import tensorflow as tf
from baselines.her.util import store_args, nn , process_input, features, flat_process_input
# from baselines.her.model import features


class ActorCritic:
    @store_args
    def __init__(self, inputs_tf, dimo,dim_rgb,dim_depth,dim_other,dimg, dimu, max_u, g_stats, hidden, layers,ddpg_scope=None,
                 **kwargs):
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


        rgb_tf,depth_tf,other_tf = flat_process_input(self.o_tf,size=self.dim_image)

        # Prepare inputs for actor and critic.
        rgb_img = self.rgb_stats.normalize(rgb_tf)
        depth_img = self.depth_stats.normalize(depth_tf)
        other = self.other_stats.normalize(other_tf)
        g = self.g_stats.normalize(self.g_tf)

      
        rgb_img = tf.reshape(rgb_img,[-1,self.dim_image,self.dim_image,3])
        depth_img = tf.reshape(depth_img,[-1,self.dim_image,self.dim_image,1])
    

        R_use = False
        if tf.get_variable_scope().name == "ddpg/target":
            R_use = True



        with tf.variable_scope(self.ddpg_scope,reuse=R_use):
            with tf.variable_scope('rgb') as vs:
                # print("actor_critic_rgb..............",tf.get_variable_scope().name)
                self.rgb_vec = features(rgb_img,penulti_linear=256,feature_size=32)
                # vs.reuse_variables()
            with tf.variable_scope('depth') as vs:
                # print("actor_critic_depth..............",tf.get_variable_scope().name)
                self.depth_vec = features(depth_img,penulti_linear=256,feature_size=32)
                # vs.reuse_variables()
       
        self.depth_vector = self.depth_vec

        with tf.variable_scope('pi'):
            # input_pi = tf.concat(axis=1, values=[o, g])  # for actor
            input_pi = tf.concat(axis=1, values=[self.rgb_vec,self.depth_vector,other, g])  # for actor
        # Networks.
            self.pi_tf = self.max_u * tf.tanh(nn(input_pi, [self.hidden] * self.layers + [self.dimu]))
        with tf.variable_scope('Q'):
            # for policy training
            input_Q = tf.concat(axis=1, values=[tf.stop_gradient(self.rgb_vec),tf.stop_gradient(self.depth_vector),other, g, self.pi_tf / self.max_u]) #stop gradient used
            # # print("input_q_shape_before",input_Q.get_shape())
            # input_Q = tf.concat(axis=1, values=[o, g, self.pi_tf / self.max_u])
            self.Q_pi_tf = nn(input_Q, [self.hidden] * self.layers + [1])               
            # self.Q_pi_tf = nn(input_Q, [self.hidden] * self.layers + [1])               
            # for critic training
            # input_Q = tf.concat(axis=1, values=[o, g, self.u_tf / self.max_u])
            input_Q = tf.concat(axis=1, values=[tf.stop_gradient(self.rgb_vec),tf.stop_gradient(self.depth_vector),other,g, self.u_tf / self.max_u]) #stop gradient used
            # print("input_q_shape_later",input_Q.get_shape())
            self._input_Q = input_Q  # exposed for tests
            self.Q_tf = nn(input_Q, [self.hidden] * self.layers + [1], reuse=True)
