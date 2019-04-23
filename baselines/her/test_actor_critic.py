import tensorflow as tf
from baselines.her.util import store_args, nn , process_input, features, flat_process_input
# from baselines.her.model import features


class ActorCritic:
    @store_args
    def __init__(self, inputs_tf, dimo,dim_rgb,dim_depth,dim_other,dimg,pred_depth_vec,dimu, max_u, g_stats, hidden,feature_size,layers,net_type="main",ddpg_scope=None,
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
    


        self.depth_vector = self.pred_depth_vec


        with tf.variable_scope(self.ddpg_scope,reuse=True) as scope:
            with tf.variable_scope('rgb'):
                # print("name_under_rgb",tf.get_variable_scope().name)
                self.rgb_vec = features(rgb_img,penulti_linear=256,feature_size=self.feature_size)


            with tf.variable_scope(self.net_type):
                with tf.variable_scope('pi'):
                    # print("name_main/pi",tf.get_variable_scope().name)
                    input_pi = tf.concat(axis=1, values=[self.rgb_vec,self.depth_vector,other, g])  # for actor
                    self.pi_tf = self.max_u * tf.tanh(nn(input_pi, [self.hidden] * self.layers + [self.dimu]))
                with tf.variable_scope('Q'):
                    # print("name_main/Q",tf.get_variable_scope().name)                    
                    input_Q = tf.concat(axis=1, values=[tf.stop_gradient(self.rgb_vec),tf.stop_gradient(self.depth_vector),other, g, self.pi_tf / self.max_u])
                    self.Q_pi_tf = nn(input_Q, [self.hidden] * self.layers + [1])               
        