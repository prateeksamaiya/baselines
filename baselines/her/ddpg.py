from collections import OrderedDict

import numpy as np
import math
import tensorflow as tf
from tensorflow.contrib.staging import StagingArea
from baselines import logger
from baselines.her.util import (
    import_function, store_args, flatten_grads, transitions_in_episode_batch, convert_episode_to_batch_major,flat_process_input,flat_process_input_np,nn)
from baselines.her.normalizer import Normalizer
from baselines.common.mpi_adam import MpiAdam
from baselines.her.replay_buffer import ReplayBuffer
from baselines.common import tf_util



def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}


global DEMO_BUFFER #buffer for demonstrations

class DDPG(object):
    @store_args
    def __init__(self,input_dims, buffer_size, hidden, layers, network_class, polyak, batch_size,
                 Q_lr, pi_lr, norm_eps, norm_clip, max_u, action_l2, clip_obs, scope, T,
                 rollout_batch_size, subtract_goals, relative_goals, clip_pos_returns, clip_return,
                 bc_loss, q_filter, num_demo, demo_batch_size, prm_loss_weight, aux_loss_weight,
                 sample_transitions, gamma,penulti_linear,feature_size,other_obs_size,n_concat_images,image_size,is_rgb,is_depth,is_other,is_pred_depth,critic_depth,
                 critic_rgb,critic_other,reuse=False,**kwargs):
        """Implementation of DDPG that is used in combination with Hindsight Experience Replay (HER).
            Added functionality to use demonstrations for training to Overcome exploration problem.

        Args:
            input_dims (dict of ints): dimensions for the observation (o), the goal (g), and the
                actions (u)
            buffer_size (int): number of transitions that are stored in the replay buffer
            hidden (int): number of units in the hidden layers
            layers (int): number of hidden layers
            network_class (str): the network class that should be used (e.g. 'baselines.her.ActorCritic')
            polyak (float): coefficient for Polyak-averaging of the target network
            batch_size (int): batch size for training
            Q_lr (float): learning rate for the Q (critic) network
            pi_lr (float): learning rate for the pi (actor) network
            norm_eps (float): a small value used in the normalizer to avoid numerical instabilities
            norm_clip (float): normalized inputs are clipped to be in [-norm_clip, norm_clip]
            max_u (float): maximum action magnitude, i.e. actions are in [-max_u, max_u]
            action_l2 (float): coefficient for L2 penalty on the actions
            clip_obs (float): clip observations before normalization to be in [-clip_obs, clip_obs]
            scope (str): the scope used for the TensorFlow graph
            T (int): the time horizon for rollouts
            rollout_batch_size (int): number of parallel rollouts per DDPG agent
            subtract_goals (function): function that subtracts goals from each other
            relative_goals (boolean): whether or not relative goals should be fed into the network
            clip_pos_returns (boolean): whether or not positive returns should be clipped
            clip_return (float): clip returns to be in [-clip_return, clip_return]
            sample_transitions (function) function that samples from the replay buffer
            gamma (float): gamma used for Q learning updates
            reuse (boolean): whether or not the networks should be reused
            bc_loss: whether or not the behavior cloning loss should be used as an auxilliary loss
            q_filter: whether or not a filter on the q value update should be used when training with demonstartions
            num_demo: Number of episodes in to be used in the demonstration buffer
            demo_batch_size: number of samples to be used from the demonstrations buffer, per mpi thread
            prm_loss_weight: Weight corresponding to the primary loss
            aux_loss_weight: Weight corresponding to the auxilliary loss also called the cloning loss

        """
        logger.log(self.__dict__)
        self.train_steps = 0

        self.mean_rd_loss = 0.0

        if self.clip_return is None:
            self.clip_return = np.inf

        self.create_actor_critic = import_function(self.network_class)
        # print(self.network_class)
        self.test_create_actor_critic = import_function("baselines.her.test_actor_critic:ActorCritic")

        input_shapes = dims_to_shapes(self.input_dims)
        print(input_shapes)
        flat_image_size = image_size * image_size
        print("flat_image_size",flat_image_size)
        self.dim_image = image_size
        self.dim_rgb = n_concat_images*1*flat_image_size # 3 channels and n images concatenated
        self.dim_depth = n_concat_images*flat_image_size
        self.dimo = self.input_dims['o']
        self.dim_other = self.other_obs_size
        self.dimu = self.input_dims['u']
        self.dim_rd = 1
        self.image_mode = is_rgb or is_depth or critic_rgb or critic_depth
        self.other_mode = is_other or critic_other

        # self.full_obs_size = 0

        # if is_other or critic_other:
        #     self.full_obs_size += self.dim_other
        # if is_depth or critic_depth:
        #     self.full_obs_size += self.dim_depth
        # if is_rgb or critic_rgb:
        #     self.full_obs_size += self.dim_rgb


        # Prepare staging area for feeding data to the model.
        stage_shapes = OrderedDict()

        for key in sorted(self.input_dims.keys()):
            if key.startswith('info_'):
                continue
            stage_shapes[key] = (None, *input_shapes[key])
        for key in ['o',]:
            # stage_shapes[key] = (None,self.full_obs_size)
            stage_shapes[key + '_2'] = stage_shapes[key]
        stage_shapes['r'] = (None,)
        self.stage_shapes = stage_shapes

        print("stage_shapes",stage_shapes)
        # Create network.
        with tf.device('/device:GPU:0'):
            with tf.variable_scope(self.scope):
                self.staging_tf = StagingArea(dtypes=[tf.float32 for _ in self.stage_shapes.keys()],shapes=list(self.stage_shapes.values()))
                self.buffer_ph_tf = [tf.placeholder(tf.float32, shape=shape) for shape in self.stage_shapes.values()]
                self.stage_op = self.staging_tf.put(self.buffer_ph_tf)

                self._create_network(reuse=reuse)


        print("input_shapes",input_shapes)
        # Configure the replay buffer.
        buffer_shapes = {key: (self.T-1 if key != 'o' else self.T, *input_shapes[key])
                         for key, val in input_shapes.items()}

        buffer_size = (self.buffer_size // self.rollout_batch_size) * self.rollout_batch_size
        self.buffer = ReplayBuffer(buffer_shapes, buffer_size, self.T, self.sample_transitions,self.other_mode,other_obs_size,n_concat_images,self.image_mode)

        global DEMO_BUFFER
        DEMO_BUFFER = ReplayBuffer(buffer_shapes, buffer_size, self.T, self.sample_transitions,self.other_mode,other_obs_size,n_concat_images,self.image_mode) #initialize the demo buffer; in the same way as the primary data buffer

    def _random_action(self, n):
        return np.random.uniform(low=-self.max_u, high=self.max_u, size=(n, self.dimu))

    def _preprocess_og(self, o):
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        return o

    def step(self, obs,test=False):
        actions = self.get_actions(obs['observation'],test=test)
        return actions, None, None, None


    def get_actions(self, o, noise_eps=0., random_eps=0., use_target_net=False,compute_Q=False,test=False):
        o = self._preprocess_og(o)

        if test:
            # print("called")
            policy = self.target_test if use_target_net else self.main_test
        else:
            policy= self.target if use_target_net else self.main
        
        vals = [policy.pi_tf]

        if compute_Q:
            vals += [policy.Q_pi_tf]
        # feed
        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32)
        }

        ret = self.sess.run(vals, feed_dict=feed)
        # action postprocessing
        u = ret[0]
        noise = noise_eps * self.max_u * np.random.randn(*u.shape)  # gaussian noise
        u += noise
        u = np.clip(u, -self.max_u, self.max_u)
        u += np.random.binomial(1, random_eps, u.shape[0]).reshape(-1, 1) * (self._random_action(u.shape[0]) - u)  # eps-greedy
        if u.shape[0] == 1:
            u = u[0]
        u = u.copy()
        ret[0] = u

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def init_demo_buffer(self, demoDataFile, update_stats=True): #function that initializes the demo buffer

        demoData = np.load(demoDataFile) #load the demonstration data from data file
        info_keys = [key.replace('info_', '') for key in self.input_dims.keys() if key.startswith('info_')]
        info_values = [np.empty((self.T - 1, 1, self.input_dims['info_' + key]), np.float32) for key in info_keys]

        demo_data_obs = demoData['obs']
        demo_data_acs = demoData['acs']
        demo_data_info = demoData['info']


        for epsd in range(self.num_demo): # we initialize the whole demo buffer at the start of the training
            obs, acts,rewards = [], [] ,[]
            i = 0
            for transition in range(self.T - 1):
                obs.append([demo_data_obs[epsd][transition].get('observation')])
                acts.append([demo_data_acs[epsd][transition]])
                for idx, key in enumerate(info_keys):
                    info_values[idx][transition, i] = demo_data_info[epsd][transition][key]


            obs.append([demo_data_obs[epsd][self.T - 1].get('observation')])

            episode = dict(o=obs,
                           u=acts,
                           r=rewards,
                           )
            for key, value in zip(info_keys, info_values):
                episode['info_{}'.format(key)] = value

            episode = convert_episode_to_batch_major(episode)
            global DEMO_BUFFER
            DEMO_BUFFER.store_episode(episode) # create the observation dict and append them into the demonstration buffer
            logger.debug("Demo buffer size currently ", DEMO_BUFFER.get_current_size()) #print out the demonstration buffer size

            if update_stats:
                # add transitions to normalizer to normalize the demo data as well
                episode['o_2'] = episode['o'][:, 1:, :]
                num_normalizing_transitions = transitions_in_episode_batch(episode)
                transitions = self.sample_transitions(episode, num_normalizing_transitions,self.other_mode,self.other_obs_size,self.n_concat_images,self.image_mode)

                o = transitions['o']
                transitions['o'] = self._preprocess_og(o)
                # No need to preprocess the o_2 and g_2 since this is only used for stats
                flat_obs = flat_process_input_np(transitions['o'],self.is_rgb,self.is_depth,self.is_other,self.critic_rgb,self.critic_depth,self.critic_other,self.other_obs_size,self.dim_image,self.n_concat_images)
                if self.is_rgb or self.critic_rgb:
                    self.rgb_stats.update(flat_obs['rgb'])
                    self.rgb_stats.recompute_stats()
                if self.is_depth or self.critic_depth:
                    self.depth_stats.update(flat_obs['depth'])
                    self.depth_stats.recompute_stats()
                if self.is_other or self.critic_other:
                    self.other_stats.update(flat_obs['other'])
                    self.other_stats.recompute_stats()

            episode.clear()

        logger.info("Demo buffer size: ", DEMO_BUFFER.get_current_size()) #print out the demonstration buffer size

    def store_episode(self, episode_batch,update_stats=True):
        """
        episode_batch: array of batch_size x (T or T+1) x dim_key
                       'o' is of size T+1, others are of size T
        """

        self.buffer.store_episode(episode_batch)

        if update_stats:
            # add transitions to normalizer
            episode_batch['o_2'] = episode_batch['o'][:, 1:, :]
            num_normalizing_transitions = transitions_in_episode_batch(episode_batch)
            # print("num_normalizing_transitions",num_normalizing_transitions)
            transitions = self.sample_transitions(episode_batch,num_normalizing_transitions,self.other_mode,self.other_obs_size,self.n_concat_images,self.image_mode)

            o = transitions['o']
            # print("ddpg store o shape",o.shape)
            transitions['o']= self._preprocess_og(o)
            # No need to preprocess the o_2 and g_2 since this is only used for stats

            flat_obs = flat_process_input_np(transitions['o'],self.is_rgb,self.is_depth,self.is_other,self.critic_rgb,self.critic_depth,self.critic_other,self.other_obs_size,self.dim_image,self.n_concat_images)
            if self.is_rgb or self.critic_rgb:
                self.rgb_stats.update(flat_obs['rgb'])
                self.rgb_stats.recompute_stats()
            if self.is_depth or self.critic_depth:
                self.depth_stats.update(flat_obs['depth'])
                self.depth_stats.recompute_stats()
            if self.is_other or self.critic_other:
                self.other_stats.update(flat_obs['other'])
                self.other_stats.recompute_stats()


    def get_current_buffer_size(self):
        return self.buffer.get_current_size()

    def _sync_optimizers(self):
        self.Q_adam.sync()
        self.pi_adam.sync()
        if self.is_pred_depth:
            self.pred_depth_adam.sync()

    def _grads(self):
        # Avoid feed_dict here for performance!

        input = {'critic_loss':self.Q_loss_tf,
                 'actor_loss':self.main.Q_pi_tf,
                 'Q_grad':self.Q_grad_tf,
                 'pi_grad':self.pi_grad_tf
        }

        if self.is_pred_depth:
            input.update({'pred_depth_grad':self.pred_depth_grad_tf,'pred_depth_loss':self.pred_depth_loss})
        
        output = self.sess.run(input)

        if self.is_pred_depth:
            self.rd_loss.update(output['pred_depth_loss'])
            self.rd_loss.recompute_stats()

        return output


    def _update(self, Q_grad, pi_grad, pred_depth_grad=None):
        self.Q_adam.update(Q_grad, self.Q_lr)
        self.pi_adam.update(pi_grad, self.pi_lr)
        if self.is_pred_depth:
            self.pred_depth_adam.update(pred_depth_grad, self.Q_lr)

    def sample_batch(self):

        if self.bc_loss: #use demonstration buffer to sample as well if bc_loss flag is set TRUE
            transitions = self.buffer.sample(self.batch_size - self.demo_batch_size)
            global DEMO_BUFFER
            transitions_demo = DEMO_BUFFER.sample(self.demo_batch_size) #sample from the demo buffer
            for k, values in transitions_demo.items():
                rolloutV = transitions[k].tolist()
                for v in values:
                    rolloutV.append(v.tolist())
                transitions[k] = np.array(rolloutV)
        else:
            transitions = self.buffer.sample(self.batch_size) #otherwise only sample from primary buffer

        o, o_2= transitions['o'], transitions['o_2']
        transitions['o'] = self._preprocess_og(o)
        transitions['o_2'] = self._preprocess_og(o_2)

        transitions_batch = [transitions[key] for key in self.stage_shapes.keys()]
        return transitions_batch

    def stage_batch(self, batch=None):
        if batch is None:
            batch = self.sample_batch()
        assert len(self.buffer_ph_tf) == len(batch)
        self.sess.run(self.stage_op, feed_dict=dict(zip(self.buffer_ph_tf, batch)))


    def train(self, stage=True):
        # writer = tf.summary.FileWriter('/home/vision/tensorboard/', self.sess.graph)
        if stage:
            self.stage_batch()
       
        output = self._grads()

        if self.is_pred_depth:
            self._update(output['Q_grad'], output['pi_grad'],output['pred_depth_grad'])
        else:
            self._update(output['Q_grad'], output['pi_grad'])

        return output['critic_loss'], output['actor_loss']

    def _init_target_net(self):
        self.sess.run(self.init_target_net_op)

    def update_target_net(self):
        self.sess.run(self.update_target_net_op)

    def clear_buffer(self):
        self.buffer.clear_buffer()

    def _vars(self, scope):
        # print("current scope",self.scope)
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/' + scope)
        assert len(res) > 0
        return res

    def _global_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '/' + scope)
        return res

    def _create_network(self, reuse=False):
        logger.info("Creating a DDPG agent with action space %d x %s..." % (self.dimu, self.max_u))
        self.sess = tf_util.get_session()

        # running averages
        if self.is_rgb or self.critic_rgb:
            with tf.variable_scope('rgb_stats') as vs:
                if reuse:
                    vs.reuse_variables()
                self.rgb_stats = Normalizer(self.dim_rgb, self.norm_eps, self.norm_clip, sess=self.sess)
        if self.is_depth or self.critic_depth:
            with tf.variable_scope('depth_stats') as vs:
                if reuse:
                    vs.reuse_variables()
                self.depth_stats = Normalizer(self.dim_depth, self.norm_eps, self.norm_clip, sess=self.sess)
        if self.is_other or self.critic_other:
            with tf.variable_scope('other_stats') as vs:
                if reuse:
                    vs.reuse_variables()
                self.other_stats = Normalizer(self.dim_other, self.norm_eps, self.norm_clip, sess=self.sess)
        if self.is_pred_depth:
            with tf.variable_scope('rd_loss') as vs:
                if reuse:
                    vs.reuse_variables()
                self.rd_loss = Normalizer(self.dim_rd, self.norm_eps, self.norm_clip, sess=self.sess)
    

        # mini-batch sampling.
        batch = self.staging_tf.get()
        batch_tf = OrderedDict([(key, batch[i]) for i, key in enumerate(self.stage_shapes.keys())])
        batch_tf['r'] = tf.reshape(batch_tf['r'], [-1, 1])

        #choose only the demo buffer samples
        mask = np.concatenate((np.zeros(self.batch_size - self.demo_batch_size), np.ones(self.demo_batch_size)), axis = 0)
        
        # print("printint.......................................")
        # networks
        with tf.variable_scope('main') as vs:
            if reuse:
                vs.reuse_variables()
            self.main = self.create_actor_critic(batch_tf, net_type='main', **self.__dict__)
            vs.reuse_variables()
        with tf.variable_scope('target') as vs:
            if reuse:
                vs.reuse_variables()
            target_batch_tf = batch_tf.copy()
            target_batch_tf['o'] = batch_tf['o_2']
            self.target = self.create_actor_critic(target_batch_tf, net_type='target', **self.__dict__)
            vs.reuse_variables()
        assert len(self._vars("main")) == len(self._vars("target"))

        #real-depth-network
        if self.is_pred_depth:
            output_size = self.feature_size
            with tf.variable_scope("pred_depth") as vs:
                self.pred_depth_vec = nn(self.main.pi_rgb_vec, [self.hidden] * self.layers + [output_size], reuse=False)

        
            
        # networks
        if self.is_pred_depth:

            with tf.variable_scope('main',reuse=True):
                self.main_test = self.test_create_actor_critic(batch_tf, net_type='main', **self.__dict__)

            with tf.variable_scope('target',reuse=True):
                self.target_test = self.test_create_actor_critic(target_batch_tf, net_type='target',**self.__dict__)
               
           


        # loss functions
        target_Q_pi_tf = self.target.Q_pi_tf
        clip_range = (-self.clip_return, 0. if self.clip_pos_returns else np.inf)
        target_tf = tf.clip_by_value(batch_tf['r'] + self.gamma * target_Q_pi_tf, *clip_range)
        self.Q_loss_tf = tf.reduce_mean(tf.square(tf.stop_gradient(target_tf) - self.main.Q_tf))

        if self.bc_loss ==1 and self.q_filter == 1 : # train with demonstrations and use bc_loss and q_filter both
            maskMain = tf.reshape(tf.boolean_mask(self.main.Q_tf > self.main.Q_pi_tf, mask), [-1]) #where is the demonstrator action better than actor action according to the critic? choose those samples only
            #define the cloning loss on the actor's actions only on the samples which adhere to the above masks
            self.cloning_loss_tf = tf.reduce_sum(tf.square(tf.boolean_mask(tf.boolean_mask((self.main.pi_tf), mask), maskMain, axis=0) - tf.boolean_mask(tf.boolean_mask((batch_tf['u']), mask), maskMain, axis=0)))
            self.pi_loss_tf = -self.prm_loss_weight * tf.reduce_mean(self.main.Q_pi_tf) #primary loss scaled by it's respective weight prm_loss_weight
            self.pi_loss_tf += self.prm_loss_weight * self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u)) #L2 loss on action values scaled by the same weight prm_loss_weight
            self.pi_loss_tf += self.aux_loss_weight * self.cloning_loss_tf #adding the cloning loss to the actor loss as an auxilliary loss scaled by its weight aux_loss_weight

        elif self.bc_loss == 1 and self.q_filter == 0: # train with demonstrations without q_filter
            self.cloning_loss_tf = tf.reduce_sum(tf.square(tf.boolean_mask((self.main.pi_tf), mask) - tf.boolean_mask((batch_tf['u']), mask)))
            self.pi_loss_tf = -self.prm_loss_weight * tf.reduce_mean(self.main.Q_pi_tf)
            self.pi_loss_tf += self.prm_loss_weight * self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))
            self.pi_loss_tf += self.aux_loss_weight * self.cloning_loss_tf

        else: #If  not training with demonstrations
            self.pi_loss_tf = -tf.reduce_mean(self.main.Q_pi_tf)
            self.pi_loss_tf += self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))

        
        #pred_depth_loss
        if self.is_pred_depth:
            self.pred_depth_loss = tf.losses.mean_squared_error(tf.stop_gradient(self.main.pi_depth_vec),self.pred_depth_vec)


        if self.is_pred_depth:
            pred_depth_grads_tf = tf.gradients(self.pred_depth_loss, self._vars('pred_depth')+self._vars('main/pi/rgb'))
            assert len(self._vars('pred_depth')+self._vars('main/pi/rgb')) == len(pred_depth_grads_tf)
            self.pred_depth_grads_vars_tf = zip(pred_depth_grads_tf, self._vars('pred_depth')+self._vars('main/pi/rgb'))
            self.pred_depth_grad_tf = flatten_grads(grads=pred_depth_grads_tf, var_list=self._vars('pred_depth')+self._vars('main/pi/rgb'))

        Q_grads_tf = tf.gradients(self.Q_loss_tf, self._vars('main/Q'))
        assert len(self._vars('main/Q')) == len(Q_grads_tf)
        self.Q_grads_vars_tf = zip(Q_grads_tf, self._vars('main/Q'))
        self.Q_grad_tf = flatten_grads(grads=Q_grads_tf, var_list=self._vars('main/Q'))
       
        pi_grads_tf = tf.gradients(self.pi_loss_tf, self._vars('main/pi'))
        assert len(self._vars('main/pi')) == len(pi_grads_tf)
        self.pi_grads_vars_tf = zip(pi_grads_tf, self._vars('main/pi'))
        self.pi_grad_tf = flatten_grads(grads=pi_grads_tf, var_list=self._vars('main/pi'))
       
        
        
        # optimizers
        self.Q_adam = MpiAdam(self._vars('main/Q'), scale_grad_by_procs=False)
        self.pi_adam = MpiAdam(self._vars('main/pi'), scale_grad_by_procs=False)
        if self.is_pred_depth:
            self.pred_depth_adam = MpiAdam(self._vars('pred_depth')+self._vars('main/pi/rgb'), scale_grad_by_procs=False)
    
        stat_vars = None

        if self.is_rgb or self.critic_rgb:
            stat_vars = self._global_vars('rgb_stats') 

        if self.is_depth or self.critic_depth:
            if stat_vars:
                stat_vars +=  self._global_vars('depth_stats')
            else:
                stat_vars =  self._global_vars('depth_stats')

        if self.is_other or self.critic_other:
            if stat_vars:
                stat_vars +=  self._global_vars('other_stats')
            else:
                stat_vars =  self._global_vars('other_stats')

        # polyak averaging
        self.main_vars = self._vars('main/Q') + self._vars('main/pi')
        self.target_vars = self._vars('target/Q') + self._vars('target/pi')
        self.stats_vars =  stat_vars
        self.init_target_net_op = list(map(lambda v: v[0].assign(v[1]), zip(self.target_vars, self.main_vars)))
        self.update_target_net_op = list(map(lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]), zip(self.target_vars, self.main_vars)))

        # initialize all variables
        tf.variables_initializer(self._global_vars('')).run()
        self._sync_optimizers()
        self._init_target_net()

    def logs(self, prefix=''):
        logs = []
        if self.is_rgb or self.critic_rgb:
            logs += [('stats_rgb/mean', np.mean(self.sess.run([self.rgb_stats.mean])))]
            logs += [('stats_rgb/std', np.mean(self.sess.run([self.rgb_stats.std])))]
        if self.is_depth or self.critic_depth:
            logs += [('stats_depth/mean', np.mean(self.sess.run([self.depth_stats.mean])))]
            logs += [('stats_depth/std', np.mean(self.sess.run([self.depth_stats.std])))]
        if self.is_other or self.critic_other:
            logs += [('stats_other/mean', np.mean(self.sess.run([self.other_stats.mean])))]
            logs += [('stats_other/std', np.mean(self.sess.run([self.other_stats.std])))]
        if self.is_pred_depth:
            logs += [('stats_rd_loss/mean', np.mean(self.sess.run([self.rd_loss.mean])))]
            logs += [('stats_rd_loss/std', np.mean(self.sess.run([self.rd_loss.std])))]
        
        if prefix != '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def __getstate__(self):
        """Our policies can be loaded from pkl, but after unpickling you cannot continue training.
        """
        excluded_subnames = ['_tf', '_op', '_vars', '_adam', 'buffer', 'sess', '_stats',
                             'main', 'target', 'lock', 'env', 'sample_transitions',
                             'stage_shapes', 'create_actor_critic']

        state = {k: v for k, v in self.__dict__.items() if all([not subname in k for subname in excluded_subnames])}
        state['buffer_size'] = self.buffer_size
        state['tf'] = self.sess.run([x for x in self._global_vars('') if 'buffer' not in x.name])
        return state

    def __setstate__(self, state):
        if 'sample_transitions' not in state:
            # We don't need this for playing the policy.
            state['sample_transitions'] = None

        self.__init__(**state)
        # set up stats (they are overwritten in __init__)
        for k, v in state.items():
            if k[-6:] == '_stats':
                self.__dict__[k] = v
        # load TF variables
        vars = [x for x in self._global_vars('') if 'buffer' not in x.name]
        assert(len(vars) == len(state["tf"]))
        node = [tf.assign(var, val) for var, val in zip(vars, state["tf"])]
        self.sess.run(node)

    def save(self, save_path):
        tf_util.save_variables(save_path)

