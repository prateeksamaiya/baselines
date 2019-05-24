from collections import deque

import numpy as np
import pickle
import rospy

from baselines.her.util import convert_episode_to_batch_major, store_args


class RolloutWorker:

    @store_args
    def __init__(self, venv, policy, dims, logger, T, rollout_batch_size=1,
                 exploit=False, use_target_net=False, compute_Q=False, noise_eps=0,
                 random_eps=0, history_len=100, render=False, monitor=False, **kwargs):
        """Rollout worker generates experience by interacting with one or many environments.

        Args:
            make_env (function): a factory function that creates a new instance of the environment
                when called
            policy (object): the policy that is used to act
            dims (dict of ints): the dimensions for observations (o), goals (g), and actions (u)
            logger (object): the logger that is used by the rollout worker
            rollout_batch_size (int): the number of parallel rollouts that should be used
            exploit (boolean): whether or not to exploit, i.e. to act optimally according to the
                current policy without any exploration
            use_target_net (boolean): whether or not to use the target net for rollouts
            compute_Q (boolean): whether or not to compute the Q values alongside the actions
            noise_eps (float): scale of the additive Gaussian noise
            random_eps (float): probability of selecting a completely random action
            history_len (int): length of history for statistics smoothing
            render (boolean): whether or not to render the rollouts
        """

        assert self.T > 0

        self.info_keys = [key.replace('info_', '') for key in dims.keys() if key.startswith('info_')]

        self.first_collision_history = deque(maxlen=history_len)
        self.reward_history = deque(maxlen=history_len)
        self.max_distance_history = deque(maxlen=history_len)
        self.collision_history = deque(maxlen=history_len)
        self.epi_len_history = deque(maxlen=history_len)
        self.Q_history = deque(maxlen=history_len)

        self.n_episodes = 0
        self.reset_all_rollouts()
        self.clear_history()

    def reset_all_rollouts(self):
        self.obs_dict = self.venv.reset()
        self.initial_o = self.obs_dict['observation']

    def generate_rollouts(self):
        """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
        policy acting on it accordingly.
        """
        self.reset_all_rollouts()

        # compute observations
        o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations

        # generate episodes
        obs,rewards,acts, all_collisions = [], [], [], []
        dones = []
        max_distance = 0
        info_values = [np.empty((self.T - 1, self.rollout_batch_size, self.dims['info_' + key]), np.float32) for key in self.info_keys]
        Qs = []
        flag = 1
        first_collision = 0
        for t in range(self.T):
            policy_output = self.policy.get_actions(
                o,
                compute_Q=self.compute_Q,
                noise_eps=self.noise_eps if not self.exploit else 0.,
                random_eps=self.random_eps if not self.exploit else 0.,
                use_target_net=self.use_target_net)

            if self.compute_Q:
                u, Q = policy_output
                Qs.append(Q)
            else:
                u = policy_output

            if u.ndim == 1:
                # The non-batched case should still have a reasonable shape.
                u = u.reshape(1, -1)

            o_new = np.empty((self.rollout_batch_size, self.dims['o']))
            # compute new states and observations
            obs_dict_new, reward, done, info = self.venv.step(u)
            o_new = obs_dict_new['observation']
            collision = np.array([i.get('collision', 0.0) for i in info])
            distance_from_origin =  np.array([i.get('distance', 0.0) for i in info])

            max_distance = max(max_distance,distance_from_origin)

            
            if collision[0] == True and flag:
                flag = 0
                first_collision = t+1

            if any(done):
                if not t:
                    self.reset_all_rollouts()
                    return self.generate_rollouts()
                # here we assume all environments are done is ~same number of steps, so we terminate rollouts whenever any of the envs returns done
                # trick with using vecenvs is not to add the obs from the environments that are "done", because those are already observations
                # after a reset
                break

            for i, info_dict in enumerate(info):
                for idx, key in enumerate(self.info_keys):
                    info_values[idx][t, i] = info[i][key]

            if np.isnan(o_new).any():
                self.logger.warn('NaN caught during rollout generation. Trying again...')
                self.reset_all_rollouts()
                return self.generate_rollouts()

            # print("time_per_step",rospy.get_rostime() - start)

            dones.append(done)
            rewards.append(reward)
            obs.append(o.copy())
            all_collisions.append(collision.copy())
            acts.append(u.copy())
            o[...] = o_new


        obs.append(o.copy())

        episode = dict(o=obs,
                       u=acts,
                       r=rewards,
                       )

        for key, value in zip(self.info_keys, info_values):
            episode['info_{}'.format(key)] = value[:t]

        reward_rate = np.mean(np.array(rewards).sum())
        collision_rate = np.mean(np.array(all_collisions))
        self.collision_history.append(collision_rate)
        self.first_collision_history.append(first_collision)
        self.max_distance_history.append(max_distance)
        self.reward_history.append(reward_rate)
        self.epi_len_history.append(t+1)
        if self.compute_Q:
            self.Q_history.append(np.mean(Qs))
        self.n_episodes += self.rollout_batch_size

        return convert_episode_to_batch_major(episode)

    def clear_history(self):
        """Clears all histories that are used for statistics
        """
        self.first_collision_history.clear()
        self.collision_history.clear()
        self.epi_len_history.clear()
        self.Q_history.clear()
        self.reward_history.clear()
        self.max_distance_history.clear()
    
    def current_collision_rate(self):
        return np.mean(self.collision_history)

    def current_mean_Q(self):
        return np.mean(self.Q_history)

    def reward_mean(self):
        return np.mean(self.reward_history)

    def save_policy(self, path):
        """Pickles the current policy for later inspection.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.policy, f)

    def logs(self, prefix='worker'):
        """Generates a dictionary that contains all collected statistics.
        """
        logs = []
        logs += [('first_collision', np.mean(self.first_collision_history))]        
        logs += [('collision_rate', np.mean(self.collision_history))]
        logs += [('episode_length', np.mean(self.epi_len_history))]
        logs += [('reward_rate', np.mean(self.reward_history))]
        logs += [('max_distance', np.mean(self.max_distance_history))]

        if self.compute_Q:
            logs += [('mean_Q', np.mean(self.Q_history))]
        logs += [('episode', self.n_episodes)]

        if prefix != '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

