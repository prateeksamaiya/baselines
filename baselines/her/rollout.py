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
        self.success_history = deque(maxlen=history_len)
        self.target_reached_history = deque(maxlen=history_len)
        self.collision_history = deque(maxlen=history_len)
        self.epi_len_history = deque(maxlen=history_len)
        self.Q_history = deque(maxlen=history_len)

        self.n_episodes = 0
        self.reset_all_rollouts()
        self.clear_history()

    def reset_all_rollouts(self):
        self.obs_dict = self.venv.reset()
        self.initial_o = self.obs_dict['observation']
        self.initial_ag = self.obs_dict['achieved_goal']
        self.g = self.obs_dict['desired_goal']

    def generate_rollouts(self):
        """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
        policy acting on it accordingly.
        """
        self.reset_all_rollouts()

        # compute observations
        o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        o[:] = self.initial_o
        ag[:] = self.initial_ag

        # generate episodes
        obs, achieved_goals, acts, goals, successes, all_collisions = [], [], [], [], [], []
        dones = []
        rewards = []
        info_values = [np.empty((self.T - 1, self.rollout_batch_size, self.dims['info_' + key]), np.float32) for key in self.info_keys]
        Qs = []
        flag = 1
        first_collision = 0
        for t in range(self.T):
            # print(t)
            # start = rospy.get_rostime()
            policy_output = self.policy.get_actions(
                o, ag, self.g,
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
            ag_new = np.empty((self.rollout_batch_size, self.dims['g']))
            success = np.zeros(self.rollout_batch_size)
            # compute new states and observations
            obs_dict_new, reward, done, info = self.venv.step(u)
            o_new = obs_dict_new['observation']
            ag_new = obs_dict_new['achieved_goal']
            success = np.array([i.get('is_success', 0.0) for i in info])
            collision = np.array([i.get('collision', 0.0) for i in info])

            
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
            achieved_goals.append(ag.copy())
            successes.append(success.copy())
            all_collisions.append(collision.copy())
            acts.append(u.copy())
            goals.append(self.g.copy())
            o[...] = o_new
            ag[...] = ag_new


        obs.append(o.copy())
        achieved_goals.append(ag.copy())

        episode = dict(o=obs,
                       u=acts,
                       g=goals,
                       ag=achieved_goals)

        for key, value in zip(self.info_keys, info_values):
            episode['info_{}'.format(key)] = value[:t]

        # stats
        # print(t)
        # print(len(successes))
        # print(self.rollout_batch_size)
        # print(successes)

        successful = np.array(successes)[-1, :]
        target_reached = np.array([np.array(successes).any()])
        assert successful.shape == (self.rollout_batch_size,)
        success_rate = np.mean(successful)
        target_reached_rate = np.mean(target_reached)
        reward_rate = np.mean(np.array(rewards).sum())
        collision_rate = np.mean(np.array(all_collisions))
        self.collision_history.append(collision_rate)
        self.first_collision_history.append(first_collision)
        self.success_history.append(success_rate)
        self.target_reached_history.append(target_reached_rate)
        self.reward_history.append(reward_rate)
        self.epi_len_history.append(t+1)
        if self.compute_Q:
            self.Q_history.append(np.mean(Qs))
        self.n_episodes += self.rollout_batch_size

        return convert_episode_to_batch_major(episode)

    def clear_history(self):
        """Clears all histories that are used for statistics
        """
        self.success_history.clear()
        self.target_reached_history.clear()
        self.first_collision_history.clear()
        self.collision_history.clear()
        self.epi_len_history.clear()
        self.Q_history.clear()
        self.reward_history.clear()

    def current_success_rate(self):
        return np.mean(self.success_history)
    
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
        logs += [('success_rate', np.mean(self.success_history))]
        logs += [('target_reached_rate', np.mean(self.target_reached_history))]
        logs += [('first_collision', np.mean(self.first_collision_history))]        
        logs += [('collision_rate', np.mean(self.collision_history))]
        logs += [('episode_length', np.mean(self.epi_len_history))]
        logs += [('reward_rate', np.mean(self.reward_history))]
        if self.compute_Q:
            logs += [('mean_Q', np.mean(self.Q_history))]
        logs += [('episode', self.n_episodes)]

        if prefix != '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

