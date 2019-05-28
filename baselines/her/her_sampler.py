import numpy as np

def make_sample_her_transitions():
    def _sample_her_transitions(episode_batch,batch_size_in_transitions,other_on,other_size,n_concat,image_on):

        rollout_batch_size = len(episode_batch['u'])

        batch_size = batch_size_in_transitions

        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)

        epi_len = np.array([episode_batch['u'][x].shape[0] for x in episode_idxs])

        t_sample = np.random.uniform(0,epi_len,batch_size).astype(int)

        transitions = {}

        for key in episode_batch.keys():
            if key == 'o' or key == 'o_2':
                continue
            transitions[key] = np.array([episode_batch[key][episode][sample] for episode,sample in zip(episode_idxs,t_sample)])


        for key in ['o','o_2']:
            if image_on:
                key_list = []
                for episode,sample in zip(episode_idxs,t_sample):
                    length = episode_batch[key][0][0].shape[0]
                    if n_concat-1 <= sample:
                        if other_on:
                            obs = np.concatenate(episode_batch[key][episode][sample-n_concat+1:sample+1,:-1 * other_size])
                            obs = np.concatenate([obs,episode_batch[key][episode][sample][-1*other_size:]])
                        else:
                            obs = np.concatenate(episode_batch[key][episode][sample-n_concat+1:sample+1])
                    else:
                        if other_on:
                            length -= other_size
                        zeros = np.zeros((length*(n_concat - sample - 1),))
                        if other_on:
                            obs = np.concatenate(episode_batch[key][episode][:sample+1,:-1 * other_size])
                            obs = np.concatenate([zeros,obs,episode_batch[key][episode][sample][-1*other_size:]])
                        else:
                            obs = np.concatenate(episode_batch[key][episode][:sample+1])
                            obs = np.concatenate([zeros,obs])
                    key_list.append(obs.copy())
                transitions[key] = np.array(key_list)
            else:
                transitions[key] = np.array([episode_batch[key][episode][sample] for episode,sample in zip(episode_idxs,t_sample)])



            

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}
                       


        assert(transitions['u'].shape[0] == batch_size_in_transitions)

        return transitions

    return _sample_her_transitions
