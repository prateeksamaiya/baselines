import numpy as np

def make_sample_her_transitions():
    def _sample_her_transitions(episode_batch,batch_size_in_transitions):

        rollout_batch_size = len(episode_batch['u'])

        batch_size = batch_size_in_transitions

        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)

        epi_len = np.array([episode_batch['u'][x].shape[0] for x in episode_idxs])

        t_samples = np.random.uniform(0,epi_len,batch_size).astype(int)

        transitions = {}

        for key in episode_batch.keys():
            transitions[key] = np.array([episode_batch[key][episode][sample] for episode,sample in zip(episode_idxs,t_samples)])
            

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}


        assert(transitions['u'].shape[0] == batch_size_in_transitions)

        return transitions

    return _sample_her_transitions
