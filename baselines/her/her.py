import os

import click
import numpy as np
import json
from mpi4py import MPI
import time
from baselines import logger
import os.path as osp
from baselines.common import set_global_seeds, tf_util
from baselines.common.mpi_moments import mpi_moments
import baselines.her.experiment.config as config
from baselines.her.rollout import RolloutWorker
import tensorflow as tf
import psutil

def mpi_average(value,dtype=np.float32):
    if not isinstance(value, list):
        value = [value]
    if not any(value):
        value = [0.]

    if hasattr(value[0],"dtype"):
        # print(value[0].dtype)
        dtype = value[0].dtype
    return mpi_moments(np.array(value,dtype=dtype))[0]


def train(*, policy, rollout_worker, evaluator,
          n_epochs, n_test_rollouts, n_cycles, n_batches, policy_save_interval,
          save_path, demo_file, **kwargs):
    rank = MPI.COMM_WORLD.Get_rank()


    if save_path:
        latest_policy_path = os.path.join(save_path, 'policy_latest.pkl')
        best_policy_path = os.path.join(save_path, 'policy_best.pkl')
        periodic_policy_path = os.path.join(save_path, 'policy_{}.pkl')

    logger.info("Training...")
    best_success_rate = -1

    if policy.bc_loss == 1: policy.init_demo_buffer(demo_file) #initialize demo buffer if training with demonstrations

    # # num_timesteps = n_epochs * n_cycles * rollout_length * number of rollout workers
   
    
    # for epoch in range(n_epochs):
    #     start_time = time.time()
    #     # train
    #     rollout_worker.clear_history()
    #     for i in range(n_cycles):
    #         print(i)
    #         episode = rollout_worker.generate_rollouts()
    #         # print("checkpoint 1")
    #         policy.store_episode(episode)
    #         # print("checkpoint 2")
    #         for _ in range(n_batches):
    #             policy.train()
    #         policy.update_target_net()

    #         print("single_episode_time",time.time()-start_time)

    mean_epoch_time = 0
    mean_rest_time = 0
    for epoch in range(n_epochs):
        print("memory stat",psutil.virtual_memory())
        epoch_start_time = time.time()
        a = b = c = d = 0.0
        # train
        rollout_worker.clear_history()
        for count in range(n_cycles):
            # print(rank,count)
            # print(count)
            rollout_start_time = time.time()
            episode = rollout_worker.generate_rollouts()
            # print("rollout_epoch_time : ",time.time()-rollout_start_time)
            a += time.time()-rollout_start_time
            # A = time.time()-rollout_start_time

            # print("checkpoint 1")
            store_start_time = time.time()
            policy.store_episode(episode)
            # print("store_epoch_time : ",time.time()-store_start_time)
            b += time.time()-store_start_time
            # B = time.time()-store_start_time
            # print("checkpoint 2")
            train_start_time = time.time()
            for _ in range(n_batches):
                policy.train()
            # print("train_epoch_time : ",time.time()-train_start_time)
            c += time.time()-train_start_time
            # C  = time.time()-train_start_time
            # print("checkpoint 3")

            update_start_time = time.time()
            policy.update_target_net()
            # print("update_epoch_time : ",time.time()-update_start_time)
            d += time.time()-update_start_time
            # D = time.time()-update_start_time
            # print(A,B,C,D)
            # print("checkpoint 4")


        rest_start_time= time.time()

        # test
        evaluator.clear_history()
        for _ in range(n_test_rollouts):
            evaluator.generate_rollouts()

        # record logs
        logger.record_tabular('epoch', epoch)
        for key, val in evaluator.logs('test'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in rollout_worker.logs('train'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in policy.logs():
            logger.record_tabular(key, mpi_average(val))


            

        if rank == 0:
            logger.dump_tabular()

        # save the policy if it's better than the previous ones
        # success_rate = mpi_average(evaluator.current_success_rate())
        # if rank == 0 and success_rate >= best_success_rate and save_path:
        #     best_success_rate = success_rate
        #     logger.info('New best success rate: {}. Saving policy to {} ...'.format(best_success_rate, best_policy_path))
        #     evaluator.save_policy(best_policy_path)
        #     evaluator.save_policy(latest_policy_path)
        # if rank == 0 and policy_save_interval > 0 and epoch % policy_save_interval == 0 and save_path:
        #     policy_path = periodic_policy_path.format(epoch)
        #     logger.info('Saving periodic policy to {} ...'.format(policy_path))
        #     evaluator.save_policy(policy_path)

        # make sure that different threads have different seeds
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        if rank != 0:
            assert local_uniform[0] != root_uniform[0]

        if epoch%5==0 and rank == 0:
            path = osp.expanduser(save_path)
            policy.save(path+"/policy_"+str(epoch))


        rest_epoch_time = time.time()-rest_start_time
        mean_rest_time += (rest_epoch_time - mean_rest_time)/(epoch+1)
        print("avg:(rollout,store,train,update,) times : ",a/n_cycles,b/n_cycles,c/n_cycles,d/n_cycles,mean_rest_time)
        # print("rest_epoch_time : ",time.time()-rest_start_time)
        current_epoch_time = time.time()-epoch_start_time
        mean_epoch_time += (current_epoch_time - mean_epoch_time)/(epoch+1)
        print("average_epoch_time : ",mean_epoch_time)

    return policy


def learn(*, network, env, total_timesteps,
    seed=None,
    eval_env=None,
    replay_strategy='future',
    policy_save_interval=0,
    clip_return=True,
    demo_file=None,
    override_params=None,
    load_path=None,
    save_path=None,
    **kwargs
):

    override_params = override_params or {}
    if MPI is not None:
        rank = MPI.COMM_WORLD.Get_rank()
        num_cpu = MPI.COMM_WORLD.Get_size()

    # Seed everything.
    rank_seed = seed + 1000000 * rank if seed is not None else None
    set_global_seeds(rank_seed)

    # Prepare params.
    params = config.DEFAULT_PARAMS
    env_name = env.specs[0].id
    params['env_name'] = env_name
    params['replay_strategy'] = replay_strategy
    if env_name in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
    params.update(**override_params)  # makes it possible to override any parameter
    with open(os.path.join(logger.get_dir(), 'params.json'), 'w') as f:
         json.dump(params, f)
    params = config.prepare_params(params)
    params['rollout_batch_size'] = env.num_envs

    if demo_file is not None:
        params['bc_loss'] = 1
    params.update(kwargs)

    config.log_params(params, logger=logger)

    if num_cpu == 1:
        logger.warn()
        logger.warn('*** Warning ***')
        logger.warn(
            'You are running HER with just a single MPI worker. This will work, but the ' +
            'experiments that we report in Plappert et al. (2018, https://arxiv.org/abs/1802.09464) ' +
            'were obtained with --num_cpu 19. This makes a significant difference and if you ' +
            'are looking to reproduce those results, be aware of this. Please also refer to ' +
            'https://github.com/openai/baselines/issues/314 for further details.')
        logger.warn('****************')
        logger.warn()

    dims = config.configure_dims(params)
    policy_dims = config.configure_dims(params)
    policy = config.configure_ddpg(dims=policy_dims, params=params, clip_return=clip_return)
    if load_path is not None:
        tf_util.load_variables(load_path)

    rollout_params = {
        'exploit': False,
        'use_target_net': False,
        'use_demo_states': True,
        'compute_Q': False,
        'T': params['T'],
    }

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'use_demo_states': False,
        'compute_Q': True,
        'T': params['T'],
    }

    for name in ['T', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps']:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]

    eval_env = eval_env or env

    rollout_worker = RolloutWorker(env, policy, dims, logger, monitor=True, **rollout_params)
    evaluator = RolloutWorker(eval_env, policy, dims, logger, **eval_params)

    n_cycles = params['n_cycles']
    n_epochs = total_timesteps // n_cycles // rollout_worker.T // rollout_worker.rollout_batch_size

    with tf.device('/device:GPU:0'):   
        return train(
            save_path=save_path, policy=policy, rollout_worker=rollout_worker,
            evaluator=evaluator, n_epochs=n_epochs, n_test_rollouts=params['n_test_rollouts'],
            n_cycles=params['n_cycles'], n_batches=params['n_batches'],
            policy_save_interval=policy_save_interval, demo_file=demo_file)


@click.command()
@click.option('--env', type=str, default='FetchReach-v1', help='the name of the OpenAI Gym environment that you want to train on')
@click.option('--total_timesteps', type=int, default=int(5e5), help='the number of timesteps to run')
@click.option('--seed', type=int, default=0, help='the random seed used to seed both the environment and the training code')
@click.option('--policy_save_interval', type=int, default=5, help='the interval with which policy pickles are saved. If set to 0, only the best and latest policy will be pickled.')
@click.option('--replay_strategy', type=click.Choice(['future', 'none']), default='future', help='the HER replay strategy to be used. "future" uses HER, "none" disables HER.')
@click.option('--clip_return', type=int, default=1, help='whether or not returns should be clipped')
@click.option('--demo_file', type=str, default = 'PATH/TO/DEMO/DATA/FILE.npz', help='demo data file path')
def main(**kwargs):
    learn(**kwargs)


if __name__ == '__main__':
    main()
