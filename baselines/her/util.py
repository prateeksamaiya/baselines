import os
import subprocess
import sys
import importlib
import inspect
import functools

import tensorflow as tf
import numpy as np

from baselines.common import tf_util as U


def store_args(method):
    """Stores provided method args as instance attributes.
    """
    argspec = inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults = dict(
            zip(argspec.args[-len(argspec.defaults):], argspec.defaults))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        # Get default arg values
        args = defaults.copy()
        # Add provided arg values
        for name, value in zip(arg_names, positional_args[1:]):
            args[name] = value
        args.update(keyword_args)
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper


def import_function(spec):
    """Import a function identified by a string like "pkg.module:fn_name".
    """
    mod_name, fn_name = spec.split(':')
    module = importlib.import_module(mod_name)
    fn = getattr(module, fn_name)
    return fn

    

def flat_process_input(x,is_rgb,is_depth,is_other,critic_rgb,critic_depth,critic_other,other_obs_size,image_width,n_concat_images):
    # print("x",x.get_shape())
    flat_obs = {}
    flat_image = image_width * image_width
    if is_other or critic_other:
        flat_obs['other'] = x[:,-1*other_obs_size:]
        x = x[:,:-1*other_obs_size]

        print(x.get_shape())
        if x.get_shape()[1].value == 0:
            return flat_obs

    obs_len = x.get_shape()[1].value//n_concat_images

    x = tf.reshape(x,[-1,n_concat_images,obs_len])
    x = tf.transpose(x,[0,2,1])

    if is_rgb or critic_rgb:
        flat_obs['rgb'] = tf.reshape(x[:,:flat_image,:],[-1,1*n_concat_images*flat_image])
        x = x[:,flat_image:,:]


    if is_depth or critic_depth:
         flat_obs['depth'] = tf.reshape(x,[-1,n_concat_images*flat_image])

    return flat_obs

def flat_process_input_np(x,is_rgb,is_depth,is_other,critic_rgb,critic_depth,critic_other,other_obs_size,image_width,n_concat_images):
    flat_obs = {}
    flat_image = image_width * image_width
    if is_other or critic_other:
        flat_obs['other'] = x[:,-1*other_obs_size:]
        x = x[:,:-1*other_obs_size]

        if x.shape[1] == 0:
            return flat_obs

    obs_len = x.shape[1]//n_concat_images

    x = x.reshape([-1,n_concat_images,obs_len])
    x = x.transpose([0,2,1])

    if is_rgb or critic_rgb:
        flat_obs['rgb'] = x[:,:flat_image,:].reshape([-1,1*n_concat_images*flat_image])
        x = x[:,flat_image:,:]


    if is_depth or critic_depth:
         flat_obs['depth'] = x.reshape([-1,n_concat_images*flat_image])

    return flat_obs

# def flat_process_input_np(x,size=100,n_concat_images=3):
#     img_len = size*size
#     real_depth = 4*img_len
#     size_all_images = 1*real_depth
#     other = x[:,size_all_images:]
#     x = x[:,:size_all_images]
#     x = x.reshape([-1,n_concat_images,real_depth])
#     x = x.transpose([0,2,1])
#     rgb_image = x[:,:img_len*3,:].reshape([-1,9*img_len])
#     depth_image = x[:,img_len*3:4*img_len,:].reshape([-1,1*img_len])
#     return rgb_image, depth_image, other

def flatten_grads(var_list, grads):
    """Flattens a variables and their gradients.
    """
    return tf.concat([tf.reshape(grad, [U.numel(v)])
                      for (v, grad) in zip(var_list, grads)], 0)

def features(input,penulti_linear,feature_size=50):

    out = input 
    # print(out.get_shape())
    
    # print(tf.get_variable_scope().name)

    for i in range(1,5):
        out = tf.layers.conv2d(out,filters=32,kernel_size=[3,3],strides=2,padding='same',activation=tf.nn.relu,name="cov2d_%d" % i)
    out = tf.layers.conv2d(out,filters=64,kernel_size=[2,2],strides=2,padding='same',activation=tf.nn.relu,name="cov2d_%d" % 5)


    shape = out.get_shape().as_list()        # a list: [None, 9, 2]
    dim = np.prod(shape[1:])            # dim = prod(9,2) = 18
    x = tf.reshape(out, [-1, dim])           # -1 means "all"
    
    
    # x =  tf.layers.dense(inputs=x,units=penulti_linear,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer(),reuse=False)
    # feature =  tf.layers.dense(inputs=x,units=feature_size,kernel_initializer=tf.contrib.layers.xavier_initializer(),reuse=False)

    # print("after_convolution_feature",feature.get_shape())
    print("feature_shape",x.get_shape())
    return x


def nn(input, layers_sizes, reuse=None, flatten=False, name=""):
    """Creates a simple neural network
    """
    for i, size in enumerate(layers_sizes):
        activation = tf.nn.relu if i < len(layers_sizes) - 1 else None
        input = tf.layers.dense(inputs=input,
                                units=size,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                reuse=reuse,
                                name=name + '_' + str(i))
        if activation:
            input = activation(input)
    if flatten:
        assert layers_sizes[-1] == 1
        input = tf.reshape(input, [-1])
    return input


def install_mpi_excepthook():
    import sys
    from mpi4py import MPI
    old_hook = sys.excepthook

    def new_hook(a, b, c):
        old_hook(a, b, c)
        sys.stdout.flush()
        sys.stderr.flush()
        MPI.COMM_WORLD.Abort()
    sys.excepthook = new_hook


def mpi_fork(n, extra_mpi_args=[]):
    """Re-launches the current script with workers
    Returns "parent" for original parent, "child" for MPI children
    """
    if n <= 1:
        return "child"
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        # "-bind-to core" is crucial for good performance
        args = ["mpirun", "-np", str(n)] + \
            extra_mpi_args + \
            [sys.executable]

        args += sys.argv
        subprocess.check_call(args, env=env)
        return "parent"
    else:
        install_mpi_excepthook()
        return "child"


def convert_episode_to_batch_major(episode):
    """Converts an episode to have the batch dimension in the major (first)
    dimension.
    """
    episode_batch = {}
    for key in episode.keys():
        val = np.array(episode[key]).copy()
        # make inputs batch-major instead of time-major
        episode_batch[key] = val.swapaxes(0, 1)

    return episode_batch


def transitions_in_episode_batch(episode_batch):
    """Number of transitions in a given episode batch.
    """
    shape = episode_batch['u'].shape
    return shape[0] * shape[1]


def reshape_for_broadcasting(source, target):
    """Reshapes a tensor (source) to have the correct shape and dtype of the target
    before broadcasting it with MPI.
    """
    dim = len(target.get_shape())
    shape = ([1] * (dim - 1)) + [-1]
    return tf.reshape(tf.cast(source, target.dtype), shape)
