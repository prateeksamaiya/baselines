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

    

def flat_process_input(x,size=100):
    size_o = tf.size(x[0])//3
    x = tf.reshape(x,[-1,3,size_o])
    x = tf.transpose(x,[0,2,1])
    # print("x = ",x.get_shape().as_list())
    img_len = size*size
    rgb_image = tf.reshape(x[:,:img_len*3,:],[-1,9*img_len])
    depth_image = tf.reshape(x[:,img_len*3:4*img_len,:],[-1,3*img_len])
    other = tf.reshape(x[:,4*img_len:,:],[-1,3*(size_o - 4*img_len)])
   
    return rgb_image, depth_image, other

def flat_process_input_np(x,size=100):
    size_o = len(x[0])//3
    x = x.reshape([-1,3,size_o])
    x = x.transpose([0,2,1])
    img_len = size*size
    rgb_image = x[:,:img_len*3,:].reshape([-1,9*img_len])
    depth_image = x[:,img_len*3:4*img_len,:].reshape([-1,3*img_len])
    other = x[:,4*img_len:,:].reshape([-1,3*(size_o - 4*img_len)])
    return rgb_image, depth_image, other

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
    
    shape = out.get_shape().as_list()        # a list: [None, 9, 2]
    dim = np.prod(shape[1:])            # dim = prod(9,2) = 18
    x = tf.reshape(out, [-1, dim])           # -1 means "all"
    
    
    x =  tf.layers.dense(inputs=x,units=penulti_linear,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer(),reuse=False)
    feature =  tf.layers.dense(inputs=x,units=feature_size,kernel_initializer=tf.contrib.layers.xavier_initializer(),reuse=False)

    # print("after_convolution_feature",feature.get_shape())

    return feature


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
