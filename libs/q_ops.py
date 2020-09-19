"""
q_ops.py

--- Quaternion Operations
This module stores all the operations required to work on quaternions
"""

import numpy as pn
import torch
import torch.nn as nn

from scipy.stats import chi


"""
Init functions
"""
def quaternion_init(in_features, out_features, kernel_size=None, criterion='glorot'):
    if kernel_size is not None:
        receptive_field = np.prod(kernel_size)
        fan_in  = in_features  * receptive_field
        fan_out = out_features * receptive_field
    else:
        fan_in  = in_features
        fan_out = out_features

    if criterion == 'glorot':
        s = 1. / np.sqrt(2 * (fan_in + fan_out))
    elif criterion == 'he':
        s = 1. / np.sqrt(2 * fan_in)
    else:
        raise ValueError('Invalid criterion: ' + criterion)

    rng = np.random.RandomState(np.random.randint(1, 1234))

    # Generate randoms and purely imaginary quaternions
    if kernel_size is None:
        kernel_shape = (in_features, out_features)
    else:
        # TODO(Emanuele): Expand different kernel sizes...
        kernel_shape = ...

    # Produce random variable vector
    modulus = chi.rvs(4, loc=0, scale=s, size=kernel_size)
    no_weights = np.prod(kernel_shape)
    v_i = np.random.uniform(-1.0, 1.0, no_weights)
    v_j = np.random.uniform(-1.0, 1.0, no_weights)
    v_k = np.random.uniform(-1.0, 1.0, no_weights)
    



def affect_init(r_weight, i_weight, j_weight, k_weight, init_func, rng, init_criterion):
    # Check input quaternion dimensions
    if r_weight.size() != i_weight.size() or r_weight.size() != j_weight.size() or \
       r_weight.size() != k_weight.size():
        raise ValueError('The real and imaginary weights should have the same size. Found:'
                         'r:' + str(r_weight.size()) + 'i:' + str(i_weight.size()) + 
                         'j:' + str(j_weight.size()) + 'k:' + str(k_weight.size()))
    elif 2 >= r_weight.dim():
        raise Exception('affect_init accepts only matrices. Found dimensions = ' + str(r_weight.dim()))

    kernel_size = None
    r, i, j, k = init_func(r_weight.size(0), r_weight.size(1), rng, kernel_size, init_criterion)
    r, i, j, k = torch.from_numpy(r), torch.from_numpy(i), torch.from_numpy(j), torch.from_numpy(k)
    # Initialize weights
    r_weight.data = r.type_as(r_weight.data)
    i_weight.data = i.type_as(i_weight.data)
    j_weight.data = j.type_as(j_weight.data)
    k_weight.data = k.type_as(k_weight.data)
    pass


