"""
q_ops.py

--- Quaternion Operations
This module stores all the operations required to work on quaternions
"""

import numpy as np
import torch
import torch.nn as nn

from scipy.stats import chi

"""
Init functions
"""


def quaternion_init(in_features, out_features, rng, kernel_size=None, criterion='glorot'):
    """Initialize quaternion layer weights
    """
    if kernel_size is not None:
        receptive_field = np.prod(kernel_size)
        fan_in = in_features * receptive_field
        fan_out = out_features * receptive_field
    else:
        fan_in = in_features
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

    # Generate purely imaginary quaternions
    for i in range(0, no_weights):
        norm = np.sqrt(v_i[i] ** 2 + v_j[i] ** 2 + v_k[i] ** 2 + 0.0001)
        v_i[i] /= norm
        v_j[i] /= norm
        v_k[i] /= norm
    v_i = v_i.reshape(kernel_shape)
    v_j = v_j.reshape(kernel_shape)
    v_k = v_k.reshape(kernel_shape)

    phase = rng.uniform(low=np.pi, high=np.pi, size=kernel_shape)
    weight_r = modulus * np.cos(phase)
    weight_i = modulus * v_i * np.sin(phase)
    weight_j = modulus * v_j * np.sin(phase)
    weight_k = modulus * v_k * np.sin(phase)

    return weight_r, weight_i, weight_j, weight_k


def affect_init(r_weight, i_weight, j_weight, k_weight, init_func, rng, init_criterion):
    # Check input quaternion dimensions
    if r_weight.size() != i_weight.size() or r_weight.size() != j_weight.size() or \
            r_weight.size() != k_weight.size():
        raise ValueError('The real and imaginary weights should have the same size. Found:'
                         'r:' + str(r_weight.size()) + 'i:' + str(i_weight.size()) +
                         'j:' + str(j_weight.size()) + 'k:' + str(k_weight.size()))
    elif 2 != r_weight.dim():
        raise Exception('affect_init accepts only matrices. Found dimensions = ' + str(r_weight.dim()))

    kernel_size = None
    r, i, j, k = init_func(r_weight.size(0), r_weight.size(1), rng, kernel_size, init_criterion)
    r, i, j, k = torch.from_numpy(r), torch.from_numpy(i), torch.from_numpy(j), torch.from_numpy(k)
    # Initialize weights
    r_weight.data = r.type_as(r_weight.data)
    i_weight.data = i.type_as(i_weight.data)
    j_weight.data = j.type_as(j_weight.data)
    k_weight.data = k.type_as(k_weight.data)


def quaternion_linear(qinput, r_weight, i_weight, j_weight, k_weight, bias=None):
    """Apply a quaternion linear transformation on input data
    The function applies the Hamilton Product *
    To ease the computation, a weight matrix W is obtained, and transformed data is
    computed as W * Input.
    """

    w_r = torch.cat([r_weight, -i_weight, -j_weight, -k_weight], dim=0)
    w_i = torch.cat([i_weight, r_weight, -k_weight, j_weight], dim=0)
    w_j = torch.cat([j_weight, k_weight, r_weight, -i_weight], dim=0)
    w_k = torch.cat([k_weight, -j_weight, i_weight, r_weight], dim=0)
    W = torch.cat([w_r, w_i, w_j, w_k], dim=1).float()

    if qinput.dim() == 2:
        if bias is not None:
            return torch.addmm(bias, qinput, W)
        else:
            return torch.mm(qinput, W)
    else:
        output = torch.matmul(qinput, W)
        if bias is not None:
            return output + bias
        else:
            return output


def tessarine_linear(qinput, r_weight, i_weight, j_weight, k_weight, bias=None):
    """Apply a tessarine linear transformation on input data
    The function applies the Hamilton Product *
    To ease the computation, a weight matrix W is obtained, and transformed data is
    computed as W * Input.
    """

    w_r = torch.cat([r_weight, -i_weight, j_weight, -k_weight], dim=0)
    w_i = torch.cat([i_weight, r_weight, k_weight, j_weight], dim=0)
    w_j = torch.cat([j_weight, -k_weight, r_weight, -i_weight], dim=0)
    w_k = torch.cat([k_weight, j_weight, i_weight, r_weight], dim=0)
    W = torch.cat([w_r, w_i, w_j, w_k], dim=1).float()

    if qinput.dim() == 2:
        if bias is not None:
            return torch.addmm(bias, qinput, W)
        else:
            return torch.mm(qinput, W)
    else:
        output = torch.matmul(qinput, W)
        if bias is not None:
            return output + bias
        else:
            return output
