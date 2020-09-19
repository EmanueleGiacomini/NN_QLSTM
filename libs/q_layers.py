"""
q_layers.py

--- Quaternion Operations
This module stores all the custom layers that uses quaternion operations
"""

import numpy as pn
import torch
import torch.nn as nn
import q_ops


class QuaternionLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, seed=None, quaternion_format=True, scale=False):
        super(QuaternionLinear, self).__init__()
        self.in_features = in_features // 4
        self.out_features = out_features // 4
        self.r_weight = nn.Parameter(torch.Tensor(self.in_features, self.out_features))
        self.i_weight = nn.Parameter(torch.Tensor(self.in_features, self.out_features))
        self.j_weight = nn.Parameter(torch.Tensor(self.in_features, self.out_features))
        self.k_weight = nn.Parameter(torch.Tensor(self.in_features, self.out_features))
        self.quaternion_format = quaternion_format
        self.scale = scale

        if self.scale:
            self.scale_param = nn.Parameter(torch.Tensor(self.in_features, self.out_features))
        else:
            self.scale_param = None

        self.seed = seed if seed is not None else np.random.randint(0, 1024)
        self.rng = np.random.RandomState(self.seed)

        self.reset_parameters()

    def reset_parameters(self):
        if self.scale_param is not None:
            torch.nn.init.xavier_uniform_(self.scale_param.data)
        if self.bias is not None:
            self.bias.data.fill_(0)
        affect_init(self.r_weight, self.i_weight, self.j_weight, self.k_weight, winit,
                    self.rng, self.init_criterion)
