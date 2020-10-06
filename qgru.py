"""qgru.py
"""

import torch
import torch.nn as nn
from torch_qnn.q_layers import QuaternionLinear


class QGRU(nn.Module):
    def __init__(self, feat_size, hidden_size, CUDA, num_classes=None):
        super(QGRU, self).__init__()

        self.act = nn.Tanh()
        self.act_gate = nn.Sigmoid()
        self.input_dim = feat_size
        self.hidden_dim = hidden_size
        self.CUDA = CUDA

        if num_classes is None:
            self.num_classes = feat_size + 1  # +1 because feat_size = no. on the sequence and the output one hot will
            # also have a blank dimension so FEAT_SIZE + 1 BLANK
        else:
            self.num_classes = num_classes

        self.wrx = QuaternionLinear(self.input_dim, self.hidden_dim)
        self.wrh = QuaternionLinear(self.hidden_dim, self.hidden_dim, bias=False)

        self.wx1 = QuaternionLinear(self.input_dim, self.hidden_dim)
        self.wh1 = QuaternionLinear(self.hidden_dim, self.hidden_dim, bias=False)

        self.wux = QuaternionLinear(self.input_dim, self.hidden_dim)
        self.wuh = QuaternionLinear(self.hidden_dim, self.hidden_dim, bias=False)

        # Output layer initialization
        self.fco = nn.Linear(self.hidden_dim, self.num_classes)

        # Optimizer
        self.adam = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        # Initialize latent space h
        h_init = torch.autograd.Variable(torch.zeros(x.shape[1], self.hidden_dim))

        if self.CUDA:
            x = x.cuda()
            h_init = h_init.cuda()

        wrx_out = self.wrx(x)
        wx1_out = self.wx1(x)
        wux_out = self.wux(x)

        out = []
        h = h_init

        for k in range(x.shape[0]):
            # gate_reset = sigmoid(W_rx * X + W_rh * h)
            gr = self.act_gate(wrx_out[k] + self.wrh(h))
            # gate_update = sigmoid(W_ux * X + W_uh * h)
            gu = self.act_gate(wux_out[k] + self.wuh(h))
            # r = tanh(gate_reset * (W_h1 * h) + W_x1 * X)
            r = self.act(gr * (self.wh1(h)) + wx1_out[k])
            u = gu * h

            h = r * (1 - gu) + u
            output = self.fco(h)
            out.append(output.unsqueeze(0))
        return torch.cat(out, 0)


class GRU(nn.Module):
    def __init__(self, feat_size, hidden_size, CUDA, num_classes=None):
        super(GRU, self).__init__()

        self.act = nn.Tanh()
        self.act_gate = nn.Sigmoid()
        self.input_dim = feat_size
        self.hidden_dim = hidden_size
        self.CUDA = CUDA

        if num_classes is None:
            self.num_classes = feat_size + 1  # +1 because feat_size = no. on the sequence and the output one hot will
            # also have a blank dimension so FEAT_SIZE + 1 BLANK
        else:
            self.num_classes = num_classes

        self.wrx = nn.Linear(self.input_dim, self.hidden_dim)
        self.wrh = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        self.wx1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.wh1 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        self.wux = nn.Linear(self.input_dim, self.hidden_dim)
        self.wuh = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        # Output layer initialization
        self.fco = nn.Linear(self.hidden_dim, self.num_classes)

        # Optimizer
        self.adam = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        # Initialize latent space h
        h_init = torch.autograd.Variable(torch.zeros(x.shape[1], self.hidden_dim))

        if self.CUDA:
            x = x.cuda()
            h_init = h_init.cuda()

        wrx_out = self.wrx(x)
        wx1_out = self.wx1(x)
        wux_out = self.wux(x)

        out = []
        h = h_init

        for k in range(x.shape[0]):
            # gate_reset = sigmoid(W_rx * X + W_rh * h)
            gr = self.act_gate(wrx_out[k] + self.wrh(h))
            # gate_update = sigmoid(W_ux * X + W_uh * h)
            gu = self.act_gate(wux_out[k] + self.wuh(h))
            # r = tanh(gate_reset * (W_h1 * h) + W_x1 * X)
            r = self.act(gr * (self.wh1(h)) + wx1_out[k])
            u = gu * h

            h = r * (1 - gu) + u
            output = self.fco(h)
            out.append(output.unsqueeze(0))
        return torch.cat(out, 0)


class LGRU(nn.Module):
    """Light GRU Implementation"""
    def __init__(self, feat_size, hidden_size, CUDA, num_classes=None):
        super(LGRU, self).__init__()
        # TODO: Complete implementation