"""
qlstm.py
"""

import torch
import torch.nn as nn
from torch_qnn.q_layers import QuaternionLinear, TessarineLinear


class QLSTM(nn.Module):
    def __init__(self, feat_size, hidden_size, CUDA, num_classes=None):
        super(QLSTM, self).__init__()

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

        # Gates initialization
        # Forget Gate
        self.wfx = QuaternionLinear(self.input_dim, self.hidden_dim)  # W * x
        self.wfh = QuaternionLinear(self.hidden_dim, self.hidden_dim, bias=False)  # W * h
        # Input Gate
        self.wix = QuaternionLinear(self.input_dim, self.hidden_dim)
        self.wih = QuaternionLinear(self.hidden_dim, self.hidden_dim, bias=False)  # W * h
        # Output Gate
        self.wox = QuaternionLinear(self.input_dim, self.hidden_dim)
        self.woh = QuaternionLinear(self.hidden_dim, self.hidden_dim, bias=False)  # W * h
        # Memory Cell
        self.wcx = QuaternionLinear(self.input_dim, self.hidden_dim)
        self.wch = QuaternionLinear(self.hidden_dim, self.hidden_dim, bias=False)  # W * h

        # Output layer initialization
        self.fco = nn.Linear(self.hidden_dim, self.num_classes)

        # Optimizer
        self.adam = torch.optim.Adam(self.parameters(), lr=0.001)#, weight_decay=5e-3)

    def forward(self, x):
        # Initialize latent space h
        h_init = torch.autograd.Variable(torch.zeros(x.shape[1], self.hidden_dim))

        if self.CUDA:
            x = x.cuda()
            h_init = h_init.cuda()

        # Feed forward affine transformations
        wfx_out = self.wfx(x)
        wix_out = self.wix(x)
        wox_out = self.wox(x)
        wcx_out = self.wcx(x)

        # Processing time steps
        out = []
        c = h_init
        h = h_init

        for k in range(x.shape[0]):
            ft = self.act_gate(wfx_out[k] + self.wfh(h))
            # ft = W_fx * X + W_fh * h
            it = self.act_gate(wix_out[k] + self.wih(h))
            # it = W_ix * X + W_ih * h
            ot = self.act_gate(wox_out[k] + self.woh(h))
            # ot = W_ox * X + W_oh * h
            at = wcx_out[k] + self.wch(h)
            # at = W_cx * X + W_ch * h
            c = it * self.act(at) + ft * c
            # c -> Updated memory cell
            h = ot * self.act(c)
            output = self.fco(h)
            out.append(output.unsqueeze(0))

        return torch.cat(out, 0)


class LSTM(nn.Module):
    def __init__(self, feat_size, hidden_size, CUDA, num_classes=None):
        super(LSTM, self).__init__()

        # Reading options
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

        # Gates initialization
        # Forget Gate
        self.wfx = nn.Linear(self.input_dim, self.hidden_dim)  # W * x
        self.wfh = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)  # W * h
        # Input Gate
        self.wix = nn.Linear(self.input_dim, self.hidden_dim)
        self.wih = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)  # W * h
        # Output Gate
        self.wox = nn.Linear(self.input_dim, self.hidden_dim)
        self.woh = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)  # W * h
        # Memory Cell
        self.wcx = nn.Linear(self.input_dim, self.hidden_dim)
        self.wch = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)  # W * h

        # Output layer initialization
        self.fco = nn.Linear(self.hidden_dim, self.num_classes)

        # Optimizer
        self.adam = torch.optim.Adam(self.parameters(), lr=0.001)#, weight_decay=1e-3)

    def forward(self, x):
        # Initialize latent space h
        h_init = torch.autograd.Variable(torch.zeros(x.shape[1], self.hidden_dim))

        if self.CUDA:
            x = x.cuda()
            h_init = h_init.cuda()

        # Feed forward affine transformations
        wfx_out = self.wfx(x)
        wix_out = self.wix(x)
        wox_out = self.wox(x)
        wcx_out = self.wcx(x)

        # Processing time steps
        out = []
        c = h_init
        h = h_init

        for k in range(x.shape[0]):
            ft = self.act_gate(wfx_out[k] + self.wfh(h))
            # ft = W_fx * X + W_fh * h
            it = self.act_gate(wix_out[k] + self.wih(h))
            # it = W_ix * X + W_ih * h
            ot = self.act_gate(wox_out[k] + self.woh(h))
            # ot = W_ox * X + W_oh * h
            at = wcx_out[k] + self.wch(h)
            # at = W_cx * X + W_ch * h
            c = it * self.act(at) + ft * c
            # c -> Updated memory cell
            h = ot * self.act(c)

            output = self.fco(h)
            out.append(output.unsqueeze(0))

        return torch.cat(out, 0)

class TLSTM(nn.Module):
    def __init__(self, feat_size, hidden_size, CUDA, num_classes=None):
        super(TLSTM, self).__init__()

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

        # Gates initialization
        # Forget Gate
        self.wfx = TessarineLinear(self.input_dim, self.hidden_dim)  # W * x
        self.wfh = TessarineLinear(self.hidden_dim, self.hidden_dim, bias=False)  # W * h
        # Input Gate
        self.wix = TessarineLinear(self.input_dim, self.hidden_dim)
        self.wih = TessarineLinear(self.hidden_dim, self.hidden_dim, bias=False)  # W * h
        # Output Gate
        self.wox = TessarineLinear(self.input_dim, self.hidden_dim)
        self.woh = TessarineLinear(self.hidden_dim, self.hidden_dim, bias=False)  # W * h
        # Memory Cell
        self.wcx = TessarineLinear(self.input_dim, self.hidden_dim)
        self.wch = TessarineLinear(self.hidden_dim, self.hidden_dim, bias=False)  # W * h

        # Output layer initialization
        self.fco = nn.Linear(self.hidden_dim, self.num_classes)

        # Optimizer
        self.adam = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=5e-3)

    def forward(self, x):
        # Initialize latent space h
        h_init = torch.autograd.Variable(torch.zeros(x.shape[1], self.hidden_dim))

        if self.CUDA:
            x = x.cuda()
            h_init = h_init.cuda()

        # Feed forward affine transformations
        wfx_out = self.wfx(x)
        wix_out = self.wix(x)
        wox_out = self.wox(x)
        wcx_out = self.wcx(x)

        # Processing time steps
        out = []
        c = h_init
        h = h_init

        for k in range(x.shape[0]):
            ft = self.act_gate(wfx_out[k] + self.wfh(h))
            # ft = W_fx * X + W_fh * h
            it = self.act_gate(wix_out[k] + self.wih(h))
            # it = W_ix * X + W_ih * h
            ot = self.act_gate(wox_out[k] + self.woh(h))
            # ot = W_ox * X + W_oh * h
            at = wcx_out[k] + self.wch(h)
            # at = W_cx * X + W_ch * h
            c = it * self.act(at) + ft * c
            # c -> Updated memory cell
            h = ot * self.act(c)
            output = self.fco(h)
            out.append(output.unsqueeze(0))

        return torch.cat(out, 0)