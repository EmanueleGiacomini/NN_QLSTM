"""har.py
Human Activity Recognition through smartphone data
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from torch_qnn.q_layers import QuaternionLinear, TessarineLinear
from experiments.har.har import load_ucihar, QLSTM, tovar, load_cb, NUM_CLASSES, run_epoch


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
        self.adam = torch.optim.Adam(self.parameters(), lr=0.005)

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
        out = self.fco(h)
        return out


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
        self.adam = torch.optim.Adam(self.parameters(), lr=0.005)

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
        out = self.fco(h)
        return out


CUDA = True
FEAT_SIZE = 8
SEQ_LEN = 128
QHIDDEN_SIZE = 40
GRUHIDDEN_SIZE = 48
OUTPUT_SIZE = NUM_CLASSES
EPOCHS = 51
BATCH_SIZE = 64

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)

    x_train, y_train = load_ucihar(train=True, onehot=False)
    x_test, y_test = load_ucihar(train=False, onehot=False)

    x_train, y_train, x_test, y_test = torch.from_numpy(x_train), \
                                       torch.from_numpy(y_train), \
                                       torch.from_numpy(x_test), \
                                       torch.from_numpy(y_test)

    train_data = TensorDataset(x_train.float(), y_train.float())
    test_data = TensorDataset(x_test.float(), y_test.float())

    train_loader = DataLoader(train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
    test_loader = DataLoader(test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)

    if CUDA:
        net_q = QLSTM(FEAT_SIZE, QHIDDEN_SIZE, CUDA, OUTPUT_SIZE).cuda()
        net_t = TLSTM(FEAT_SIZE, QHIDDEN_SIZE, CUDA, OUTPUT_SIZE).cuda()
        net_gru = QGRU(FEAT_SIZE, GRUHIDDEN_SIZE, CUDA, OUTPUT_SIZE).cuda()
    else:
        net_q = QLSTM(FEAT_SIZE, QHIDDEN_SIZE, CUDA, OUTPUT_SIZE)
        net_t = TLSTM(FEAT_SIZE, QHIDDEN_SIZE, CUDA, OUTPUT_SIZE)
        net_gru = QGRU(FEAT_SIZE, GRUHIDDEN_SIZE, CUDA, OUTPUT_SIZE)

    nb_param_q = sum(p.numel() for p in net_q.parameters() if p.requires_grad)
    nb_param_t = sum(p.numel() for p in net_t.parameters() if p.requires_grad)
    nb_param_gru = sum(p.numel() for p in net_gru.parameters() if p.requires_grad)

    print(f'(QLSTM) Number of trainable parameters : {nb_param_q}')
    print(f'(TLSTM) Number of trainable parameters : {nb_param_t}')
    print(f'(QGRU) Number of trainable parameters : {nb_param_gru}')

    """Training Loop"""
    acc_q = []
    acc_t = []
    acc_gru = []
    loss_q = []
    loss_t = []
    loss_gru = []
    val_acc_q = []
    val_acc_t = []
    val_acc_gru = []
    val_loss_q = []
    val_loss_t = []
    val_loss_gru = []

    for epoch in range(EPOCHS):
        tacc_q, tloss_q, vacc_q, vloss_q = run_epoch(net_q, train_loader,
                                                     test_loader, CUDA)
        tacc_t, tloss_t, vacc_t, vloss_t = run_epoch(net_t, train_loader,
                                                     test_loader, CUDA)
        tacc_gru, tloss_gru, vacc_gru, vloss_gru = run_epoch(net_gru, train_loader,
                                                             test_loader, CUDA)
        print(f'Epoch: {epoch}')
        print(f'QLSTM: loss={tloss_q:.4f}, acc={tacc_q:.3f}, val_loss={vloss_q:.4f}, val_acc={vacc_q:.3f}')
        print(f'TLSTM: loss={tloss_t:.4f}, acc={tacc_t:.3f}, val_loss={vloss_t:.4f}, val_acc={vacc_t:.3f}')
        print(f' QGRU: loss={tloss_gru:.4f}, acc={tacc_gru:.3f}, val_loss={vloss_gru:.4f}, val_acc={vacc_gru:.3f}')
        acc_q.append(tacc_q)
        acc_t.append(tacc_t)
        acc_gru.append(tacc_gru)
        loss_q.append(tloss_q)
        loss_t.append(tloss_t)
        loss_gru.append(tloss_gru)
        val_acc_q.append(vacc_q)
        val_acc_t.append(vacc_t)
        val_acc_gru.append(vacc_gru)
        val_loss_q.append(vloss_q)
        val_loss_t.append(vloss_t)
        val_loss_gru.append(vloss_gru)

    print('Training phase ended.')
    np.savetxt(f'out/2_har_task_acc_q.txt', acc_q)
    np.savetxt(f'out/2_har_task_acc_t.txt', acc_t)
    np.savetxt(f'out/2_har_task_acc_gru.txt', acc_gru)
    np.savetxt(f'out/2_har_task_loss_q.txt', loss_q)
    np.savetxt(f'out/2_har_task_loss_t.txt', loss_t)
    np.savetxt(f'out/2_har_task_loss_gru.txt', loss_gru)
    np.savetxt(f'out/2_har_task_val_acc_q.txt', val_acc_q)
    np.savetxt(f'out/2_har_task_val_acc_t.txt', val_acc_t)
    np.savetxt(f'out/2_har_task_val_acc_gru.txt', val_acc_gru)
    np.savetxt(f'out/2_har_task_val_loss_q.txt', val_loss_q)
    np.savetxt(f'out/2_har_task_val_loss_t.txt', val_loss_t)
    np.savetxt(f'out/2_har_task_val_loss_gru.txt', val_loss_gru)
