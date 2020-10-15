"""har.py
Human Activity Recognition through smartphone data
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from qlstm import QLSTM, TLSTM, LSTM
from qgru import QGRU, GRU

DATASET_ROOT_FS = './UCI HAR Dataset/'
TRAINING_FS = 'train/'
TEST_FS = 'test/'
INERTIAL_FS = 'Inertial Signals/'
"""
DATA_FILES = [
    'body_acc_x', 'body_acc_y', 'body_acc_z',
    'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
    'total_acc_x', 'total_acc_y', 'total_acc_z'
]
"""

DATA_FILES = [
    'body_acc_x', 'body_acc_y', 'body_acc_z',
    'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
    'total_acc_x', 'total_acc_y'
]

NUM_CLASSES = 6


def read_data_lst(path: str, max_length=128) -> [np.array]:
    """
    Returns a list of np.arrays. Each element represent a time series for
    measurements indentified by the file 'path'
    """
    data = []
    with open(path, 'r') as f:
        for line in f.readlines():
            line_data = []
            for elem in line.strip().split():
                val = float(elem)
                line_data.append(val)
            data.append(np.array(line_data[:max_length]))
    return np.array(data)


def merge_data(data: [np.array]) -> np.array:
    """
    Merge the lists of features into a single higher-dimensional numpy array.
    Each element of the resulting array contains a vector of features
    """
    return np.transpose(data, (1, 2, 0))


def read_labels_lst(path: str, zero_idx=True) -> [int]:
    """
    Returns the list of labels for a given file identified by 'path'
    """
    labels = []
    with open(path, 'r') as f:
        for lab in f.readlines():
            label = np.array(int(lab.strip()))
            if zero_idx:
                label -= 1
            labels.append(label)
    labels = np.array(labels)
    return np.reshape(labels, (labels.shape[0], 1))


def load_ucihar(train=True):
    main_fs = DATASET_ROOT_FS
    main_fs += TRAINING_FS if train else TEST_FS
    extension = '_train.txt' if train else '_test.txt'
    # Load data
    data_lst = [read_data_lst(main_fs + INERTIAL_FS + dname + extension)
                for dname in DATA_FILES]
    labels_fs = main_fs
    labels_fs += 'y_train.txt' if train else 'y_test.txt'
    labels = read_labels_lst(labels_fs, True)
    data = merge_data(data_lst)
    return data, labels


LABELS = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']

"""class QLSTM(nn.Module):
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
        self.adam = torch.optim.Adam(self.parameters(), lr=0.01)

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
        return output


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
        self.adam = torch.optim.Adam(self.parameters(), lr=0.01)

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
            #out.append(output.unsqueeze(0))
        return out

class GTModel(nn.Module):
    def __init__(self, feat_size, hidden_size, CUDA, num_classes=None):
        super(GTModel, self).__init__()
        self.lstm = nn.LSTM(feat_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, num_classes)
        self.act = nn.Softmax()

        self.adam = torch.optim.Adam(self.parameters(), lr=0.001)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc1(x[-1, :, :])
        return self.act(x)
"""


def tovar(x, cuda):
    """Transform x in a tensor stored in RAM or VRAM based on cuda value"""
    if cuda:
        return Variable(torch.FloatTensor(x.float()).cuda())
    else:
        return Variable(torch.FloatTensor(x.float()))


def load_cb(cval, fval, title, total_time):
    NUM_BARS = 30
    ratio = cval / fval
    ratio *= NUM_BARS
    ratio = int(ratio)
    print('\r|' + '=' * int(ratio) + '>' + ' ' * (NUM_BARS - ratio) +
          '| ' + f'{total_time:.3f}' + 's ' + title, end='')
    if cval == fval:
        print(' Done.')
    return


CUDA = True
FEAT_SIZE = 8
SEQ_LEN = 128
QHIDDEN_SIZE = 64
HIDDEN_SIZE = 32
OUTPUT_SIZE = NUM_CLASSES
EPOCHS = 51
BATCH_SIZE = 256

import time


def run_epoch(net, train_loader, test_loader, CUDA):
    train_acc = 0.0
    train_loss = 0.0
    test_acc = 0.0
    test_loss = 0.0
    epoch_it = 0
    final_it = len(train_loader)
    t0 = time.perf_counter()
    for (x, y) in train_loader:
        x = x.transpose(0, 1).contiguous()

        x, y = tovar(x, CUDA), tovar(y, CUDA)
        net.zero_grad()
        p = net(x)[-1, :]
        y_pred = p
        y = y.view(-1).long()

        loss = nn.CrossEntropyLoss()
        val_loss = loss(y_pred, y)
        # Backpropagate
        val_loss.backward()
        net.adam.step()

        # Compute train acc and loss
        with torch.no_grad():
            y_pred = y_pred.argmax(1).cpu().data.numpy()
            train_acc += sum(y.cpu().data.numpy() == y_pred) / y_pred.shape[0]
            train_loss += val_loss.cpu().data.numpy()
        epoch_it += 1
        t1 = time.perf_counter() - t0
        load_cb(epoch_it, final_it, 'Training', t1)
    train_acc /= epoch_it
    train_loss /= epoch_it

    # Validation step
    epoch_it = 0
    final_it = len(test_loader)
    t0 = time.perf_counter()
    with torch.no_grad():
        for (x, y) in test_loader:
            x = x.transpose(0, 1).contiguous()
            x, y = tovar(x, CUDA), tovar(y, CUDA)
            p = net(x)[-1, :]
            y_pred = p
            y = y.view(-1).long()

            loss = nn.CrossEntropyLoss()
            val_loss = loss(y_pred, y)
            y_pred = y_pred.argmax(1).cpu().data.numpy()
            test_acc += sum(y.cpu().data.numpy() == y_pred) / y_pred.shape[0]
            test_loss += val_loss.cpu().data.numpy()
            epoch_it += 1
            t1 = time.perf_counter() - t0
            load_cb(epoch_it, final_it, 'Testing', t1)
        test_acc /= epoch_it
        test_loss /= epoch_it
    return train_acc, train_loss, test_acc, test_loss


TEST_MODELS = {'LSTM80': LSTM(FEAT_SIZE, 80, CUDA, OUTPUT_SIZE).cuda(),
               'QLSTM160': QLSTM(FEAT_SIZE, 160, CUDA, OUTPUT_SIZE).cuda(),
               'LSTM32': LSTM(FEAT_SIZE, 32, CUDA, OUTPUT_SIZE).cuda(),
               'QLSTM64': QLSTM(FEAT_SIZE, 64, CUDA, OUTPUT_SIZE).cuda(),
               'LSTM16': LSTM(FEAT_SIZE, 16, CUDA, OUTPUT_SIZE).cuda(),
               'QLSTM32': QLSTM(FEAT_SIZE, 32, CUDA, OUTPUT_SIZE).cuda(),
               'GRU16': GRU(FEAT_SIZE, 16, CUDA, OUTPUT_SIZE).cuda(),
               'QGRU32': QGRU(FEAT_SIZE, 32, CUDA, OUTPUT_SIZE).cuda(),
               'TLSTM32': TLSTM(FEAT_SIZE, 32, CUDA, OUTPUT_SIZE).cuda()}

RESULT_DICT = {}

if __name__ == '__main__':
    x_train, y_train = load_ucihar(train=True)
    x_test, y_test = load_ucihar(train=False)

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

    for mname in TEST_MODELS.keys():
        RESULT_DICT[mname] = {}
        RESULT_DICT[mname]['train_acc'] = list()
        RESULT_DICT[mname]['train_loss'] = list()
        RESULT_DICT[mname]['val_acc'] = list()
        RESULT_DICT[mname]['val_loss'] = list()

    for epoch in range(EPOCHS):
        for mname in TEST_MODELS.keys():
            model = TEST_MODELS[mname]
            train_acc, train_loss, val_acc, val_loss = run_epoch(model, train_loader, test_loader, CUDA)
            RESULT_DICT[mname]['train_acc'].append(train_acc)
            RESULT_DICT[mname]['train_loss'].append(train_loss)
            RESULT_DICT[mname]['val_acc'].append(val_acc)
            RESULT_DICT[mname]['val_loss'].append(val_loss)

        if epoch % 5 == 0:
            print(f'Epoch={epoch}')
            for mname in TEST_MODELS.keys():
                train_loss = RESULT_DICT[mname]['train_loss'][-1]
                train_acc = RESULT_DICT[mname]['train_acc'][-1]
                val_loss = RESULT_DICT[mname]['val_loss'][-1]
                val_acc = RESULT_DICT[mname]['val_acc'][-1]
                print(f'{mname}: loss={train_loss:.4f}, acc={train_acc:.3f}, val_loss={val_loss:.4f},'
                      f' val_acc={val_acc:.3f}')
    print('Training ended.')
    pd.DataFrame(RESULT_DICT).to_csv('./out/results.csv')

    """if CUDA:
        net_q = QLSTM(FEAT_SIZE, QHIDDEN_SIZE, CUDA, OUTPUT_SIZE).cuda()
        net_r = LSTM(FEAT_SIZE, HIDDEN_SIZE, CUDA, OUTPUT_SIZE).cuda()
    else:
        net_q = QLSTM(FEAT_SIZE, QHIDDEN_SIZE, CUDA, OUTPUT_SIZE)
        net_r = LSTM(FEAT_SIZE, HIDDEN_SIZE, CUDA, OUTPUT_SIZE)

    nb_param_q = sum(p.numel() for p in net_q.parameters() if p.requires_grad)
    nb_param_p = sum(p.numel() for p in net_r.parameters() if p.requires_grad)

    print(f'(QLSTM) Number of trainable parameters : {nb_param_q}')
    print(f'(LSTM) Number of trainable parameters : {nb_param_p}')
    
    acc_r = []
    acc_q = []
    loss_r = []
    loss_q = []
    val_acc_r = []
    val_acc_q = []
    val_loss_r = []
    val_loss_q = []
    for epoch in range(EPOCHS):
        tacc_r, tloss_r, vacc_r, vloss_r = run_epoch(net_r, train_loader,
                                                     test_loader, CUDA)
        tacc_q, tloss_q, vacc_q, vloss_q = run_epoch(net_q, train_loader,
                                                     test_loader, CUDA)
        print(f'Epoch: {epoch}')
        print(f'LSTM: loss={tloss_r:.4f}, acc={tacc_r:.3f}, val_loss={vloss_r:.4f}, val_acc={vacc_r:.3f}')
        print(f'QLSTM: loss={tloss_q:.4f}, acc={tacc_q:.3f}, val_loss={vloss_q:.4f}, val_acc={vacc_q:.3f}')
        acc_r.append(tacc_r)
        acc_q.append(tacc_q)
        loss_r.append(tloss_r)
        loss_q.append(tloss_q)
        val_acc_r.append(vacc_r)
        val_acc_q.append(vacc_q)
        val_loss_r.append(vloss_r)
        val_loss_q.append(vloss_q)

    print('Training phase ended.')
    np.savetxt(f'out/2_har_task_acc_q.txt', acc_q)
    np.savetxt(f'out/2_har_task_acc_r.txt', acc_r)
    np.savetxt(f'out/2_har_task_loss_q.txt', loss_q)
    np.savetxt(f'out/2_har_task_loss_r.txt', loss_r)
    np.savetxt(f'out/2_har_task_val_acc_q.txt', val_acc_q)
    np.savetxt(f'out/2_har_task_val_acc_r.txt', val_acc_r)
    np.savetxt(f'out/2_har_task_val_loss_q.txt', val_loss_q)
    np.savetxt(f'out/2_har_task_val_loss_r.txt', val_loss_r)"""
