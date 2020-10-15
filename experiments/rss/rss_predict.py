"""rss_predict.py
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from qlstm import LSTM, QLSTM, TLSTM
from qgru import GRU, QGRU
from os.path import join
import time
import random
from experiments.har.har import load_cb


def tovar(x, cuda):
    """Transform x in a tensor stored in RAM or VRAM based on cuda value"""
    if cuda:
        return Variable(torch.FloatTensor(x.float()).cuda())
    else:
        return Variable(torch.FloatTensor(x.float()))


BASE_NAME = 'MovementAAL_'
DATA_BASE_NAME = BASE_NAME + 'RSS_'


def read_labels(path: str):
    df = pd.read_csv(join(path, BASE_NAME + 'target.csv'))
    return df[' class_label'].to_numpy()


def read_sequences(path: str):
    seq_lst = []
    for i in range(1, 315):
        seq_lst.append(pd.read_csv(join(path, DATA_BASE_NAME + str(i) + '.csv')))
    return seq_lst


def extract_seq_data(seq_lst: [pd.DataFrame]) -> [np.array]:
    arr_lst = []
    for seq in seq_lst:
        arr_lst.append(seq.to_numpy())
    return arr_lst


def preprocess_seq_lst(seq_lst: [np.array], labels: np.array, seq_len=None) -> np.array:
    """Preprocess seq_lst into a single np.array.
    Each row contains min_len timesteps """
    if seq_len is None:
        min_len = min(x.shape[0] for x in seq_lst)
    else:
        min_len = seq_len
    X = []
    y = []
    for i, seq in enumerate(seq_lst):
        # no. subsequences to be extracted = seq.shape[0] // min_len
        for j in range(seq.shape[0] // min_len):
            X.append(seq[j * min_len:(j + 1) * min_len])
            y.append(labels[i])
    return np.array(X), np.array(y)


CUDA = True
FEAT_SIZE = 4
# SEQ_LEN = 19 # Minimum seq length
SEQ_LEN = 19
QHIDDEN_SIZE = 32
HIDDEN_SIZE = 16
OUTPUT_SIZE = 1
EPOCHS = 300
BATCH_SIZE = 64


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
        p = net(x)
        y_pred = p[-1, :, :].view(-1)

        loss = nn.SoftMarginLoss()
        val_loss = loss(y_pred, y)

        val_loss.backward()
        net.adam.step()
        with torch.no_grad():
            y_pred = np.array([1. if x > 0 else -1. for x in y_pred])
            train_acc += sum(y_pred == y.cpu().data.numpy()) / y_pred.shape[0]
            train_loss += val_loss.cpu().data.numpy()
        epoch_it += 1
        t1 = time.perf_counter() - t0
        #load_cb(epoch_it, final_it, 'Training', t1)
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
            p = net(x)
            y_pred = p[-1, :, :].view(-1)
            loss = nn.SoftMarginLoss()
            val_loss = loss(y_pred, y)
            test_loss += val_loss.cpu().data.numpy()
            y_pred = np.array([1. if x > 0 else -1 for x in y_pred])
            test_acc += sum(y_pred == y.cpu().data.numpy()) / y_pred.shape[0]
            epoch_it += 1
            t1 = time.perf_counter() - t0
            #load_cb(epoch_it, final_it, 'Testing', t1)
        test_acc /= epoch_it
        test_loss /= epoch_it
    return train_acc, train_loss, test_acc, test_loss

TEST_MODELS = {'LSTM16'  : LSTM(FEAT_SIZE, 16, CUDA, OUTPUT_SIZE).cuda(),
               'QLSTM32' : QLSTM(FEAT_SIZE, 32, CUDA, OUTPUT_SIZE).cuda(),
               'GRU16'   : GRU(FEAT_SIZE, 16, CUDA, OUTPUT_SIZE).cuda(),
               'QGRU32'  : QGRU(FEAT_SIZE, 32, CUDA, OUTPUT_SIZE).cuda(),
               'TLSTM32' : TLSTM(FEAT_SIZE, 32, CUDA, OUTPUT_SIZE).cuda(),
               'LSTM8': LSTM(FEAT_SIZE, 8, CUDA, OUTPUT_SIZE).cuda(),
               'QLSTM16': QLSTM(FEAT_SIZE, 16, CUDA, OUTPUT_SIZE).cuda(),
               'QLSTM8': QLSTM(FEAT_SIZE, 8, CUDA, OUTPUT_SIZE).cuda()}

RESULT_DICT = {}

if __name__ == '__main__':
    torch.manual_seed(40)
    torch.cuda.manual_seed(40)
    torch.cuda.manual_seed_all(40)
    np.random.seed(40)
    torch.manual_seed(40)
    random.seed(40)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    labels = read_labels('./dataset')
    seq_lst = extract_seq_data(read_sequences('./dataset'))
    X, y = preprocess_seq_lst(seq_lst, labels, SEQ_LEN)
    # X, y = experimental_preprocess(seq_lst, labels)

    # exit(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    train_data = TensorDataset(torch.from_numpy(X_train).float(),
                               torch.from_numpy(y_train))
    test_data = TensorDataset(torch.from_numpy(X_test).float(),
                              torch.from_numpy(y_test))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE,
                                              shuffle=True)

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
    pd.DataFrame(RESULT_DICT).to_csv('out/results.csv')



















"""
    if CUDA:
        net_q = QLSTM(FEAT_SIZE, QHIDDEN_SIZE, CUDA, OUTPUT_SIZE).cuda()
        net_r = LSTM(FEAT_SIZE, HIDDEN_SIZE, CUDA, OUTPUT_SIZE).cuda()
        # net_r = GTModel(FEAT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, CUDA).cuda()
    else:
        net_q = QLSTM(FEAT_SIZE, QHIDDEN_SIZE, CUDA, OUTPUT_SIZE)
        net_r = LSTM(FEAT_SIZE, HIDDEN_SIZE, CUDA, OUTPUT_SIZE)
        # net_r = GTModel(FEAT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, CUDA)

    nb_param_q = sum(p.numel() for p in net_q.parameters() if p.requires_grad)
    nb_param_p = sum(p.numel() for p in net_r.parameters() if p.requires_grad)

    print(f'(QLSTM) Number of trainable parameters : {nb_param_q}')
    print(f'(LSTM) Number of trainable parameters : {nb_param_p}')
"""
"""
    acc_r = []
    acc_q = []
    loss_r = []
    loss_q = []
    val_acc_r = []
    val_acc_q = []
    val_loss_r = []
    val_loss_q = []
    for epoch in range(EPOCHS):
        it = 0
        acc_loss_r = 0.0
        acc_acc_r = 0.0
        acc_loss_q = 0.0
        acc_acc_q = 0.0
        for (x, y) in train_loader:
            # x shape: (SEQ_LEN, BATCH_SIZE, FEAT_SIZE)
            x = x.transpose(0, 1).contiguous()
            # x = x.reshape((SEQ_LEN, -1, FEAT_SIZE))
            x, y = tovar(x, CUDA), tovar(y, CUDA)

            net_r.zero_grad()
            p = net_r(x)
            y_pred = p[-1, :, :].view(-1)

            loss = nn.SoftMarginLoss()
            val_loss = loss(y_pred, y)
            # Optimizer step
            val_loss.backward()
            net_r.adam.step()

            # Store LOSS and ACC metrics
            acc_loss_r += val_loss
            y_pred = np.array([1. if x > 0 else -1. for x in y_pred])

            acc = sum(y_pred == y.cpu().data.numpy()) / y_pred.shape[0]
            acc_acc_r += acc

            net_q.zero_grad()
            p = net_q(x)
            y_pred = p[-1, :, :].view(-1)

            loss = nn.SoftMarginLoss()
            val_loss = loss(y_pred, y)
            # Optimizer step
            val_loss.backward()
            net_q.adam.step()

            y_pred = np.array([1. if x > 0 else -1. for x in y_pred])
            acc = sum(y_pred == y.cpu().data.numpy()) / y_pred.shape[0]
            acc_acc_q += acc

            it += 1

            # Store LOSS and ACC metrics
            acc_loss_q += val_loss

        # Execute validation step
        vacc_q = 0.0
        vacc_r = 0.0
        vloss_q = 0.0
        vloss_r = 0.0
        val_it = 0
        with torch.no_grad():
            for (x, y) in test_loader:
                # x shape: (SEQ_LEN, BATCH_SIZE, FEAT_SIZE)
                x = x.transpose(0, 1).contiguous()
                x, y = tovar(x, CUDA), tovar(y, CUDA)

                p = net_r(x)

                y_pred = p[-1, :, :].view(-1)
                loss = nn.SoftMarginLoss()
                val_loss = loss(y_pred, y)
                vloss_r += val_loss

                y_pred = np.array([1. if x > 0 else -1 for x in y_pred])
                vacc_r += sum(y_pred == y.cpu().data.numpy()) / y_pred.shape[0]

                p = net_q(x)
                y_pred = p[-1, :, :].view(-1)
                loss = nn.SoftMarginLoss()
                val_loss = loss(y_pred, y)
                vloss_q += val_loss
                y_pred = np.array([1. if x > 0 else -1 for x in y_pred])
                vacc_q += sum(y_pred == y.cpu().data.numpy()) / y_pred.shape[0]
                val_it += 1

        acc_loss_r /= it
        acc_loss_q /= it
        acc_acc_r /= it
        acc_acc_q /= it
        vacc_r /= val_it
        vacc_q /= val_it
        vloss_r /= val_it
        vloss_q /= val_it

        if epoch % 5 == 0:
            loss_r.append(acc_loss_r)
            loss_q.append(acc_loss_q)
            acc_r.append(acc_acc_r)
            acc_q.append(acc_acc_q)
            val_acc_r.append(vacc_r)
            val_acc_q.append(vacc_q)
            val_loss_r.append(vloss_r)
            val_loss_q.append(vloss_q)
        if epoch % 10 == 0:
            print(
                f' LSTM epoch {epoch}: loss={acc_loss_r:.4f}, acc={acc_acc_r:.3f}, val_loss={vloss_r:.4f}, val_acc={vacc_r:.3f}')
            print(
                f'QLSTM epoch {epoch}: loss={acc_loss_q:.4f}, acc={acc_acc_q:.3f}, val_loss={vloss_q:.4f}, val_acc={vacc_q:.3f}')
    np.savetxt(f'out/rss_task_acc_q.txt', acc_q)
    np.savetxt(f'out/rss_task_acc_r.txt', acc_r)
    np.savetxt(f'out/rss_task_loss_q.txt', loss_q)
    np.savetxt(f'out/rss_task_loss_r.txt', loss_r)
    np.savetxt(f'out/rss_task_val_acc_q.txt', val_acc_q)
    np.savetxt(f'out/rss_task_val_acc_r.txt', val_acc_r)
    np.savetxt(f'out/rss_task_val_loss_q.txt', val_loss_q)
    np.savetxt(f'out/rss_task_val_loss_r.txt', val_loss_r)
"""