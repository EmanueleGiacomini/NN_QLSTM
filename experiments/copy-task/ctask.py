"""
ctask.py

Synthetic memory copy-task as sanity check.
It consists of an input sequence of a lenght L, composed of S different symbols
followed by a sequence of time-lags or blanks of size T, and ended by a delimiter
that announces the beginning of the copy operation
"""

import os
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from qlstm import LSTM, QLSTM


def tovar(x, cuda):
    """Transform x in a tensor stored in RAM or VRAM based on cuda value"""
    if cuda:
        return Variable(torch.FloatTensor(x).cuda())
    else:
        return Variable(torch.FloatTensor(x.astype(np.float64)))


def generateTask(batch, seq_len, feat_size, blank_size, embedding):
    data = []
    labels = []
    for i in range(batch):
        # Generate a single batch
        # Symbols contained in the sequence
        blank = feat_size
        delim = feat_size + 1
        # Embedding symbols
        blank_emb = torch.tensor(blank, dtype=torch.long)
        blank_emb = embedding(blank_emb).data.numpy()
        delim_emb = torch.tensor(delim, dtype=torch.long)
        delim_emb = embedding(delim_emb).data.numpy()

        # Generate seq_len random indices to be embedded
        feat_idx_lst = []
        feat_lst = []
        target = []
        for _ in range(seq_len):
            rng_idx = np.random.randint(feat_size, size=1)
            feat = torch.tensor(rng_idx, dtype=torch.long)
            feat = embedding(feat).data.numpy()[0]
            # Store embedded result in feat_lst
            feat_lst.append(feat)
            # Store rng_idx for labelling
            feat_idx_lst.append(rng_idx)
            # Store in target blank symbols
            target.append(blank)

        # Generate blank_size blank embeddings into feat_lst
        for _ in range(blank_size - 1):
            feat_lst.append(blank_emb)
            target.append(blank)

        # Append the last blank element to feat_lst and
        # the delimiter symbol to the sequence
        target.append(blank)
        feat_lst.append(delim_emb)

        # Fill target with the generated rng_idx's
        for rng_idx in feat_idx_lst:
            target.append(rng_idx[0])
            feat_lst.append(delim_emb)

        # Add current sequence to data and labels
        data.append(feat_lst)
        labels.append(target)

    return np.array(data), np.array(labels)


""" Experiment setup """
CUDA = True
N_BATCH_TRAIN = 10
SEQ_LENGTH = 10
FEAT_SIZE = 8
BLANK_SIZE = 10
EPOCHS = 2000
QHIDDEN_SIZE = 80
HIDDEN_SIZE = 40


if __name__ == '__main__':

    # Create out folder if it doesnt exists
    if not os.path.isdir('out'):
        os.system('mkdir out')

    if CUDA:
        net_r = LSTM(FEAT_SIZE, HIDDEN_SIZE, CUDA).cuda()
        net_q = QLSTM(FEAT_SIZE, QHIDDEN_SIZE, CUDA).cuda()
    else:
        net_r = LSTM(FEAT_SIZE, HIDDEN_SIZE, CUDA)
        net_q = QLSTM(FEAT_SIZE, QHIDDEN_SIZE, CUDA)

    emb = nn.Embedding(FEAT_SIZE + 2, FEAT_SIZE, max_norm=1.0)

    no_params_q = sum(p.numel() for p in net_q.parameters() if p.requires_grad)
    no_params_r = sum(p.numel() for p in net_r.parameters() if p.requires_grad)
    print(f'QLSTM Trainable parameters: {no_params_q}')
    print(f'LSTM Trainable parameters: {no_params_r}')

    """Training Loop"""
    acc_r = []
    acc_q = []
    loss_r = []
    loss_q = []
    for epoch in range(EPOCHS):
        x_train, y_train = generateTask(N_BATCH_TRAIN, SEQ_LENGTH, FEAT_SIZE, BLANK_SIZE, emb)

        # x_train shape must be equal to (SEQ_LENGTH, BATCH_SIZE, FEATURE_SIZE) for the networks
        x_train = x_train.reshape((BLANK_SIZE + (2 * SEQ_LENGTH), N_BATCH_TRAIN, FEAT_SIZE))
        x_train_var = tovar(x_train, CUDA)
        y_train_var = tovar(y_train, CUDA)
        # LSTM TRAINING
        net_r.zero_grad()
        p = net_r.forward(x_train_var)
        # y_pred shape: (SEQ_LENGTH * BATCH_SIZE, FEATURE_SIZE)
        y_pred_r = p.view(-1, FEAT_SIZE + 1)

        # Target shape to (BATCH_SIZE)
        targets = y_train_var.view(-1)
        loss = nn.CrossEntropyLoss()
        val_loss = loss(y_pred_r, targets.long())

        val_loss.backward()
        net_r.adam.step()

        # Store ACC and LOSS metrics
        p = p.cpu().data.numpy()
        shape = np.argmax(p, axis=2).shape
        p = np.reshape(np.argmax(p, axis=2), shape[0] * shape[1])
        targets = targets.cpu().data.numpy()
        acc = np.sum(p == targets) / y_train.size

        if epoch % 5 == 0:
            acc_r.append(acc)
            loss_r.append(val_loss)
        if epoch % 10 == 0:
            print(f'LSTM It: {epoch} | Train Loss = {float(val_loss.data)} | Train Acc = {acc}')
        # QLSTM TRAINING
        net_q.zero_grad()
        p = net_q.forward(x_train_var)
        y_pred_q = p.view(-1, FEAT_SIZE + 1)
        targets = y_train_var.view(-1)
        loss = nn.CrossEntropyLoss()
        val_loss = loss(y_pred_q, targets.long())

        val_loss.backward()
        net_q.adam.step()

        p = p.cpu().data.numpy()
        shape = np.argmax(p, axis=2).shape
        p = np.reshape(np.argmax(p, axis=2), shape[0] * shape[1])
        targets = targets.cpu().data.numpy()
        acc = np.sum(p == targets) / y_train.size

        if epoch % 5 == 0:
            acc_q.append(acc)
            loss_q.append(val_loss)
        if epoch % 10 == 0:
            print(f'QLSTM It: {epoch} | Train Loss = {float(val_loss.data)} | Train Acc = {acc}')

    print('Training phase ended.')
    np.savetxt(f'out/memory_task_acc_q_{BLANK_SIZE}.txt', acc_q)
    np.savetxt(f'out/memory_task_acc_r_{BLANK_SIZE}.txt', acc_r)
    np.savetxt(f'out/memory_task_loss_q_{BLANK_SIZE}.txt', loss_q)
    np.savetxt(f'out/memory_task_loss_r_{BLANK_SIZE}.txt', loss_r)

    exit(0)
