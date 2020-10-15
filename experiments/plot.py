"""
plot.py
Generates pyplot images for the given experiments
"""

import os

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


def load_data() -> {str: {str: np.array}}:
    def copy_task_load_data_cb(path: str) -> {str, np.array}:
        data = {}
        FILE_LST = ['acc_q_10', 'acc_q_50', 'acc_q_100',
                    'acc_r_10', 'acc_r_50', 'acc_r_100',
                    'acc_t_10', 'acc_t_100', 'acc_q_150', 'acc_r_150',
                    'loss_r_10', 'loss_r_50', 'loss_r_100',
                    'loss_q_10', 'loss_q_50', 'loss_q_100',
                    'loss_t_10', 'loss_t_100', 'loss_q_150', 'loss_r_150']
        for file in FILE_LST:
            fpath = os.path.join(path, 'memory_task_' + file + '.txt')
            data[file] = np.loadtxt(fpath)
        return data

    def copy_task_gru_load_data_cb(path: str) -> {str, np.array}:
        data = {}
        FILE_LST = ['acc_q_10', 'acc_q_50', 'acc_q_100',
                    'acc_r_10', 'acc_r_50', 'acc_r_100',
                    'loss_r_10', 'loss_r_50', 'loss_r_100',
                    'loss_q_10', 'loss_q_50', 'loss_q_100']
        for file in FILE_LST:
            fpath = os.path.join(path, 'memory_task_gru_' + file + '.txt')
            data[file] = np.loadtxt(fpath)
        return data

    def har_task_1_load_data_cb(path: str):
        data = {}
        FILE_LST = ['acc_r', 'acc_q', 'loss_r', 'loss_q',
                    'val_acc_r', 'val_acc_q', 'val_loss_r', 'val_loss_q']
        for file in FILE_LST:
            fpath = os.path.join(path, '1_har_task_' + file + '.txt')
            data[file] = np.loadtxt(fpath)
        return data

    def har_task_2_load_data_cb(path: str):
        data = {}
        FILE_LST = ['acc_r', 'acc_q', 'loss_r', 'loss_q',
                    'val_acc_r', 'val_acc_q', 'val_loss_r', 'val_loss_q',
                    'acc_t', 'acc_gru', 'loss_t', 'loss_gru',
                    'val_acc_t', 'val_acc_gru', 'val_loss_t', 'val_loss_gru', ]
        for file in FILE_LST:
            fpath = os.path.join(path, '2_har_task_' + file + '.txt')
            data[file] = np.loadtxt(fpath)
        return data

    def har_task_3_load_data_cb(path: str):
        data = {}
        FILE_LST = ['acc_r', 'acc_q', 'loss_r', 'loss_q',
                    'val_acc_r', 'val_acc_q', 'val_loss_r', 'val_loss_q']
        for file in FILE_LST:
            fpath = os.path.join(path, '3_har_task_' + file + '.txt')
            data[file] = np.loadtxt(fpath)
        return data

    def rss_task_load_data_cb(path: str):
        data = {}
        FILE_LST = ['acc_r', 'acc_q', 'loss_r', 'loss_q',
                    'val_acc_r', 'val_acc_q', 'val_loss_r', 'val_loss_q']
        for file in FILE_LST:
            fpath = os.path.join(path, 'rss_task_' + file + '.txt')
            data[file] = np.loadtxt(fpath)
        return data

    data_dict = {}

    DATA_FOLDER_LST = ['./copy-task/out/',
                       './har/out/',
                       './rss/out/']
    DATA_CB_DICT = {'copy': (0, copy_task_load_data_cb),
                    'copy_gru': (0, copy_task_gru_load_data_cb),
                    'har_1': (1, har_task_1_load_data_cb),
                    'har_2': (1, har_task_2_load_data_cb),
                    'har_3': (1, har_task_3_load_data_cb),
                    'rss': (2, rss_task_load_data_cb)}
    for k in DATA_CB_DICT.keys():
        data_path = DATA_FOLDER_LST[DATA_CB_DICT[k][0]]
        data_load_cb = DATA_CB_DICT[k][1]
        data_dict[k] = data_load_cb(data_path)
    return data_dict


def plot_copy_task_cmpl(data_dict, gru=False):
    dkey = 'copy' if gru is False else 'copy_gru'
    fig, axs = plt.subplots(2, 3, figsize=(6, 3))
    x_lab = np.zeros(data_dict['copy']['acc_q_10'].shape[0])
    for i in range(x_lab.shape[0]):
        x_lab[i] = 5 * i
    axs[0, 0].set_title('T = 10')
    axs[0, 0].set_ylabel('accuracy')
    axs[0, 0].set_xlabel('epochs')
    axs[0, 0].plot(x_lab, data_dict[dkey]['acc_q_10'], label='QLSTM')
    axs[0, 0].plot(x_lab, data_dict[dkey]['acc_r_10'], label='LSTM')
    axs[0, 0].legend()
    axs[0, 1].set_title('T = 50')
    axs[0, 1].plot(x_lab, data_dict[dkey]['acc_q_50'])
    axs[0, 1].plot(x_lab, data_dict[dkey]['acc_r_50'])
    axs[0, 2].set_title('T = 150')
    axs[0, 2].plot(x_lab, data_dict[dkey]['acc_q_150'])
    axs[0, 2].plot(x_lab, data_dict[dkey]['acc_r_150'])
    axs[1, 0].set_ylabel('loss')
    axs[1, 0].set_xlabel('epochs')
    axs[1, 0].plot(x_lab, data_dict[dkey]['loss_q_10'])
    axs[1, 0].plot(x_lab, data_dict[dkey]['loss_r_10'])
    axs[1, 0].set_yscale('log')
    axs[1, 1].plot(x_lab, data_dict[dkey]['loss_q_50'])
    axs[1, 1].plot(x_lab, data_dict[dkey]['loss_r_50'])
    axs[1, 1].set_yscale('log')
    axs[1, 2].plot(x_lab, data_dict[dkey]['loss_q_150'])
    axs[1, 2].plot(x_lab, data_dict[dkey]['loss_r_150'])
    axs[1, 2].set_yscale('log')
    plt.tight_layout()
    if gru is False:
        plt.savefig('./images/copy_task_cmpl.png')
    else:
        plt.savefig('./images/copy_task_gru_cmpl.png')


def plot_copy_task_comparison(data_dict):
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    x_lab = np.zeros(data_dict['copy']['acc_q_10'].shape[0])
    for i in range(x_lab.shape[0]):
        x_lab[i] = 5 * i
    axs[0].plot(x_lab, data_dict['copy']['acc_q_100'], label='QLSTM')
    axs[0].plot(x_lab, data_dict['copy_gru']['acc_q_100'], label='QGRU')
    axs[0].plot(x_lab, data_dict['copy']['acc_t_100'], label='TLSTM')
    axs[0].legend()
    axs[1].plot(x_lab, data_dict['copy']['loss_q_100'], label='QLSTM')
    axs[1].plot(x_lab, data_dict['copy_gru']['loss_q_100'], label='QGRU')
    axs[1].plot(x_lab, data_dict['copy']['loss_t_100'], label='TLSTM')
    plt.tight_layout()

    plt.savefig('./images/copy_task_compare.png')


def plot_copy_task_gru(data_dict):
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    x_lab = np.zeros(data_dict['copy_gru']['acc_r_100'].shape[0])
    for i in range(x_lab.shape[0]):
        x_lab[i] = 5 * i
    axs[0].set_title('Accuracy (T=100)')
    axs[0].plot(x_lab, data_dict['copy_gru']['acc_r_100'], label='GRU')
    axs[0].plot(x_lab, data_dict['copy_gru']['acc_q_100'], label='QGRU')
    axs[0].legend()
    axs[1].set_title('Loss')
    axs[1].plot(x_lab, data_dict['copy_gru']['loss_r_100'], label='GRU')
    axs[1].plot(x_lab, data_dict['copy_gru']['loss_q_100'], label='QGRU')
    plt.tight_layout()

    plt.savefig('./images/copy_task_gru.png')


def plot_har_task(data_dict, exp_num=1):
    dkey = 'har_' + str(exp_num)
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    x_lab = np.zeros(data_dict[dkey]['acc_q'].shape[0])
    for i in range(x_lab.shape[0]):
        x_lab[i] = i
    axs[0].plot(x_lab, data_dict[dkey]['acc_r'], label='LSTM')
    axs[0].plot(x_lab, data_dict[dkey]['acc_q'], label='QLSTM')
    axs[0].legend()
    axs[1].plot(x_lab, data_dict[dkey]['loss_r'])
    axs[1].plot(x_lab, data_dict[dkey]['loss_q'])
    plt.tight_layout()

    plt.savefig('./images/har_task_' + str(exp_num) + '.png')


def plot_har_2_task_comparison(data_dict):
    dkey = 'har_2'
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    x_lab = np.zeros(data_dict[dkey]['acc_q'].shape[0])
    for i in range(x_lab.shape[0]):
        x_lab[i] = i
    axs[0].set_title('Accuracy')
    axs[0].plot(x_lab, data_dict[dkey]['val_acc_q'], label='QLSTM')
    axs[0].plot(x_lab, data_dict[dkey]['val_acc_t'], label='TLSTM')
    axs[0].plot(x_lab, data_dict[dkey]['val_acc_gru'], label='QGRU')
    axs[0].legend()
    axs[1].set_title('Loss')
    axs[1].plot(x_lab, data_dict[dkey]['val_loss_q'], label='QLSTM')
    axs[1].plot(x_lab, data_dict[dkey]['val_loss_t'], label='TLSTM')
    axs[1].plot(x_lab, data_dict[dkey]['val_loss_gru'], label='QGRU')
    plt.tight_layout()

    plt.savefig('./images/har_task_2_comparison.png')


def plot_rss_task(data_dict):
    dkey = 'rss'
    fig, axs = plt.subplots(1, 2, figsize=(8, 3))
    x_lab = np.zeros(data_dict[dkey]['acc_q'].shape[0])
    for i in range(x_lab.shape[0]):
        x_lab[i] = i
    axs[0].plot(data_dict[dkey]['acc_r'], label='LSTM')
    axs[0].plot(data_dict[dkey]['acc_q'], label='QLSTM')
    axs[0].legend()
    axs[1].plot(data_dict[dkey]['loss_r'])
    axs[1].plot(data_dict[dkey]['loss_q'])
    plt.tight_layout()

    plt.savefig('./images/rss_task.png')


def table_har_complete(data_dict):
    """Print results of HAR experiments in the form
    of a table."""
    res_dict = {}
    res_dict['har_1'] = {}
    res_dict['har_2'] = {}
    res_dict['har_3'] = {}
    res_dict['har_1']['acc_q'] = np.max(data_dict['har_1']['acc_q'])
    res_dict['har_1']['acc_r'] = np.max(data_dict['har_1']['acc_r'])
    res_dict['har_1']['loss_q'] = np.min(data_dict['har_1']['loss_q'])
    res_dict['har_1']['loss_r'] = np.min(data_dict['har_1']['loss_r'])

    res_dict['har_2']['acc_q'] = np.max(data_dict['har_2']['acc_q'])
    res_dict['har_2']['acc_r'] = np.max(data_dict['har_2']['acc_r'])
    res_dict['har_2']['acc_gru'] = np.max(data_dict['har_2']['acc_gru'])
    res_dict['har_2']['acc_t'] = np.max(data_dict['har_2']['acc_t'])
    res_dict['har_2']['loss_q'] = np.min(data_dict['har_2']['loss_q'])
    res_dict['har_2']['loss_r'] = np.min(data_dict['har_2']['loss_r'])
    res_dict['har_2']['loss_gru'] = np.min(data_dict['har_2']['loss_gru'])
    res_dict['har_2']['loss_t'] = np.min(data_dict['har_2']['loss_t'])

    res_dict['har_3']['acc_q'] = np.max(data_dict['har_3']['acc_q'])
    res_dict['har_3']['acc_r'] = np.max(data_dict['har_3']['acc_r'])
    res_dict['har_3']['loss_q'] = np.min(data_dict['har_3']['loss_q'])
    res_dict['har_3']['loss_r'] = np.min(data_dict['har_3']['loss_r'])

    print(pd.DataFrame(res_dict))


if __name__ == '__main__':
    # Create out folder if it doesnt exists
    if not os.path.isdir('images'):
        os.system('mkdir images')

    data_dict = load_data()

    table_har_complete(data_dict)

    plot_copy_task_cmpl(data_dict)
    plot_copy_task_cmpl(data_dict, gru=False)
    plot_copy_task_comparison(data_dict)
    plot_copy_task_gru(data_dict)
    plot_har_task(data_dict, 1)
    plot_har_task(data_dict, 2)
    plot_har_task(data_dict, 3)
    plot_har_2_task_comparison(data_dict)
    plot_rss_task(data_dict)

    exit(0)
"""
    plot_rss_task()
    plot_har_task()
    plot_har_task_2()
    plot_har_task_3()
    plot_gru_copy_task()
    exit(0)

    x_lab = np.zeros(400)
    for i in range(400):
        x_lab[i] = i * 5

    # first experiment [f]
    first_exp = str(10)
    second_exp = str(100)
    # second experiment [s]
    acc_q_f = np.loadtxt(f'./copy-task/out/memory_task_acc_q_{first_exp}.txt')
    loss_q_f = np.loadtxt(f'./copy-task/out/memory_task_loss_q_{first_exp}.txt')
    acc_r_f = np.loadtxt(f'./copy-task/out/memory_task_acc_r_{first_exp}.txt')
    loss_r_f = np.loadtxt(f'./copy-task/out/memory_task_loss_r_{first_exp}.txt')
    acc_q_s = np.loadtxt(f'./copy-task/out/memory_task_acc_q_{second_exp}.txt')
    loss_q_s = np.loadtxt(f'./copy-task/out/memory_task_loss_q_{second_exp}.txt')
    acc_r_s = np.loadtxt(f'./copy-task/out/memory_task_acc_r_{second_exp}.txt')
    loss_r_s = np.loadtxt(f'./copy-task/out/memory_task_loss_r_{second_exp}.txt')

    # Normalize to [0, 100] the acc values
    acc_q_f = acc_q_f[:] * 100
    acc_r_f = acc_r_f[:] * 100
    acc_q_s = acc_q_s[:] * 100
    acc_r_s = acc_r_s[:] * 100

    fig, axs = plt.subplots(2, 2, figsize=(10, 5.5))

    axs[0, 0].plot(x_lab, loss_q_f, label='QLSTM')
    axs[0, 0].plot(x_lab, loss_r_f, label='LSTM')
    axs[1, 0].plot(x_lab, acc_q_f)
    axs[1, 0].plot(x_lab, acc_r_f)
    axs[0, 1].plot(x_lab, loss_q_s)
    axs[0, 1].plot(x_lab, loss_r_s)
    axs[1, 1].plot(x_lab, acc_q_s)
    axs[1, 1].plot(x_lab, acc_r_s)

    axs[0, 0].set_title('T=10')
    axs[0, 1].set_title('T=100')

    axs[0, 0].legend()
    plt.savefig('images/copy-task.png')"""

exit(0)
