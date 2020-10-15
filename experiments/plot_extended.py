"""plot_ext
"""

import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter


def load_data() -> {str: pd.DataFrame}:
    def copy_task_load_cb(path: str) -> pd.DataFrame:
        ...

    def har_task_load_cb(path: str) -> pd.DataFrame:
        ...

    def rss_task_load_cb(path: str, name='results.csv') -> pd.DataFrame:
        data_raw = pd.read_csv(os.path.join(path, name))
        labels = data_raw.keys()
        features = data_raw['Unnamed: 0']
        labels = labels[1:]
        data = {}
        for key in labels:
            data[key] = {}
            for i in range(4):
                data[key][features[i]] = np.array(eval(data_raw[key][i]))
        return pd.DataFrame(data)

    res_dict = {}
    DATA_FOLDER_LST = ['./copy-task/out/',
                       './har/out/',
                       './rss/out/']
    CB_DICT = {'copy': (0, None),
               'har': (1, rss_task_load_cb),
               'rss': (2, rss_task_load_cb),
               'rss_overfit': (2, rss_task_load_cb)}
    for k in CB_DICT.keys():
        data_path = DATA_FOLDER_LST[CB_DICT[k][0]]
        data_load_cb = CB_DICT[k][1]
        if data_load_cb is not None:
            if k == 'rss_overfit':
                res_dict[k] = data_load_cb(data_path, 'results_base.csv')
            else:
                res_dict[k] = data_load_cb(data_path)

    return res_dict


def plot_rss_task(res_dict):
    fig, axs = plt.subplots(1, 2, figsize=(8, 3))
    x_lab = np.zeros(res_dict['rss']['LSTM16']['train_acc'].shape[0])
    for i in range(x_lab.shape[0]):
        x_lab[i] = i
    axs[0].set_title('val accuracy')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('accuracy')
    for mname in res_dict['rss'].keys():
        model = res_dict['rss'][mname]
        axs[0].plot(x_lab, savgol_filter(model['val_acc'], 25, 3), label=mname)
    axs[0].legend()
    axs[1].set_title('val loss')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('loss')
    for mname in res_dict['rss'].keys():
        model = res_dict['rss'][mname]
        axs[1].plot(x_lab, savgol_filter(model['val_loss'], 25, 3), label=mname)
    axs[1].set_yscale('log')
    plt.tight_layout()
    plt.savefig('./images/rss_task.png')


def plot_task(res_dict, experiment: str, figsize=(8, 3), savgol_window=51, early_cut=None, subname=None,
              y_scale=None):
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    if experiment not in res_dict:
        print('Invalid experiment.')
        return
    res_dict = res_dict[experiment]
    if early_cut:
        x_lab = np.zeros(early_cut)
    else:
        x_lab = np.zeros(res_dict[res_dict.keys()[0]]['train_acc'].shape[0])
    for i in range(x_lab.shape[0]):
        x_lab[i] = i
    axs[0].set_title('val accuracy')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('accuracy')
    for mname in res_dict.keys():
        model = res_dict[mname]
        data = model['val_acc']
        if early_cut:
            data = data[:early_cut]
        axs[0].plot(x_lab, savgol_filter(data, savgol_window, 3), label=mname)
    axs[0].legend()
    axs[1].set_title('val loss')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('loss')
    if y_scale == 'log':
        axs[1].set_yscale('log')
    for mname in res_dict.keys():
        model = res_dict[mname]
        data = model['val_loss']
        if early_cut:
            data = data[:early_cut]
        axs[1].plot(x_lab, savgol_filter(data, savgol_window, 3), label=mname)
    plt.tight_layout()
    if subname:
        plt.savefig('./images/' + experiment + '_' + subname + '.png')
    else:
        plt.savefig('./images/' + experiment + '.png')


def table_task(res_dict, experiment: str):
    if experiment not in res_dict:
        print('Invalid experiment.')
        return
    data = res_dict[experiment].T
    for i in data.keys():
        for j in data[i].keys():
            data[i][j] = np.min(data[i][j]) if 'loss' in i else np.max(data[i][j])
    print(data.to_latex())

def rss_weight_decay_comparison(res_dict):
    fig, axs = plt.subplots(1, 1, figsize=(4, 4))
    x_lab = np.zeros(res_dict['rss']['QLSTM32']['val_loss'].shape[0])
    for i in range(x_lab.shape[0]):
        x_lab[i] = i
    axs.set_title('Validation Loss')
    axs.set_xlabel('epochs')
    axs.plot(x_lab, savgol_filter(res_dict['rss_overfit']['QLSTM32']['val_loss'][:300], 21, 3), label='no regularization')
    axs.plot(x_lab, savgol_filter(res_dict['rss']['QLSTM32']['val_loss'][:300], 21, 3), label='weight_decay=1e-3')
    axs.legend()
    plt.tight_layout()
    plt.savefig('./images/rss_weight_decay.png')


if __name__ == '__main__':
    res_dict = load_data()
    # plot_rss_task(res_dict)
    plot_task(res_dict, 'har', savgol_window=5)
    table_task(res_dict, 'har')
    plot_task(res_dict, 'rss', savgol_window=21)
    plot_task(res_dict, 'rss', savgol_window=13, early_cut=150, subname='cut_150', y_scale='log')
    table_task(res_dict, 'rss')
    rss_weight_decay_comparison(res_dict)
    # table_rss_task(res_dict)
