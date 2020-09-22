"""
plot.py
Generates pyplot images for the given experiments
"""

import os
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':

    # Create out folder if it doesnt exists
    if not os.path.isdir('images'):
        os.system('mkdir images')

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
    plt.savefig('images/copy-task.png')

    exit(0)