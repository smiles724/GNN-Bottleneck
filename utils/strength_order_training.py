import matplotlib.pyplot as plt

import torch


def strength_order_training(data='qm7', n_sample=10, fontsize=28):
    J_ = []
    for data in ['qm7', 'qm8', 'hamiltonian', 'newtonian']:

        J_.append(torch.load(f'../save/J_{data}_egnn.pt'))

    fig, ax = plt.subplots(1, 4, figsize=(32, 7))
    plt.rcParams.update({'font.size': fontsize})
    ax[0].set_ylabel('Strength', fontsize=fontsize)

    for k, data in enumerate(['qm7', 'qm8', 'hamiltonian', 'newtonian']):
        m_start = 2
        if data not in ['newtonian', 'md']: m_start += 1
        if data in ['hamiltonian', 'newtonian']: n_sample *= 10
        if data == 'qm7':
            epochs = [0, 10, 30, 50]
        elif data == 'qm8':
            epochs = [0, 10, 30, 100]
        elif data == 'hamiltonian':
            epochs = [0, 10, 20, 100]
        else:
            epochs = [0, 10, 30, 100]

        J = J_[k]
        for i in range(len(J['strength'])):
            if J['epoch'][i] in epochs:
                if data in ['qm8']:
                    J_tmp = [[n + 3, 0] for n in range(max([len(x) for x in J['strength'][i]]))]
                    for m in range(1, len(J['strength'][i])):
                        for j in range(len(J['strength'][i][m])):
                            J_tmp[j][1] += J['strength'][i][m][j][1]
                    J['strength'][i] = J_tmp

                strength = [j[1] for j in J['strength'][i]]
                strength = [x / sum(strength) for x in strength]
                ax[k].plot([j[0] / J['strength'][i][-1][0] for j in J['strength'][i]], strength, marker='o', label='epoch ' + str(J['epoch'][i]))
        ax[k].tick_params(labelsize=fontsize)
        ax[k].tick_params(labelsize=fontsize)
        ax[k].legend()
        ax[k].grid()
        ax[k].set_title(f'{data.upper()}')
        ax[k].set_xlabel('Order / n', fontsize=fontsize)
        if data == 'hamiltonian':
            ax[k].set_ylim([-0.05, 0.6])
    fig.tight_layout()
    plt.savefig(f'../../../../J_training.pdf', bbox_inches='tight')


if __name__ == '__main__':
    strength_order_training()




