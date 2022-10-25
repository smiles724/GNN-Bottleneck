import matplotlib.pyplot as plt

import torch
import pandas as pd
from sklearn.manifold import TSNE


def strength_order_plot(data='qm7', save=True, n_sample=10, fontsize=28, m_start=2, average=False):
    J_m, J_d = [], []
    if data in ['qm7', 'qm8']: m_start += 1
    if data in ['hamiltonian', 'newtonian']: n_sample *= 10
    for m in ['egnn', 'molformer']:
        J_m.append(torch.load(f'../save/strength_{data}_{m}_n{n_sample}.pt')[0])
        J_d.append(torch.load(f'../save/strength_{data}_{m}_n{n_sample}_random.pt')[0])
    print('Learned Strength for EGNN:', J_m[0])
    print('Data Strength for EGNN:', J_d[0])
    print('Learned Strength for Molformer:', J_m[1])
    print('Data Strength for Molformer:', J_d[1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.tick_params(labelsize=fontsize)
    ax2.tick_params(labelsize=fontsize)
    plt.rcParams.update({'font.size': fontsize})
    plt.rcParams['font.sans-serif'] = "Times New Roman"
    plt.rcParams["font.family"] = "Times New Roman"

    # 画一条dummy line
    from matplotlib.axis import Axis
    # ax13 = ax1.twiny()
    # ax13.set_xlabel('EGNN', fontsize=fontsize, loc='left')
    # Axis.set_label_coords(ax13.xaxis, 0.5, 0.95)
    # ax13.xaxis.set_label_coords(-0.2, 0.2)
    ax12 = ax1.twiny()
    ax1.set_title(f'EGNN')
    ax1.set_ylabel('Strength', fontsize=fontsize)
    # ax1.set_xlabel('Order / n', fontsize=fontsize, loc='right')
    ax1.set_xlabel('Order / n', fontsize=fontsize)
    if data in ['newtonian']:
        ax1.set_ylim([0.02, 0.2])
    elif data in ['md']:
        ax1.set_ylim([0.0, 0.2])
    else:
        ax1.set_ylim([-0.05, 0.7])
    max_order = 0
    if data not in ['hamiltonian', 'newtonian', 'md']:
        if average:
            for i, s in enumerate(J_m[0]):
                order = [i[0] for i in s]
                strength = [i[1] for i in s]
                ax1.plot(order, strength, marker='o', label=f'mol_{i + 1}')
                max_order = max(max(order), max_order)
        else:
            tmp = [0] * 100
            for s in J_m[0]:
                order = [i[0] for i in s]
                max_order = max(max(order), max_order)
                for i, j in enumerate(s):
                    tmp[i] += j[1]
            order = [i for i in range(3, max_order + 1)]
            order_n = [x / max_order for x in order]
            strength = tmp[:len(order)]
            strength = [x / sum(strength) for x in strength]
            ax1.plot(order_n, strength, marker='o',  label='$J^{(m)}$')
            ax12.plot(order, strength)

            tmp = [0] * 100
            for s in J_d[0]:
                order = [i[0] for i in s]
                max_order = max(max(order), max_order)
                for i, j in enumerate(s):
                    tmp[i] += j[1]
            order = [i for i in range(3, max_order + 1)]
            order_n = [x / max_order for x in order]
            strength = tmp[:len(order)]
            strength = [x / sum(strength) for x in strength]
            ax1.plot(order_n, strength, marker='o',  label='$J^{(m)}_D$')
    else:
        order = [x[0] for x in J_m[0]]
        max_order = max(order)
        order_n = [x / max_order for x in order]
        strength = [x[1] for x in J_m[0]]
        ax1.plot(order_n, strength, marker='o', label='$J^{(m)}$')
        ax12.plot(order, strength)

        order_n = [x[0] / max_order for x in J_d[0]]
        strength = [x[1] for x in J_d[0]]
        ax1.plot(order_n, strength, marker='o', label='$J^{(m)}_D$')
    ax1.set_xticks(list([0.2 * i for i in range(1, 6)]))
    if data not in ['QM7', 'QM8', 'md']:
        ax12.set_xticks(list(range(m_start, max_order + 1, 2)))
    else:
        ax12.set_xticks(list(range(m_start, max_order + 1, 3)))
    ax12.set_xlabel('Order', fontsize=fontsize)
    ax1.tick_params(axis='y')
    ax1.legend()
    ax1.grid()

    ax22 = ax2.twiny()
    ax2.set_title(f'Molformer')
    ax2.set_xlabel('Order / n', fontsize=fontsize)
    if data in ['newtonian']:
        ax2.set_ylim([0.05, 0.2])
    elif data in ['md']:
        ax2.set_ylim([0.0, 0.2])
    else:
        ax2.set_ylim([-0.05, 1.0])
    if data not in ['hamiltonian', 'newtonian', 'md']:
        if average:
            for i, s in enumerate(J_m[1]):
                order = [i[0] for i in s]
                strength = [i[1] for i in s]
                ax2.plot(order, strength, marker='o', label=f'mol_{i + 1}')
                max_order = max(max(order), max_order)
        else:
            tmp = [0] * 100
            for s in J_m[1]:
                order = [i[0] for i in s]
                max_order = max(max(order), max_order)
                for i, j in enumerate(s):
                    tmp[i] += j[1]
            order = [i for i in range(3, max_order + 1)]
            order_n = [x / max_order for x in order]
            strength = tmp[:len(order)]
            strength = [x / sum(strength) for x in strength]
            ax2.plot(order_n, strength, marker='o',  label='$J^{(m)}$')
            ax22.plot(order, strength)

            tmp = [0] * 100
            for s in J_d[1]:
                order = [i[0] for i in s]
                max_order = max(max(order), max_order)
                for i, j in enumerate(s):
                    tmp[i] += j[1]
            order = [i for i in range(3, max_order + 1)]
            order_n = [x / max_order for x in order]
            strength = tmp[:len(order)]
            strength = [x / sum(strength) for x in strength]
            ax2.plot(order_n, strength, marker='o',  label='$J^{(m)}_D$')
    else:
        order = [x[0] for x in J_m[1]]
        order_n = [x / max_order for x in order]
        strength = [x[1] for x in J_m[1]]
        ax2.plot(order_n, strength, marker='o', label='$J^{(m)}$')
        ax22.plot(order, strength)

        order_n = [x[0] / max_order for x in J_d[1]]
        strength = [x[1] for x in J_d[1]]
        ax2.plot(order_n, strength, marker='o', label='$J^{(m)}_D$')
    ax2.set_xticks(list([0.2 * i for i in range(1, 6)]))
    if data not in ['QM7', 'QM8', 'md']:
        ax22.set_xticks(list(range(m_start, max_order + 1, 2)))
    else:
        ax22.set_xticks(list(range(m_start, max_order + 1, 3)))
    ax22.set_xlabel('Order', fontsize=fontsize)
    ax2.legend()
    ax2.grid()

    fig.tight_layout()
    if save:
        plt.savefig(f'../../../strength_{data}.pdf', bbox_inches='tight')
    else:
        plt.show()


def grad_order_plt(data='qm7', n_iter=300, alpha=1e-3):
    grad_all = []
    for m in ['egnn', 'molformer']:
        grad_all.append(torch.load(f'../save/grad_{data}_{m}.pt')[0])

    print(len(grad_all[0][0]))
    print(len(grad_all[0]))
    print(len(grad_all))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    fontsize = 16
    ax1.tick_params(labelsize=fontsize)
    ax2.tick_params(labelsize=fontsize)
    plt.rcParams.update({'font.size': fontsize})

    ax1.set_title(f'EGNN on {data.upper()}')
    feats = grad_all[0].cpu().numpy()
    tsne = TSNE(n_components=2, verbose=1, n_iter=n_iter)
    feats_ = tsne.fit_transform(feats)
    df = pd.DataFrame(dict(x=feats_[:, 0], y=feats_[:, 1], label=[i + 3 for i in range(len(feats))]))
    groups = df.groupby('label')

    for name, group in groups:
        ax1.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=name)
        ax1.text(group.x, group.y, s=name)

    ax2.set_title(f'Molformer on {data.upper()}')
    feats = grad_all[1].cpu().numpy()
    tsne = TSNE(n_components=2, verbose=1, n_iter=n_iter)
    feats_ = tsne.fit_transform(feats)
    df = pd.DataFrame(dict(x=feats_[:, 0], y=feats_[:, 1], label=[i + 3 for i in range(len(feats))]))
    groups = df.groupby('label')

    for name, group in groups:
        ax2.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=name)
        ax2.text(group.x, group.y, s=name)
    ax2.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)

    plt.savefig(f'../../grad_tsne_{data}.png', bbox_inches='tight')
    print(f'Figure save at grad_tsne_{data}.png!')

    from scipy import stats
    for i, j in enumerate(['egnn', 'molformer']):
        feats = grad_all[i].cpu().numpy()
        k2, p = stats.normaltest(feats, axis=None)
        print("p = {:g}".format(p))
        if p < alpha:
            print(f"The null hypothesis: |Gradients of {j} on {data} come from a normal distribution.| can be rejected")
        else:
            print("The null hypothesis cannot be rejected")


def k_training(fontsize=16, save=True):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.tick_params(labelsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    plt.rcParams.update({'font.size': fontsize})
    for data in ['newtonian', 'hamiltonian', 'qm7', 'qm8']:
        m_start = 2
        if data not in ['newtonian', 'md']: m_start += 1
        J_ = torch.load(f'../save/J_{data}_egnn.pt')
        # 取前100 epochs
        J_['epoch'] = [x for x in J_['epoch'] if x <= 100]
        m_n = J_['k'][:len(J_['epoch'])]
        if data in ['qm7', 'qm8']:
            m_n = [i / 17 for i in m_n]
            # 手动调整
            if data == 'qm8':
                for i in range(2, len(m_n)):
                    m_n[i] += 0.3
        else:
            m_n = [i / 10 for i in m_n]
        ax.plot(J_['epoch'], m_n, marker='o', label=data)
    ax.legend()
    ax.grid()
    ax.set_ylabel('m* / n', fontsize=fontsize)
    ax.set_xlabel('Epoch', fontsize=fontsize)
    ax.set_ylim([0, 1.05])
    fig.tight_layout()
    if save:
        plt.savefig(f'../../../../k_training.pdf', bbox_inches='tight')
    else:
        plt.show()


if __name__ == '__main__':
    for x in ['qm7', 'qm8', 'newtonian', 'hamiltonian', 'md']:
        strength_order_plot(data=x)
    # strength_order_training(data='qm7')
    # k_training()
