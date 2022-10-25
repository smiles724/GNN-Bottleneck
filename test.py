import os
import time
import torch
from torch.utils.data import DataLoader, TensorDataset
from einops import rearrange
from itertools import combinations

from model.egnn import EGNN_Network
from model.molformer import Molformer
from utils.utils import parse_args, set_seed


def main(batch_test=True, min_particle=10, max_particle=18):
    args = parse_args()
    set_seed(args.seed)
    # hamiltonian的原子数目较少，取十倍的sample数
    if args.data in ['hamiltonian', 'newtonian']: args.n_sample *= 10
    if args.data not in ['hamiltonian', 'newtonian']:
        data_x, data_pos, data_y = torch.load(f'data/{args.data}.pt')
        data_pos = data_pos.float()
        if args.data in ['qm7', 'qm8']:
            # 限制原子数目在max_particle以下，加快运行速度，但也不能小于min_particle，否则low-order的strength被高估
            n_mask = (torch.sum(data_x != 0, dim=-1) < max_particle) & (torch.sum(data_x != 0, dim=-1) >= min_particle)
            data_x, data_pos, data_y = data_x[n_mask], data_pos[n_mask], data_y[n_mask]
    else:
        data_x, data_y_node, data_y_graph = torch.load(f'data/hamiltonian.pt')
        data_x = rearrange(data_x, 'ns nt n d -> (ns nt) n d')
        if args.data == 'hamiltonian':
            data_y = rearrange(data_y_graph, 'ns nt -> (ns nt)')
        else:
            data_y = rearrange(data_y_node, 'ns nt n d-> (ns nt) n d')
        data_x, data_pos = data_x[..., 2:], data_x[..., :2]
    if args.data == 'qm9': data_y = data_y[:, args.qm9_index]
    test_loader = DataLoader(TensorDataset(data_x, data_pos, data_y), batch_size=args.bs * 2, shuffle=False)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.data in ['hamiltonian', 'newtonian']: args.tokens = None
    if args.data in ['newtonian', 'md']: args.aggregate = False
    if args.method == 'egnn':
        model = EGNN_Network(num_tokens=args.tokens, dim=args.dim, depth=args.depth, num_nearest_neighbors=args.num_nearest,
                             norm_coors=True, coor_weights_clamp_value=2., aggregate=args.aggregate).cuda()
    else:
        model = Molformer(data=args.data, vocab=args.tokens, dist_bar=[0.8, 1.6, 3], depth=args.depth,
                          embed_dim=args.dim, aggregate=args.aggregate).cuda()
    if args.no_pretrain:
        print('Not loading any pretraining model! Randomly initialize a model.')
    else:
        try:
            checkpoint = torch.load(f'save/model_weight/model_{args.data}_{args.method}.pt')
            model.load_state_dict(checkpoint['model'])
            print(f'Model loading successfully from save/model_weight/model_{args.data}_{args.method}.pt.')
        except:
            print('No pretraining model found! Please train a model first. \nRandomly initialize a model.')

    train_loader, error_k = select_sample(args, model, test_loader, data_x, data_pos, data_y, echo=True)

    t0 = time.time()
    J_all = str_compute(args, model, train_loader, batch_test)
    print(f'The strength: {J_all}\n (Time: {(time.time() - t0) / 3600:.2f}h)')
    if args.no_pretrain:
        torch.save([J_all, error_k], f'save/strength_{args.data}_{args.method}_n{args.n_sample}_random.pt')
    else:
        torch.save([J_all, error_k], f'save/strength_{args.data}_{args.method}_n{args.n_sample}.pt')


def select_sample(args, model, test_loader, data_x, data_pos, data_y, echo=False, min_particle=None, max_particle=None):
    # 选取error最小的n_sample个样本
    error = []
    for x, pos, y in test_loader:
        x, pos, y = x.cuda(), pos.cuda(), y.cuda()
        if args.data not in ['hamiltonian', 'newtonian']:
            x = x.long()
            mask = (x != 0)
        else:
            mask = (x[..., -1] != 0)
        with torch.no_grad():
            _, pred = model(x, pos, mask=mask)
        if args.data in ['newtonian', 'md']:
            # 采用L1-norm计算误差，即MAE
            error.append(torch.einsum('b n d -> b', torch.abs(pred - y)))
        else:
            error.append(torch.abs(pred[:, 0] - y))
    error = torch.cat(error)

    if min_particle:
        error, idx = error.sort()
        error_k, top_k = [], []
        i = 0
        while len(error_k) < args.n_sample:
            if args.data not in ['hamiltonian', 'newtonian']:
                n_particle = torch.sum(data_x[idx[i]] != 0).item()
            else:
                n_particle = torch.sum(data_x[idx[i], ..., -1] != 0).item()
            if max_particle > n_particle >= min_particle:
                error_k.append(error[i])
                top_k.append(idx[i])
            i += 1
        error_k = torch.stack(error_k)
        top_k = torch.stack(top_k)
    else:
        error_k, top_k = [i[:args.n_sample] for i in error.sort()]
    if echo:
        print(f'Picking up {args.n_sample} samples with error: {error_k.cpu()}.')
    if args.data in ['hamiltonian', 'newtonian', 'md']:
        # hamiltonian/newtonian体系和ISO17数据集下的原子数目是恒定不变的，所以批量计算
        train_loader = DataLoader(TensorDataset(data_x[top_k], data_pos[top_k], data_y[top_k]), batch_size=args.n_sample, shuffle=False)
    else:
        train_loader = DataLoader(TensorDataset(data_x[top_k], data_pos[top_k], data_y[top_k]), batch_size=1, shuffle=False)
    return train_loader, error_k


def str_compute(args, model, train_loader, batch_test=True, echo=True):
    J_all = []
    m_start = 2
    if args.data not in ['newtonian', 'md']: m_start += 1
    model.eval()
    for x, pos, y in train_loader:
        t1 = time.time()
        x, pos, y = x.cuda(), pos.cuda(), y.cuda()
        if args.data not in ['hamiltonian', 'newtonian', 'md']:
            # 维度(1, n)
            x = x.long()
            mask = (x != 0)
            x, pos = x[mask].unsqueeze(0), pos[mask].unsqueeze(0)

        b, n = x.shape[:2]
        if echo and args.data not in ['hamiltonian', 'newtonian']:
            print(f'** This molecule has {n} atoms. **')
        elif echo:
            print(f'** This system has {n} particles. **')
        index = [i for i in range(n)]
        J_m = []

        # 遍历context包含的可能的变量数
        for m in range(m_start, n + 1):
            I_m = torch.zeros(b).cuda()
            contexts = list(combinations(index, m))

            if args.data not in ['newtonian', 'md']:
                I_m = graph_level_str(I_m, model, x, pos, contexts, args, batch_test)
            else:
                # 输出维度为(b=n_sample, n, d)
                I_m = node_level_str(I_m, model, x, pos, contexts, batch_test)

            if args.data in ['hamiltonian', 'newtonian', 'md']:
                # I_m维度(b)
                J_m.append(torch.abs(I_m))
            else:
                J_m.append(abs(I_m.item()))
        if args.data not in ['hamiltonian', 'newtonian', 'md']:
            J_m = [[m_start + i, j / sum(J_m)] for i, j in enumerate(J_m)]
            if echo: print(f'The strength: {J_m}\n (Time: {(time.time() - t1) / 3600:.2f}h)')
            # J_all组成n_sample个list，每个list是不同order的strength，数据格式[[[3, J_3], ..., ], [[3, J_3], ..., ], ...]
            J_all.append(J_m)
    if args.data in ['hamiltonian', 'newtonian', 'md']:
        # J_all维度(n_sample, n-2/n-1)
        J_all = torch.stack(J_m, dim=-1)
        J_all = torch.mean(J_all / torch.sum(J_all, dim=-1, keepdim=True), dim=0)
        J_all = [[i + m_start, j.item()] for i, j in enumerate(J_all)]
    return J_all


def graph_level_str(I_m, model, x, pos, contexts, args, batch_test=True):
    # 先考虑固定变量数目下的context，再考虑context下的pair
    for context in contexts:
        x_S_ij, pos_S_ij = x[:, context], pos[:, context]
        with torch.no_grad():
            pred_S_ij = model(x_S_ij, pos_S_ij)[1][:, 0]

        # batch输入，加快计算速度
        if batch_test:
            pred_S_i = dict()
            x_S_i_batch, pos_S_i_batch = [], []
            for i in context:
                x_S_i_batch.append(x[:, [p for p in context if p != i]])
                pos_S_i_batch.append(pos[:, [p for p in context if p != i]])
            x_S_i_batch, pos_S_i_batch = torch.cat(x_S_i_batch, dim=0), torch.cat(pos_S_i_batch, dim=0)
            with torch.no_grad():
                # 输出维度为(len(context) * n_sample)
                pred_S_i_batch = model(x_S_i_batch, pos_S_i_batch)[1][:, 0]
            for i in range(len(context)):
                if args.data != 'hamiltonian':
                    pred_S_i[context[i]] = pred_S_i_batch[i]
                else:
                    pred_S_i[context[i]] = pred_S_i_batch[i * args.n_sample: (i + 1) * args.n_sample]
        else:
            pred_S_i = dict()
            for i in context:
                x_S_i, pos_S_i = x[:, [p for p in context if p != i]], pos[:, [p for p in context if p != i]]
                with torch.no_grad():
                    pred_S_i[i] = model(x_S_i, pos_S_i)[1]

        I_m_pair = torch.zeros_like(I_m)
        pairs = list(combinations(context, 2))
        if batch_test:
            x_S_batch, pos_S_batch = [], []
            for pair in pairs:
                x_S_batch.append(x[:, [i for i in context if i not in pair]])
                pos_S_batch.append(pos[:, [i for i in context if i not in pair]])
            x_S_batch, pos_S_batch = torch.cat(x_S_batch, dim=0), torch.cat(pos_S_batch, dim=0)
            with torch.no_grad():
                pred_S_batch = model(x_S_batch, pos_S_batch)[1][:, 0]
            for pair_id in range(len(pairs)):
                if args.data != 'hamiltonian':
                    I_m_pair += pred_S_ij + pred_S_batch[pair_id] - pred_S_i[pairs[pair_id][0]] - pred_S_i[pairs[pair_id][1]]
                else:
                    I_m_pair += pred_S_ij + pred_S_batch[pair_id * args.n_sample: (pair_id + 1) * args.n_sample] - pred_S_i[
                        pairs[pair_id][0]] - pred_S_i[pairs[pair_id][1]]
        else:
            for pair in pairs:
                x_S, pos_S = x[:, [i for i in context if i not in pair]], pos[:, [i for i in context if i not in pair]]
                with torch.no_grad():
                    pred_S = model(x_S, pos_S)[1]
                I_m_pair += pred_S_ij + pred_S - pred_S_i[pair[0]] - pred_S_i[pair[1]]

        # E_ij[E_S]
        I_m += I_m_pair / len(pairs) / len(contexts)
    return I_m


def node_level_str(I_m, model, x, pos, contexts, batch_test=True):
    for context in contexts:
        x_S_ij, pos_S_ij = x[:, context], pos[:, context]
        with torch.no_grad():
            # 输出维度为(b=n_sample, len(context), d)
            pred_S_ij = model(x_S_ij, pos_S_ij)[1]

        I_m_i = torch.zeros_like(I_m)
        for i in range(len(context)):
            I_m_j = torch.zeros_like(I_m)
            # 把第i维放到最后，以便后续slice
            x_S_ij_tmp = torch.cat((x_S_ij[:, [p for p in range(len(context)) if p != i]], x_S_ij[:, i].unsqueeze(1)), dim=1)
            pos_S_ij_tmp = torch.cat((pos_S_ij[:, [p for p in range(len(context)) if p != i]], pos_S_ij[:, i].unsqueeze(1)), dim=1)

            if batch_test:
                x_S_j_batch, pos_S_j_batch = [], []
                for j in range(len(context) - 1):
                    x_S_j_batch.append(x_S_ij_tmp[:, [p for p in range(len(context)) if p != j]])
                    pos_S_j_batch.append(pos_S_ij_tmp[:, [p for p in range(len(context)) if p != j]])
                x_S_j_batch, pos_S_j_batch = torch.cat(x_S_j_batch, dim=0), torch.cat(pos_S_j_batch, dim=0)
                with torch.no_grad():
                    # 输出维度为(b * (len(context) - 1), len(context) - 1, d)
                    pred_S_j_batch = model(x_S_j_batch, pos_S_j_batch)[1]
                for j in range(len(context) - 1):
                    I_m_j += torch.sum(torch.abs(pred_S_ij[:, -1] - pred_S_j_batch[j * x.shape[0]: (j + 1) * x.shape[0], -1]), dim=-1)
            else:
                for j in range(len(context)):
                    if j != i:
                        x_S_j_batch = x_S_ij_tmp[:, [p for p in range(len(context)) if p != j]]
                        pos_S_j_batch = pos_S_ij_tmp[:, [p for p in range(len(context)) if p != j]]
                        with torch.no_grad():
                            # 输出维度为(b=n_sample, n, d)
                            pred_S_j_batch = model(x_S_j_batch, pos_S_j_batch)[1]
                        I_m_j += torch.sum(torch.abs(pred_S_ij[:, i] - pred_S_j_batch[:, i]), dim=-1)
            # E_j
            I_m_i += I_m_j / (len(context) - 1)

        # E_S[E_i[...]]，等价于E_i[E_S[...]]
        I_m += I_m_i / len(context) / len(contexts)
    return I_m


def node_level_str_mask(I_m, model, x, pos, contexts, batch_test=True):
    """ 用mask的方法写的node-level strength """
    for context in contexts:
        mask_S = torch.zeros(x.shape[:2]).cuda()
        mask_S[:, context] = 1
        with torch.no_grad():
            # 输出维度为(b=n_sample, n, d)
            pred_S_ij = model(x, pos, mask=mask_S.bool())[1]

        I_m_i = torch.zeros_like(I_m)
        for i in context:
            I_m_j = torch.zeros_like(I_m)
            if batch_test:
                x_batch, pos_batch, mask_S_j_batch = [], [], []
                for j in [x for x in context if x != i]:
                    mask_S_j = torch.zeros(x.shape[:2]).cuda()
                    mask_S_j[:, [p for p in context if p != j]] = 1
                    mask_S_j_batch.append(mask_S_j)
                    x_batch.append(x)
                    pos_batch.append(pos)
                x_batch, pos_batch, mask_S_j_batch = torch.cat(x_batch), torch.cat(pos_batch), torch.cat(mask_S_j_batch)
                with torch.no_grad():
                    # 输出维度为(b * (len(context) - 1), n, d)
                    pred_S_j_batch = model(x_batch, pos_batch, mask=mask_S_j_batch.bool())[1]
                for k in range(len(context) - 1):
                    I_m_j += torch.sum(torch.abs(pred_S_ij[:, i] - pred_S_j_batch[k * x.shape[0]: (k + 1) * x.shape[0], i]), dim=-1)
            else:
                for j in [x for x in context if x != i]:
                    mask_S_j = torch.zeros(x.shape[:2]).cuda()
                    mask_S_j[:, [p for p in context if p != j]] = 1
                    with torch.no_grad():
                        # 输出维度为(b=n_sample, n, d)
                        pred_S_j_batch = model(x, pos, mask=mask_S_j.bool())[1]
                    I_m_j += torch.sum(torch.abs(pred_S_ij[:, i] - pred_S_j_batch[:, i]), dim=-1)
            I_m_i += I_m_j / (len(context) - 1)
        I_m += I_m_i / len(context) / len(contexts)

    return I_m


if __name__ == '__main__':
    main()
