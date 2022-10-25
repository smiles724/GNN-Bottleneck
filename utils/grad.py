import os
import time
import torch
import torch.optim as opt
from torch.utils.data import DataLoader, TensorDataset
from einops import rearrange
from itertools import combinations

from model.egnn import EGNN_Network
from model.molformer import Molformer
from utils.utils import parse_args, set_seed, Logger


def main(min_particle=10, max_particle=13):
    args = parse_args()
    set_seed(args.seed)
    assert args.data not in ['newtonian', 'md'], 'Only support graph-level tasks to calculate gradients!'
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
    if args.method == 'egnn':
        model = EGNN_Network(num_tokens=args.tokens, dim=args.dim, depth=args.depth, num_nearest_neighbors=args.num_nearest,
                             norm_coors=True, coor_weights_clamp_value=2., aggregate=args.aggregate).cuda()
    else:
        model = Molformer(data=args.data, vocab=args.tokens, dist_bar=[0.8, 1.6, 3], depth=args.depth,
                          embed_dim=args.dim, aggregate=args.aggregate).cuda()
    optimizer = opt.Adam(model.parameters(), lr=args.lr)
    log = Logger(args.save_path, f'grad_{args.data}_{args.method}.log')

    if args.pretrain:
        try:
            checkpoint = torch.load(f'save/model_weight/model_{args.data}_{args.method}.pt')
            model.load_state_dict(checkpoint['model'])
            log.logger.info(f'Model loading successfully from save/model_{args.data}_{args.method}.pt.')
        except:
            log.logger.info('No pretraining model found! Please train a model first. \nRandomly initialize a model.')
    else:
        log.logger.info('Not loading any pretraining model! \nRandomly initialize a model.')

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
        error.append(torch.abs(pred[:, 0] - y))
    error = torch.cat(error)
    # hamiltonian的原子数目较少，取十倍的sample数
    if args.data in ['hamiltonian', 'newtonian']: args.n_sample *= 10
    error_k, topk = [i[:args.n_sample] for i in error.sort()]
    log.logger.info(f'Picking up {args.n_sample} samples with error: {error_k.cpu()}.')
    if args.data in ['hamiltonian', 'newtonian']:
        # hamiltonian体系下的原子数目是恒定不变的，所以批量计算
        train_loader = DataLoader(TensorDataset(data_x[topk], data_pos[topk], data_y[topk]), batch_size=args.n_sample, shuffle=False)
    else:
        train_loader = DataLoader(TensorDataset(data_x[topk], data_pos[topk], data_y[topk]), batch_size=1, shuffle=False)

    grad_all = []
    model.eval()
    for x, pos, y in train_loader:
        t0 = time.time()
        x, pos, y = x.cuda(), pos.cuda(), y.cuda()
        if args.data not in ['hamiltonian', 'newtonian']:
            x = x.long()
            mask = (x != 0)
            x, pos = x[mask].unsqueeze(0), pos[mask].unsqueeze(0)

        b, n = x.shape[:2]
        if args.data not in ['hamiltonian', 'newtonian']:
            log.logger.info(f'** This molecule has {n} atoms. **')
        else:
            log.logger.info(f'** This system has {n} particles. **')
        index = [i for i in range(n)]

        # 遍历context包含的可能的变量数
        for m in range(3, n + 1):
            grad_m = torch.zeros_like(model.out.out[-1].weight[0]).cuda()
            contexts = list(combinations(index, m))

            grad_m = graph_level_str(grad_m, model, x, pos, contexts, optimizer)

            if args.data in ['hamiltonian', 'newtonian']:
                # I_m维度(d_grad)
                grad_all.append(grad_m)
            else:
                grad_all.append(grad_m)
    # J_all维度(n-2, d_grad)
    if args.data not in ['hamiltonian', 'newtonian']:
        grad_all = torch.mean(torch.stack(grad_all, dim=0), dim=0)
    else:
        grad_all = torch.stack(grad_all, dim=0)
    log.logger.info(f'The gradients: {grad_all}\n (Time: {(time.time() - t0) / 3600:.2f}h)')
    torch.save([grad_all, error_k], f'save/grad_{args.data}_{args.method}.pt')


def graph_level_str(grad_tmp, model, x, pos, contexts, optimizer):
    # 先考虑固定变量数目下的context，再考虑context下的pair
    for context in contexts:
        x_S_ij, pos_S_ij = x[:, context], pos[:, context]
        out_S_ij = model(x_S_ij, pos_S_ij)[1][:, 0]
        optimizer.zero_grad()
        out_S_ij.mean().backward()
        grad = model.out.out[-1].weight.grad[0]
        pred_S_ij = grad.clone()

        pred_S_i = dict()
        for i in context:
            x_S_i, pos_S_i = x[:, [p for p in context if p != i]], pos[:, [p for p in context if p != i]]
            out_S_i_batch = model(x_S_i, pos_S_i)[1][:, 0]
            optimizer.zero_grad()
            out_S_i_batch.mean().backward()
            # 注意molformer最后输出层的weight应该为model.generator.proj[-1].weight.grad[0]
            grad = model.out.out[-1].weight.grad[0]
            pred_S_i[i] = grad.clone()

        pairs = list(combinations(context, 2))
        for pair in pairs:
            x_S, pos_S = x[:, [i for i in context if i not in pair]], pos[:, [i for i in context if i not in pair]]
            out_S_batch = model(x_S, pos_S)[1]
            optimizer.zero_grad()
            out_S_batch.mean().backward()
            grad = model.out.out[-1].weight.grad[0]
            pred_S = grad.clone()
            grad_tmp += pred_S_ij + pred_S - pred_S_i[pair[0]] - pred_S_i[pair[1]]

    return grad_tmp


if __name__ == '__main__':
    main()
