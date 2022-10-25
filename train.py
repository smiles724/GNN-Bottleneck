import os
from time import time

import torch
import torch.optim as opt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import l1_loss
from einops import rearrange
from model.egnn import EGNN_Network
from model.molformer import Molformer
from utils.utils import parse_args, set_seed
from test import str_compute, select_sample


def main():
    args = parse_args()
    set_seed(args.seed)
    args.bs = 512 * len(args.gpu.split(','))
    args.lr = 1e-4 * len(args.gpu.split(','))

    if args.data not in ['hamiltonian', 'newtonian']:
        data_x, data_pos, data_y = torch.load(f'data/{args.data}.pt')
    else:
        data_x, data_y_node, data_y_graph = torch.load(f'data/hamiltonian.pt')
        data_x = rearrange(data_x, 'ns nt n d -> (ns nt) n d')
        if args.data == 'hamiltonian':
            data_y = rearrange(data_y_graph, 'ns nt -> (ns nt)')
        else:
            data_y = rearrange(data_y_node, 'ns nt n d-> (ns nt) n d')

        # egnn可以只输入one-hot，也可以输入多维特征
        data_x, data_pos = data_x[..., 2:], data_x[..., :2]
        args.n_sample *= 10

    if args.data == 'qm9': data_y = data_y[:, args.qm9_index]
    x_train, x_test, pos_train, pos_test, y_train, y_test = train_test_split(data_x, data_pos, data_y,
                                                                             test_size=0.2, random_state=args.seed)
    train_size = len(x_train)
    test_size = len(x_test)
    train_loader = DataLoader(TensorDataset(x_train, pos_train, y_train), batch_size=args.bs, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, pos_test, y_test), batch_size=args.bs * 2, shuffle=False)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print(f'{"=" * 40} Training {"=" * 40}\nData: {args.data}; Model: {args.method}; KNN: {args.num_nearest};'
          f' Train: {train_size}; Test: {test_size}\nGPU: {args.gpu}; Epoch: {args.ep}; Batch_size: {args.bs};'
          f' Save: {args.save}\nMin_particle: {args.min_p}; Max_particle: {args.max_p}\n{"=" * 40} Start Training {"=" * 40}')
    if args.data == 'qm9':
        targets = ['mu', 'alpha', 'homo', 'lumo', 'gap']
        print(f'Target: {targets[args.qm9_index]}')

    if args.data in ['hamiltonian', 'newtonian']: args.tokens = None
    if args.data in ['newtonian', 'md']: args.aggregate = False
    if args.method == 'egnn':
        model = EGNN_Network(num_tokens=args.tokens, dim=args.dim, depth=args.depth, num_nearest_neighbors=args.num_nearest,
                             norm_coors=True, coor_weights_clamp_value=2., aggregate=args.aggregate).cuda()
    else:
        # 模型size很小时，也可以取得较低的error，但和大模型还是有差距，且训练时间较长
        model = Molformer(data=args.data, vocab=args.tokens, dist_bar=[0.8, 1.6, 3], depth=args.depth,
                          embed_dim=args.dim, aggregate=args.aggregate, knn=args.num_nearest).cuda()
    if len(args.gpu) > 1:  model = torch.nn.DataParallel(model)
    criterion = torch.nn.L1Loss()

    optimizer = opt.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.6, patience=10, min_lr=5e-6)
    best_metric, t0, early_stop = 1e8, time(), 0
    J_ = {'epoch': [], 'strength': [], 'k': []}

    for epoch in range(0, args.ep):
        if args.rewire and epoch % args.interval == 0:
            print(f'Current K: {model.num_nearest_neighbors}')
            if args.data in ['qm7', 'qm8']:
                new_loader, _ = select_sample(args, model, train_loader, x_train, pos_train, y_train,
                                              min_particle=args.min_p, max_particle=args.max_p)
            else:
                new_loader, _ = select_sample(args, model, train_loader, x_train, pos_train, y_train)
            J = str_compute(args, model, new_loader, echo=False)
            if args.data in ['qm7', 'qm8']:
                J_tmp = [[i + 3, 0] for i in range(max([len(x) for x in J]))]
                for i in range(1, len(J)):
                    for j in range(len(J[i])):
                        J_tmp[j][1] += J[i][j][1]
                J = J_tmp
            J_['strength'].append(J)
            J_['k'].append(model.num_nearest_neighbors)
            J_['epoch'].append(epoch)

            # 暂时只考虑EGNN
            if epoch > 0 and args.method == 'egnn':
                # 限定qm7/qm8的原子数目相同，否则strength的增加有偏误，数据格式[[3, J_3], [4, J_4], ..., ]
                J_delta = [[J[i][0], J[i][1] - J_['strength'][-2][i][1]] for i in
                           range(min(len(J), len(J_['strength'][-2])))]
                print('The difference of J^{(m)}:', J_delta)
                max_d = max([i[1] for i in J_delta])
                if max_d > args.j_bar:
                    best_k = J_delta[[i[1] for i in J_delta].index(max_d)][0]

                    # 默认已找到最优K
                    if len(J_['k']) > 3:
                        if J_['k'][-1] == J_['k'][-2] and J_['k'][-2] == J_['k'][-3] and epoch > 100:
                            args.rewire = False
                            print('Finding the best K, and the rewiring stops.')

                    # 暂时只考虑EGNN
                    print(f'Changing K from *{model.num_nearest_neighbors}* to *{best_k}*')
                    model.num_nearest_neighbors = best_k
                else:
                    print(f'{max_d} not reaching the threshold {args.j_bar}.')

        model.train()
        loss, t1, step = 0.0, time(), 0
        for src_x, src_pos, y in train_loader:
            src_x, src_pos, y = src_x.cuda(), src_pos.float().cuda(), y.cuda()
            if args.data not in ['hamiltonian', 'newtonian']:
                src_x = src_x.long()
                src_mask = (src_x != 0)
            else:
                src_mask = (src_x[..., -1] != 0)
            feat, pred = model(src_x, src_pos, mask=src_mask)

            if args.data not in ['newtonian', 'md']:
                loss_batch = criterion(pred[:, 0], y)
            else:
                loss_batch = criterion(pred, y)
            loss += loss_batch.item() / train_size * len(src_x)
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()

        model.eval()
        val_metric = 0
        for tgt_x, tgt_pos, y in test_loader:
            tgt_x, tgt_pos, y = tgt_x.cuda(), tgt_pos.float().cuda(), y.cuda()
            if args.data not in ['hamiltonian', 'newtonian']:
                tgt_x = tgt_x.long()
                tgt_mask = (tgt_x != 0)
            else:
                tgt_mask = (tgt_x[..., -1] != 0)
            with torch.no_grad():
                feat, pred = model(tgt_x, tgt_pos, mask=tgt_mask)
            if args.data not in ['newtonian', 'md']:
                val_metric += l1_loss(pred[..., 0], y).item() / test_size * len(tgt_x)
            else:
                val_metric += l1_loss(pred, y).item() / test_size * len(tgt_x)
        print('Epoch: {} | Time: {:.1f}s | Loss: {:.3f} | Val_Metric: {:.3f} | Lr: {:.3f}'
              .format(epoch + 1, time() - t1, loss, val_metric, optimizer.param_groups[0]['lr'] * 1e5))
        lr_scheduler.step(val_metric)
        if val_metric < best_metric:
            best_ep = epoch
            best_metric = val_metric
            best_model = model
            early_stop = 0
        else:
            early_stop += 1

        # 为了效果最好，测试集可见
        if early_stop >= 30:
            print('Early Stopping!!! No Improvement on metric for 30 Epochs.')
            break
    print(f'{"=" * 20} End Training (Time: {(time() - t0) / 3600:.2f}h) {"=" * 20}\n'
          f'Best Metric for {args.method} in {args.data}: {best_metric} (epoch: {best_ep})')
    if len(J_['k']) == 0:
        checkpoint = {'model': best_model.state_dict(), 'error': best_metric}
        if len(args.gpu) > 1: checkpoint['model'] = best_model.module.state_dict()
        if not os.path.exists('save/model_weight/'): os.makedirs('save/model_weight/')
        torch.save(checkpoint, f'save/model_weight/model_{args.data}_{args.method}.pt')
    else:
        if not os.path.exists('save/J_/'): os.makedirs('save/J_/')
        torch.save(J_, f'save/J_/J_{args.data}_{args.method}.pt')


if __name__ == '__main__':
    main()


