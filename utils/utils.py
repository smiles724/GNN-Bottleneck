import torch
import os
import logging
import argparse
import random
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    # set data and method
    parser.add_argument('--data', type=str, default='qm7', choices=['qm7', 'qm8', 'qm9', 'hamiltonian', 'newtonian', 'md'])
    parser.add_argument('--method', type=str, default='egnn', choices=['egnn', 'molformer'], help='The model.')
    parser.add_argument('-i', '--qm9_index', type=int, default=0, choices=[0, 1, 2, 3, 4], help='The index of selected property in QM9')
    parser.add_argument('--no_pretrain', default=False, action='store_true', help='Where to load the pretrained model weights.')
    parser.add_argument('--n_sample', type=int, default=10)
    parser.add_argument('--rewire', default=False, action='store_true')
    parser.add_argument('--j_bar', type=float, default=0.7)
    parser.add_argument('--min_p', type=int, default=10)
    parser.add_argument('--max_p', type=int, default=15)
    parser.add_argument('--interval', type=int, default=10)

    # set the hyperparameters of backbone
    parser.add_argument('--tokens', type=int, default=100, help='The default number of atom classes.')
    parser.add_argument('--num_nearest', type=int, default=8, help='The default number of nearest neighbors.')
    parser.add_argument('--depth', type=int, default=3, help='Number of stacked layers.')
    parser.add_argument('--dim', type=int, default=32, help='Dimension of features.')
    parser.add_argument('--head', type=int, default=4, help='Number of heads in multi-head attention, embed_dim % head == 0..')

    # note that you can not pass a lits directly to argsparser
    parser.add_argument('--dist_bar', default=None, help='Multi-scale distance thresholds.')
    parser.add_argument('--tgt', type=int, default=1, help='Number of prediction targets.')
    parser.add_argument('--aggregate', type=bool, default=True, help='Graph-level or node-level task.')

    # set training details
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--ep', type=int, default=1200, help='Number of epoch.')
    parser.add_argument('--bs', type=int, default=512, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate.')

    # set training environment
    parser.add_argument('--gpu', type=str, default='0', help='Index for GPU')
    parser.add_argument('--save', type=bool, default=True, help='Whether to save the model.')
    parser.add_argument('--model_path', default='', help='Path to load the model for visualization.')
    parser.add_argument('--save_path', default='save/', help='Path to save the model and the logger.')

    return parser.parse_args()


def set_seed(s):
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)
    random.seed(s)

    # keep the cudnn stable，https://learnopencv.com/ensuring-training-reproducibility-in-pytorch/
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Logger(object):
    level_relations = {'debug': logging.DEBUG, 'info': logging.INFO, 'warning': logging.WARNING,
                       'error': logging.ERROR, 'crit': logging.CRITICAL}  # mapping to different-level

    def __init__(self, path, filename, level='info'):
        if not os.path.exists(path):
            os.makedirs(path)
        self.logger = logging.getLogger(path + filename)
        self.logger.setLevel(self.level_relations.get(level))

        sh = logging.StreamHandler()
        self.logger.addHandler(sh)

        # 'w' to overwrite the loggger
        th = logging.FileHandler(path + filename, encoding='utf-8', mode='w')
        self.logger.addHandler(th)


if __name__ == '__main__':
    print()
