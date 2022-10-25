import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.utils.data.dataloader import DataLoader
import argparse
import os
import time
import math
import matplotlib
matplotlib.use('agg')

from util import (prepare, seed_torch, normalize,
                  InteractionLogitIoHandler, PairIoHandler, PlayerIoHandler, set_args)

MAX_BS = 4 * 400


def compute_order_interaction_img(args,
                                  model: torch.nn.Module,
                                  image: torch.Tensor, image_name: str,
                                  pairs: np.ndarray, ratio: float,
                                  player_io_handler: PlayerIoHandler,
                                  logit_io_handler: InteractionLogitIoHandler):
    """
    Input:
        args: Dict of args.
        model: Model (nn.Module) to be evaluated.
        image: Input image tensor of [1, C, H, W].
        image_name: str, image_name of this sample
        pairs: (pairs_num, 2) array, (i,j) pairs
        ratio: float, ratio of the order of the interaction, order=(n-2)*ratio
        player_io_handler: IO handler for loading previous pairs.
        logit_io_handler: IO handler for loading & saving model outputs.
    """
    time0 = time.time()
    model.to(args.device)
    order = int((args.grid_size ** 2 - 2) * ratio)
    print("order m=%d" % order)

    with torch.no_grad():
        model.eval()
        image_shape = image.size()
        channels = image.size(1)
        players = player_io_handler.load(round(ratio * 100), image_name)
        ori_logits = []

        forward_mask = []
        for index, pair in enumerate(pairs):
            print('\r\t\tPairs: \033[1;31m\033[5m%03d\033[0m/%03d' % (index + 1, len(pairs)), end='')
            point1, point2 = pair[0], pair[1]

            players_curr_pair = players[index] # context S for this pair of (i,j)
            mask = torch.zeros((
                4 * args.samples_number_of_s, channels, args.grid_size ** 2), device=args.device)

            if order != 0: # if order == 0, then S=emptyset, we don't need to set S
                S_cardinality = players_curr_pair.shape[1]  # |S|
                assert S_cardinality == order
                idx_multiple_of_4 = 4 * np.arange(args.samples_number_of_s)  # indices: 0, 4, 8...
                stack_idx = np.stack([idx_multiple_of_4] * S_cardinality, axis=1)  # stack the indices to match the shape of player_curr_i
                mask[stack_idx, :, players_curr_pair] = 1  # set S for v(S U {i}) and v(S)
                mask[stack_idx+1, :, players_curr_pair] = 1  # set S for v(S U {i}) and v(S)
                mask[stack_idx+2, :, players_curr_pair] = 1  # set S for v(S U {i}) and v(S)
                mask[stack_idx+3, :, players_curr_pair] = 1  # set S for v(S U {i}) and v(S)

            mask[4 * np.arange(args.samples_number_of_s) + 1, :, point1] = 1  # S U {i}
            mask[4 * np.arange(args.samples_number_of_s) + 2, :, point2] = 1  # S U {j}
            mask[4 * np.arange(args.samples_number_of_s), :, point1] = 1  # S U {i,j}
            mask[4 * np.arange(args.samples_number_of_s), :, point2] = 1  # S U {i,j}
            
            # upsampling mask to input image
            mask = mask.view(4 * args.samples_number_of_s, channels, args.grid_size, args.grid_size)
            mask = F.interpolate(mask.clone(), size=[image_shape[2], image_shape[3]], mode='nearest').float()

            # if sample number of S is too large (especially for vgg19), we need to split one batch into several iterations
            if len(mask) > MAX_BS:
                iterations = math.ceil(len(mask) / MAX_BS)
                for it in range(iterations): # in each iteration, we compute output for MAX_BS images
                    batch_mask = mask[it * MAX_BS : min((it+1) * MAX_BS, len(mask))]
                    expand_image = image.expand(len(batch_mask), channels, image_shape[2], image_shape[3]).clone()
                    masked_image = batch_mask * expand_image

                    output_ori = model(masked_image)
                    assert not torch.isnan(output_ori).any(), 'there are some nan numbers in the model output'
                    ori_logits.append(output_ori.detach())

            else: # if sample number of S is small, we can concatenate several batches and do a single inference
                forward_mask.append(mask)
                if (len(forward_mask) < args.cal_batch // args.samples_number_of_s) and (index < args.pairs_number - 1):
                    continue
                else:
                    forward_batch = len(forward_mask) * args.samples_number_of_s
                    batch_mask = torch.cat(forward_mask, dim=0)
                    expand_image = image.expand(4 * forward_batch, channels, image_shape[2], image_shape[3]).clone()
                    masked_image = batch_mask * expand_image

                    output_ori = model(masked_image)
                    assert not torch.isnan(output_ori).any(), 'there are some nan numbers in the model output'

                    ori_logits.append(output_ori.detach())
                    forward_mask = []
        print("Finish time: {:.4f}" % (time.time() - time0))

        all_logits = torch.cat(ori_logits, dim=0)  # (pairs_num*4*samples_number_of_s, class_num)
        print(f"all_logits shape: {all_logits.shape}")
        logit_io_handler.save(round(ratio * 100), image_name, all_logits)


def compute_interactions(args,
                         model: nn.Module, dataloader: DataLoader,
                         pair_io_handler: PairIoHandler,
                         player_io_handler: PlayerIoHandler,
                         logit_io_handler: InteractionLogitIoHandler):
    model.to(args.device)

    with torch.no_grad():
        model.eval()
        total_pairs = pair_io_handler.load()
        for index, (name, image, label) in enumerate(dataloader):
            print('Images: \033[1;31m\033[5m%03d\033[0m/%03d' % (index + 1, len(dataloader)))

            image = image.to(args.device)
            label = label.to(args.device)

            image = normalize(args, image)

            pairs = total_pairs[index]

            for ratio in args.ratios:
                print('\tCurrent ratio: \033[1;31m\033[5m%.2f' % ratio)
                order = int((args.grid_size ** 2 - 2) * ratio)
                seed_torch(1000 * index + order + args.seed)
                compute_order_interaction_img(args, model=model, image=image, image_name=name[0],
                                              pairs=pairs, ratio=ratio,
                                              player_io_handler=player_io_handler,
                                              logit_io_handler=logit_io_handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dirname', default="result", type=str)
    parser.add_argument('--inter_type', default="pixel", type=str, choices=["pixel"])
    parser.add_argument('--arch', default="vit", type=str,
                        choices=[
                            "resnet", "vit", "mlpmixer", "swin", "convnext", "convmixer", "poolformer",
                        ])
    parser.add_argument('--pretrained', default=False, type=bool, help="whether to use pretrained model.")
    parser.add_argument('--checkpoint_name', default=None, type=str, help="ckeckpoint name")
    parser.add_argument('--checkpoint_path', default=None, type=str, help="path to ckeckpoints")
    parser.add_argument('--dataset', default="cifar10", type=str, choices=['cifar10', 'imagenet'])
    parser.add_argument('--class_number', default=None, type=int, help="class number")
    parser.add_argument('--image_size', default=None, type=int, help="Input size of image")
    parser.add_argument("--cal_batch", default=100, type=int, help='calculate # of images per batch')
    parser.add_argument('--gpu_id', default=1, type=int, help="GPU ID")
    parser.add_argument('--chosen_class', default='random', type=str,choices=['random'])
    parser.add_argument('--seed', default=0, type=int, help="random seed")
    parser.add_argument('--grid_size', default=16, type=int,
                        help="partition the input image to grid_size * grid_size patches"
                             "each patch is considered as a player")

    args = parser.parse_args()

    set_args(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seed_torch(args.seed)

    pair_io_handler = PairIoHandler(args)
    player_io_handler = PlayerIoHandler(args)
    interaction_logit_io_handler = InteractionLogitIoHandler(args)
    
    model, dataloader = prepare(args, train=True)
    compute_interactions(args, model, dataloader, pair_io_handler, player_io_handler, interaction_logit_io_handler)
