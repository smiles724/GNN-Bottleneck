import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.utils.data.dataloader import DataLoader
import os
import argparse

from util import (InteractionIoHandler, InteractionLogitIoHandler, set_args,
                  prepare, seed_torch, get_reward)


def compute_order_interaction_img(args,
                                  image_name: str, label: torch.Tensor, ratio: float,
                                  logit_io_handler: InteractionLogitIoHandler,
                                  save_io_handler: InteractionIoHandler):
    """
    Input:
        args: Dict of args.
        image_name: str, image_name of this sample
        label: (1,) tensor, label of this sample
        ratio: float, ratio of the order of the interaction, order=(n-2)*ratio
        logit_io_handler: IO handler for loading & saving model outputs.
        save_io_handler: IO handler for loading & saving outputs.
    """
    interactions = []

    logits = logit_io_handler.load(round(ratio * 100), image_name)
    logits = logits.reshape((
        args.pairs_number, args.samples_number_of_s * 4, args.class_number)) # load saved logits

    for index in range(args.pairs_number):
        print('\r\t\tPairs: \033[1;31m\033[5m%03d\033[0m/%03d' % (index + 1, args.pairs_number), end='')
        output_ori = logits[index, :, :]

        v = get_reward(args, output_ori, label)  # (4*samples_number_of_s,)

        # Delta v(i,j,S) = v(S∪{i,j}) - v(S∪{i}) - v(S∪{j}) + v(S)
        score_ori = v[4 * np.arange(args.samples_number_of_s)] + \
                    v[4 * np.arange(args.samples_number_of_s) + 3] - \
                    v[4 * np.arange(args.samples_number_of_s) + 1] - \
                    v[4 * np.arange(args.samples_number_of_s) + 2]
        interactions.extend(score_ori.tolist())
    
    interactions = np.array(interactions).reshape(-1, args.samples_number_of_s) # (pair_num, sample_num)
    assert interactions.shape[0] == args.pairs_number

    save_io_handler.save(round(ratio * 100), image_name, interactions)  # (pair_num, sample_num)


def compute_interactions(args,
                         model: nn.Module, dataloader: DataLoader,
                         logit_io_handler: InteractionLogitIoHandler,
                         save_io_handler: InteractionIoHandler):
    model.to(args.device)

    with torch.no_grad():
        model.eval()
        for index, (name, image, label) in enumerate(dataloader):
            print('Images: \033[1;31m\033[5m%03d\033[0m/%03d' % (index + 1, len(dataloader)))
            image = image.to(args.device)
            label = label.to(args.device)

            for ratio in args.ratios:
                print('\tCurrent ratio: \033[1;31m\033[5m%.2f' % ratio)
                order = int((args.grid_size ** 2 - 2) * ratio)
                seed_torch(1000 * index + order + args.seed)
                if args.out_type == 'gt':
                    compute_order_interaction_img(
                        args, name[0], label, ratio, logit_io_handler, save_io_handler)
                else:
                    raise Exception(f"output type [{args.out_type}] not supported.")


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
    parser.add_argument("--dataset", default="cifar10", type=str, choices=['cifar10', 'imagenet'])
    parser.add_argument('--class_number', default=None, type=int, help="class number")
    parser.add_argument('--image_size', default=None, type=int, help="Input size of image")
    parser.add_argument('--gpu_id', default=1, type=int, help="GPU ID")
    parser.add_argument('--softmax_type',
                        default='modified', type=str,
                        choices=['normal','modified','yi'], help="reward function for interaction")
    parser.add_argument('--out_type',
                        default='gt', type=str,
                        choices=['gt'], help="we use the score of the ground truth class to compute interaction")
    parser.add_argument('--chosen_class', default='random', type=str, choices=['random'])
    parser.add_argument('--seed', default=0, type=int, help="random seed")
    parser.add_argument('--grid_size', default=16, type=int,
                        help="partition the input image to grid_size * grid_size patches"
                             "each patch is considered as a player")
    parser.add_argument('--no_cuda', action="store_true")


    args = parser.parse_args()

    set_args(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    args.device = torch.device(
        "cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    seed_torch(args.seed)

    logit_io_handler = InteractionLogitIoHandler(args)
    save_io_handler = InteractionIoHandler(args)

    model, dataloader = prepare(args,train=True)
    compute_interactions(args, model, dataloader, logit_io_handler, save_io_handler)
