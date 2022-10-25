import os
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import socket

from util.dataset_util import get_dataset_util
from util.model_util import get_model

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2471, 0.2435, 0.2616]
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def clamp(x: int, min: int, max: int) -> int:
    if x < min:
        return min
    elif x > max:
        return max
    else:
        return x


def seed_torch(seed, deterministic=False) -> None:
    """ set random seed for all related packages
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def prepare(args, train:bool) -> Tuple[nn.Module, DataLoader]:
    """ prepare models and dataloader for the computation of multi-order interaction
    Input:
        args: args
        train: bool,
            If train=False, use the validation set. If train=True, use the training set.
            By default we will use the training set (except for ImageNet)
    Return:
        model: nn.Module, model to be evaluated
        dataloader: DataLoader, dataloader of the training/validation set
    """
    if args.dataset == "cifar10":
        transform = transforms.Compose([transforms.ToTensor()])
    elif args.dataset == "imagenet":
        transform = transforms.Compose([
            transforms.Resize(int(args.image_size / 0.875)),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor()
        ])
    else:
        raise Exception(f"Dataset [{args.dataset}] not implemented. Error in prepareing dataloader")

    dataset = get_dataset_util(args, transform, train)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = get_model(args)
    return model, dataloader


def normalize(args, x: torch.Tensor):
    """ normalize the image before feeding it into the model
    Input:
        args: args
        x : (N,3,H,W) tensor, original image
    Return:
        (x - mean) / std (tensor): (N,3,H,W) tensor, normalized image
    """
    if args.dataset == "cifar10":
        mean_list, std_list = CIFAR10_MEAN, CIFAR10_STD
    elif args.dataset == "imagenet":
        mean_list, std_list = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    else:
        raise Exception("Dataset not implemented")
    mean = torch.tensor(mean_list).view(3, 1, 1).expand(
        x.shape[0], 3, x.shape[2], x.shape[2]).to(x.device)
    std = torch.tensor(std_list).view(3, 1, 1).expand(
        x.shape[0], 3, x.shape[2], x.shape[2]).to(x.device)
    return (x - mean) / std


class LogWriter():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


def log_args_and_backup_code(args, file_path):
    file_name = os.path.basename(file_path)
    logfile = LogWriter(os.path.join(args.output_dir, f"args_{file_name.split('.')[0]}.txt"))
    for k, v in args.__dict__.items():
        logfile.cprint(f"{k} : {v}")
    logfile.cprint("Numpy: {}".format(np.__version__))
    logfile.cprint("Pytorch: {}".format(torch.__version__))
    logfile.cprint("torchvision: {}".format(torchvision.__version__))
    logfile.cprint("Cuda: {}".format(torch.version.cuda))
    logfile.cprint("hostname: {}".format(socket.gethostname()))
    logfile.close()

    os.system(f'cp {file_path} {args.output_dir}/{file_name}.backup')


def get_reward(args, logits, label):
    """ given logits, calculate reward score for interaction computation
    Input:
        args: args.softmax_type determines which type of score to compute the interaction
            - normal: use log p, p is the probability of the {label} class
            - modified: use log p/(1-p), p is the probability of the {label} class
            - yi: use logits the {label} class
        logits: (N,num_class) tensor, a batch of logits before the softmax layer
        label: (1,) tensor, ground truth label
    Return:
        v: (N,) tensor, reward score
    """
    if args.softmax_type == "normal": # log p
        v = F.log_softmax(logits, dim=1)[:, label[0]]
    elif args.softmax_type == "modified": # log p/(1-p)
        v = logits[:, label[0]] - torch.logsumexp(
            logits[:, np.arange(args.class_number) != label[0].item()],dim=1)
    elif args.softmax_type == "yi": # logits
        v = logits[:, label[0]]
    else:
        raise Exception(f"softmax type [{args.softmax_type}] not implemented")
    return v


def denormalize_img(args, img):
    if args.dataset == "cifar10":
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    elif args.dataset == "imagenet":
        mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    else:
        raise Exception(f"dataset [{args.dataset}] is not supported")
    img_denormalized = torch.zeros_like(img)
    for channel_id in range(3):
        img_denormalized[:, channel_id, :, :] = img[:, channel_id, :, :] * std[channel_id]
        img_denormalized[:, channel_id, :, :] = img_denormalized[:, channel_id, :, :] + mean[channel_id]
    return img_denormalized
