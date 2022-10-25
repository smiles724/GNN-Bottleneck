import os
import timm
import torch
import torch.nn as nn

from mmcv.runner import load_state_dict


def load_checkpoint(args, checkpoint_path: str, model: nn.Module) -> None:
    """
    Input
        args: args
        checkpoint_path: str, path of saved model parameters
        model: nn.Module
        optimizer: torch.optim.Optimizer
    Return:
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f'File doesn\'t exists {checkpoint_path}')
    print(f'=> loading checkpoint "{checkpoint_path}"')
    checkpoint = torch.load(checkpoint_path, map_location=args.device)

    # load models saved with legacy versions
    if not ('state_dict' in checkpoint):
        checkpoint = checkpoint
    else:
        checkpoint = checkpoint['state_dict']
    if not ('model' in checkpoint):
        checkpoint = checkpoint
    else:
        checkpoint = checkpoint['model']

    # load with models trained on a single gpu or multiple gpus
    if 'module.' in list(checkpoint.keys())[0]:
        checkpoint = {k[len('module.'):]: v for k, v in checkpoint.items()}

    load_state_dict(model, checkpoint, strict=False)


def get_model(args) -> nn.Module:
    """ get model and load parameters if needed
    Input:
        args: args
            if args.checkpoint_path is "None", then do not load model parameters
    Return:
        some model: nn.Module, model to be evaluated
    """
    if args.checkpoint_path is None:
        load_model = False
        use_pretrained = True
    else:
        load_model = True
        use_pretrained = args.pretrained

    arch_dict = dict(
        resnet="resnet50",
        vit="deit_small_patch16_224",
        swin="swin_tiny_patch4_window7_224",
        mlpmixer="mixer_s16_224",
        convnext="convnext_tiny" if not use_pretrained else "convnext_tiny_hnf",
        convmixer="convmixer_768_32",
        poolformer="poolformer_s24",
    )
    assert args.arch in arch_dict.keys()
    model = timm.create_model(arch_dict[args.arch], pretrained=use_pretrained)
    if not use_pretrained or args.class_number != 1000:
        model.reset_classifier(num_classes=args.class_number)
    
    if load_model:
        load_checkpoint(args, args.checkpoint_path, model)
    else:
        print("use model at initialization")
    
    return model
