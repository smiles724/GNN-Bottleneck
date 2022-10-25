import os
from PIL import Image
import torchvision
from torch.utils.data import Dataset

from util.io_handler import SampleIoHandler


class CIFAR10_selected(Dataset):

    def __init__(self, args, root: str, transform: torchvision.transforms.Compose, train: bool) -> None:
        super().__init__()
        self.root = root
        self.transform = transform
        # list of (class_name, img index(in the WHOLE dataset, not in a specific class, 0-based), class_index)
        self.selected_imgs = SampleIoHandler(args).load()
        self.dataset = torchvision.datasets.CIFAR10(root=root, train=train, transform=transform, download=True)
        self.class_name_list = self.dataset.classes # list of class names

    def __getitem__(self, index):
        img_index_in_whole_dataset = self.selected_imgs[index][1]
        image, label = self.dataset[img_index_in_whole_dataset] # call the __getitem__ method of dataset, label is int
        assert label == self.selected_imgs[index][2]
        assert self.class_name_list[label] == self.selected_imgs[index][0]
        name = self.class_name_list[label] + "_%05d" % img_index_in_whole_dataset # e.g. airplane_00029
        return name, image, label

    def __len__(self):
        return len(self.selected_imgs)


class ImageNet_selected(Dataset):

    def __init__(self, root, list_file, transform, splitor=" ", **kwargs):
        super().__init__()
        self.list_file = list_file
        self.transform = transform
        with open(list_file, 'r') as f:
            lines = f.readlines()
        assert splitor in [" ", ",", ";"]
        self.has_labels = len(lines[0].split(splitor)) == 2
        if self.has_labels:
            self.fns, self.labels = zip(*[l.strip().split(splitor) for l in lines])
            self.labels = [int(l) for l in self.labels]
        else:
            self.labels = None
            self.fns = [l.strip() for l in lines]
        self.fns = [os.path.join(root, 'train', fn) for fn in self.fns]

    def __getitem__(self, index):
        name = self.fns[index].split("/")[-1].split(".")[0]
        image = Image.open(self.fns[index])
        image = image.convert('RGB')
        # image = np.array(image, dtype=np.uint8)
        image = self.transform(image)

        if self.has_labels:
            label = self.labels[index]
        return name, image, label

    def __len__(self):
        return len(self.fns)


def get_dataset_util(args, transform: torchvision.transforms.Compose, train: bool):
    """ get dataset
    Input:
        args:
        transform: torchvision.transforms.Compose, transform for the image; tabular data do not need transform
        train: bool, only valid when dataset is NOT ImageNet.
            If train=False, use the validation set. If train=True, use the training set.
            By default we will use the training set. When evaluating on ImageNet, we have to use the validation set.
    Return:
        some dataset: Dataset,
    """
    if args.dataset == "cifar10":
        root = os.path.join(args.prefix, args.datasets_dirname, 'cifar10')
        return CIFAR10_selected(args, root, transform, train)
    elif args.dataset == "imagenet":
        root = os.path.join(args.prefix, args.datasets_dirname, 'imagenet')
        return ImageNet_selected(root, args.samples_file, transform, splitor=" ")
    else:
        raise Exception(f"dataset [{args.dataset}] not implemented. Error in get_dataset_util")
