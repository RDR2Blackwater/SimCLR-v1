import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms


# Prepare contrastive dataset
class get_dataset_view(object):
    """ Generate a dataset with given transform policy, returning two datasets in a list object """
    """ The DataLoader would return batch like ([Tensor(B,C,H,W), Tensor(B,C,H,W)], label) """
    def __init__(self, transform_policy, contrastive_channel=2):
        self.tfm = transform_policy
        self.contrastive = contrastive_channel

    def __call__(self, x):
        return [self.tfm(x) for _ in range(self.contrastive)]


class contrastive_dataset:
    def __init__(self, path: str = './default_datasets'):
        self.root = path

    @staticmethod
    def get_transform_method(size, s=1):
        # Return a transform method following the SimCLR paper
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        transform_policy = transforms.Compose([
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            rnd_color_jitter,
            transforms.RandomGrayscale(p=0.2),
            # the kernel size is set to be 10% of the image height/width
            transforms.GaussianBlur(kernel_size=(int((0.1*size)//2)*2 + 1, 1), sigma=(0.1, 2.0)),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True)
        ])
        return transform_policy

    def get_dataset(self, name: str = "stl10", contrastive_channel: int = 2):
        """
        :param name: define the dataset would used to pretrain
        :param contrastive_channel: define the channels in contrastive learning, default 2
        :return: dataset
        """
        target_dataset = {
            'cifar10': datasets.CIFAR10(root=self.root,
                                        train=True,
                                        transform=get_dataset_view(self.get_transform_method(size=32), contrastive_channel),
                                        download=True),
            'stl10': datasets.STL10(root=self.root,
                                    split="unlabeled",
                                    transform=get_dataset_view(self.get_transform_method(size=96), contrastive_channel),
                                    download=True)
        }
        try:
            data_fn = target_dataset[name]
        except KeyError:
            raise KeyError("Unknown dataset, please select 'cifar10' or 'stl10'")
        return data_fn
