import torch
import json
from torchvision import transforms
import torchvision.datasets as D
from torchvision.datasets import ImageFolder


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_json(json_path):
    with open(json_path) as data_file:
        data = json.load(data_file)
    return data


def download_torchvision(
    dataset: str,
    dataset_root,
    transform: transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    ),
):
    if dataset == "lfw":
        return D.LFWPeople(
            dataset_root, transform=transform, target_transform=None, download=True
        )
    elif dataset == "fmnist":
        return D.FashionMNIST(dataset_root, transform=transform, target_transform=None, download=True)
    else:
        assert False, f"{dataset} can't be downloaded with torchvision"


def load_ImageSet(
    dataset_root: str,
    transform: transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    ),
):
    return ImageFolder(dataset_root, transform=transform)
