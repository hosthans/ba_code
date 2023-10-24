import torch
import json
from torchvision import transforms
import torchvision.datasets as D
from torchvision.datasets import ImageFolder
from torch.autograd import grad
import torchvision.utils as tvls
import os

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_json(json_path):
    with open(json_path) as data_file:
        data = json.load(data_file)
    return data


def load_local(
    dataset: str,
    file_name: str
):
    datapath = os.path.join("data", dataset)
    dataset_path = os.path.join(datapath, file_name)
    datas = torch.load(dataset_path)
    return datas
    


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
        return D.FashionMNIST(
            dataset_root, transform=transform, target_transform=None, download=True
        )
    elif dataset == "mnist":
        return D.MNIST(
            dataset_root, transform=transform, target_transform=None, download=True
        )
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


def freeze(net):
    for p in net.parameters():
        p.requires_grad_(False)


def unfreeze(net):
    for p in net.parameters():
        p.requires_grad_(True)


def gradient_penalty(x, y, DG):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.randn(shape).cuda()
    z = x + alpha * (y - x)
    z = z.cuda()
    z.requires_grad = True

    o = DG(z)
    g = grad(o, z, grad_outputs=torch.ones(o.size()).cuda(), create_graph=True)[0].view(
        z.size(0), -1
    )
    gp = ((g.norm(p=2, dim=1) - 1) ** 2).mean()

    return gp


def save_tensor_image(images, filename, nrow=None, normalize=True):
    if not nrow:
        tvls.save_image(images, filename, normalize=normalize, padding=0)
    else:
        tvls.save_image(images, filename, normalize=normalize, nrow=nrow, padding=0)
