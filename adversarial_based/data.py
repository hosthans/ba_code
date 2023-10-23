from __future__ import print_function, division
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

class FaceScrub(Dataset):
    def __init__(self, root, transform=None, target_transform=None, train=True):
        self.dataset = ImageFolder(root, transform=ToTensor())
        self.data, self.labels = zip(*self.dataset.samples)

        if train:
            self.data = self.data[:int(0.8 * len(self.data))]
            self.labels = self.labels[:int(0.8 * len(self.labels))]
        else:
            self.data = self.data[int(0.8 * len(self.data)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = Image.open(self.data[index]).convert('RGB'), self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class CelebA(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        data = []
        for i in range(10):
            data.append(np.load(os.path.join(self.root, 'celebA_64_{}.npy').format(i + 1)))
        data = np.concatenate(data, axis=0)

        v_min = data.min(axis=0)
        v_max = data.max(axis=0)
        data = (data - v_min) / (v_max - v_min)
        labels = np.array([0] * len(data))

        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target