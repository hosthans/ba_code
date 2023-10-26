import os, torchvision, PIL
import torch
from PIL import Image
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.utils.data.sampler import SubsetRandomSampler


def init_dataloader(model_config, dataset, iterator: bool = False):
    if iterator:
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=model_config["batch_size"],
            shuffle=True,
            drop_last=True,
            num_workers=0,
            pin_memory=True
        ).__iter__()
    else:
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=model_config["batch_size"],
            shuffle=True,
            drop_last=True,
            num_workers=0,
            pin_memory=True
        )

    print("DataLoader initialized")
    return data_loader


class ImageFolder(data.Dataset):
    def __init__(self, config, file_path, mode):
        self.config = config
        self.mode = mode
        if mode == "gan":
            self.img_path = config["img_gan_path"]
        else:
            self.img_path = config["img_path"]
        self.model_name = config["model_name"]
        # self.img_list = os.listdir(self.img_path)
        self.processor = self.get_processor()
        self.name_list, self.label_list = self.get_list(file_path)
        self.image_list = self.load_img()
        self.num_img = len(self.image_list)
        self.n_classes = config["num_classes"]
        if self.mode is not "gan":
            print("Load " + str(self.num_img) + " images")

    def get_list(self, file_path):
        name_list, label_list = [], []
        f = open(file_path, "r")
        for line in f.readlines():
            if self.mode == "gan":
                img_name = line.strip()
            else:
                img_name, iden = line.strip().split(" ")
                label_list.append(int(iden))
            name_list.append(img_name)

        return name_list, label_list

    def load_img(self):
        img_list = []
        for i, img_name in enumerate(self.name_list):
            if (
                img_name.endswith(".png")
                or img_name.endswith(".jpg")
                or img_name.endswith(".jpeg")
            ):
                path = self.img_path + "/" + img_name
                img = PIL.Image.open(path)
                img = img.convert("RGB")
                img_list.append(img)
        return img_list

    def get_processor(self):
        if self.model_name in ("FaceNet", "FaceNet_all"):
            re_size = 112
        else:
            re_size = 64
        if self.config["name"] == "celeba":
            crop_size = 108
            offset_height = (218 - crop_size) // 2
            offset_width = (178 - crop_size) // 2
        elif self.config["name"] == "facescrub":
            # NOTE: dataset face scrub
            if self.mode == "gan":
                crop_size = 54
                offset_height = (64 - crop_size) // 2
                offset_width = (64 - crop_size) // 2
            else:
                crop_size = 108
                offset_height = (218 - crop_size) // 2
                offset_width = (178 - crop_size) // 2
        elif self.config["name"] == "ffhq":
            # print('ffhq')
            # NOTE: dataset ffhq
            if self.mode == "gan":
                crop_size = 88
                offset_height = (128 - crop_size) // 2
                offset_width = (128 - crop_size) // 2
            else:
                crop_size = 108
                offset_height = (218 - crop_size) // 2
                offset_width = (178 - crop_size) // 2

        # #NOTE: dataset pf83
        # crop_size = 176
        # offset_height = (256 - crop_size) // 2
        # offset_width = (256 - crop_size) // 2
        crop = lambda x: x[
            :,
            offset_height : offset_height + crop_size,
            offset_width : offset_width + crop_size,
        ]

        proc = []
        if self.mode == "train":
            proc.append(transforms.ToTensor())
            proc.append(transforms.Lambda(crop))
            proc.append(transforms.ToPILImage())
            proc.append(transforms.Resize((re_size, re_size)))
            proc.append(transforms.RandomHorizontalFlip(p=0.5))
            proc.append(transforms.ToTensor())
        else:
            proc.append(transforms.ToTensor())
            if (
                self.mode == "test"
                or self.mode == "train"
                or self.config["name"] != "facescrub"
            ):
                proc.append(transforms.Lambda(crop))
            proc.append(transforms.ToPILImage())
            proc.append(transforms.Resize((re_size, re_size)))
            proc.append(transforms.ToTensor())

        return transforms.Compose(proc)

    def __getitem__(self, index):
        processer = self.get_processor()
        img = processer(self.image_list[index])
        if self.mode == "gan":
            return img
        label = self.label_list[index]

        return img, label

    def __len__(self):
        return self.num_img
