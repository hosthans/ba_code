import json
import torch
from torchvision import transforms
import random
import numpy as np
import os

def load_json(json_path):
    with open(json_path) as data_file:
        data = json.load(data_file)
    return data

def load_my_state_dict(self, state_dict):
    own_state = self.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            print(name)
            continue
        own_state[name].copy_(param.data)

def get_deprocessor():
    proc = []
    proc.append(transforms.Resize((112, 112)))
    proc.append(transforms.ToTensor())
    return transforms.Compose(proc)


def low2high(img):
    bs = img.size(0)
    proc = get_deprocessor()
    img_tensor = img.detach().cpu().float()
    img = torch.zeros(bs, 3, 112, 112)
    for i in range(bs):
        img_i = transforms.ToPILImage()(img_tensor[i, :, :, :]).convert('RGB')
        img_i = proc(img_i)
        img[i, :, :, :] = img_i[:, :, :]
    
    img = img.cuda()
    return img

def seed_everything(seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)  # type: ignore
        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = True  # type: ignore