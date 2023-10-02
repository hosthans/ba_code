import torch
import torch.utils.data as data
import os
import utils.helper as utils
from torchvision import transforms
import time
from model.classifier import *
import random
from copy import deepcopy

CONFIG_PATH = "./config"
DATA_PATH = "./data"
MODEL_DATA_PATH = "./data/model_data"


class Trainer:
    """Trainer class:
    Running trainig-loop for adjusting params of neural networks.
    """

    def __init__(self, model_type: str = "VGG16", dataset: str = "ffhq"):
        """Initialize neceserray values

        Args:
            model_type (str, optional): Insert the Model type implemented in models.json and implemented in python-code. Defaults to "VGG16".
            dataset (str, optional): Insert the Dataset-Name which should be used for training purposes. Defaults to None.
        """

        self.MODEL_PATH = os.path.join(MODEL_DATA_PATH, model_type)
        self.CKPT_PATH = os.path.join(self.MODEL_PATH, "ckpts")

        os.makedirs(DATA_PATH, exist_ok=True)
        os.makedirs(self.MODEL_PATH, exist_ok=True)
        os.makedirs(self.CKPT_PATH, exist_ok=True)

        self.models_config = utils.load_json(os.path.join(CONFIG_PATH, "models.json"))
        # config of actual model
        try:
            self.model_config = self.models_config[model_type]
        except NotImplementedError as e:
            print(e)
    
        self.data_config = utils.load_json(os.path.join(CONFIG_PATH, "data.json"))
        self.dataset_config = self.data_config['fmnist']
        self.dataset = self.get_dataset(dataset)
        self.trainloader, self.testloader = self.get_dataloader(self.dataset, 0.2)
        self.model = torch.nn.DataParallel(self.load_model(model_type)).to(utils.get_device())
        self.optimizer = self.load_optimizer()
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.train(model_type)

    def train(self, model_type):
        best_ACC = 0.0

        for epoch in range(self.model_config['epochs']):
            tf = time.time()
            ACC, cnt, loss_tot = 0,0,0.0

            self.model.train()

            for i, (img, iden) in enumerate(self.trainloader):
                img, iden = img.to(utils.get_device()), iden.to(utils.get_device())
                bs = img.size(0)
                iden = iden.view(-1)

                feats, out_prob = self.model(img)
                cross_loss = self.criterion(out_prob, iden)
                loss = cross_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                out_iden = torch.argmax(out_prob, dim=1).view(-1)
                ACC += torch.sum(iden == out_iden).item()
                loss_tot += loss.item() * bs
                cnt += bs
            
            train_loss, train_acc = loss_tot * 1.0 / cnt, ACC * 100.0 / cnt
            test_acc = self.test()

            interval = time.time() - tf
            if test_acc > best_ACC:
                best_ACC = test_acc
                best_model = deepcopy(self.model)

            if (epoch+1) % 10 == 0:
                torch.save({'state_dict':self.model.state_dict()}, os.path.join(self.CKPT_PATH, "ckpt_epoch{}.tar").format(epoch))

            print("Epoch:{}\tTime:{:.2f}\tTrain Loss:{:.2f}\tTrain Acc:{:.2f}\tTest Acc{:.2f}".format(epoch, interval, train_loss, train_acc, test_acc))

        torch.save({'state_dict':self.model.state_dict()}, os.path.join(self.MODEL_PATH, "full{}.tar").format(model_type))

    def test(self):
        self.model.eval()
        loss, cnt, ACC = 0.0, 0, 0

        for img, iden in self.testloader:
            img, iden = img.to(utils.get_device()), iden.to(utils.get_device())
            bs = img.size(0)
            iden = iden.view(-1)

            out_prob = self.model(img)[-1]
            out_iden = torch.argmax(out_prob, dim=1).view(-1)
            ACC += torch.sum(iden == out_iden).item()
            cnt += bs

        return ACC * 100.0 / cnt

    def get_dataset(self, dataset: str):
        dataset_config = self.data_config[dataset]
        if os.path.exists(os.path.join(DATA_PATH, dataset)):
            print("load dataset from directory")
            if dataset_config["torch"] == True:
                # each dataset downloaded from torch existing in ./data
                return utils.download_torchvision(
                    dataset=dataset,
                    dataset_root=os.path.join(DATA_PATH, dataset),
                    transform=transforms.Compose(
                        [
                            transforms.Grayscale(num_output_channels=3),
                            transforms.ToTensor(),
                            transforms.Resize((64, 64)),
                            transforms.Normalize(
                                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                            ),
                        ]
                    ),
                )
            else:
                # each dataset downloaded as directory/zip existing in ./data
                return utils.load_ImageSet(
                    os.path.join(DATA_PATH, dataset),
                    transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                            ),
                        ]
                    ),
                )

        else:
            if dataset_config["torch"] == True:
                print(" -------- download dataset from torchvision -------- ")
                return utils.download_torchvision(
                    dataset=dataset,
                    dataset_root=os.path.join(DATA_PATH, dataset),
                    transform=transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                            ),
                        ]
                    ),
                )
            else:
                assert dataset_config[
                    "torch"
                ], f" -------- Please download the dataset manually to data directory (position: {dataset}) and run the script again -------- "

        print(dataset_config)

    def get_dataloader(self, dataset: data.Dataset, test_split_percentage: float):
        # Berechne die Anzahl der Datensätze im Testset basierend auf dem Prozentsatz
        num_total_samples = len(dataset)
        num_test_samples = int(num_total_samples * test_split_percentage)
        num_train_samples = num_total_samples - num_test_samples

        # Teile den Datensatz in Trainings- und Testdaten auf
        indices = list(range(num_total_samples))
        random.shuffle(indices)
        train_indices = indices[:num_train_samples]
        test_indices = indices[num_train_samples:]

        train_sampler = data.SubsetRandomSampler(train_indices)
        test_sampler = data.SubsetRandomSampler(test_indices)

        # Erstelle Trainings- und Test-Dataloader
        train_loader = data.DataLoader(
            dataset=dataset,
            batch_size=self.model_config["batch_size"],
            sampler=train_sampler,
            num_workers=0,
            pin_memory=True,
        )

        test_loader = data.DataLoader(
            dataset=dataset,
            batch_size=self.model_config["batch_size"],
            sampler=test_sampler,
            num_workers=0,
            pin_memory=True,
        )

        return train_loader, test_loader
    
    def load_model(self, model_type: str):
        if model_type == 'VGG16':
            return VGG16(n_classes=self.dataset_config['num_classes'])
        return None
    
    def load_optimizer(self):
        return torch.optim.SGD(
            params=self.model.parameters(),
            lr=self.model_config['lr'],
            momentum=self.model_config['momentum'],
            weight_decay=self.model_config['weight_decay']
        )
