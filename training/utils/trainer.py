import torch
import torch.utils.data as data
import os
import utils.helper as utils
import utils.dataloader as dl
from torchvision import transforms
from torchvision import datasets
import time
from model.classifier import *
from model.gan import *
from model.facenet import *
import random
from copy import deepcopy
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.utils.batch_memory_manager import BatchMemoryManager
import numpy as np

CONFIG_PATH = "./config"
DATA_PATH = "./data"
MODEL_DATA_PATH = "./data/model_data"
RESULT_PATH_GAN = "./results"

MAX_GRAD_NORM = 1.2
EPSILON = 50.0
DELTA = 1e-5


class Trainer:
    """Trainer class:
    Running trainig-loop for adjusting params of neural networks.
    """

    def __init__(self, dataset: str = "ffhq", mode: str = "nn"):
        """Initialize neceserray values

        Args:
            model_type (str, optional): Insert the Model type implemented in models.json and implemented in python-code. Defaults to "VGG16".
            dataset (str, optional): Insert the Dataset-Name which should be used for training purposes. Defaults to None.
            mode (str, optional): Insert the training mode (gan or nn) for special training purposes
        """
        # load dataset configuration
        self.data_config = utils.load_json(os.path.join(CONFIG_PATH, "data.json"))
        self.dataset_config = self.data_config[dataset]

        # load model_type for specified dataset
        if mode == "gan":
            model_type = "gan"
        else:
            model_type = self.dataset_config["model_name"]

        self.MODEL_PATH = os.path.join(MODEL_DATA_PATH, model_type)
        self.CKPT_PATH = os.path.join(self.MODEL_PATH, "ckpts")
        self.RESULT_PATH = os.path.join(DATA_PATH, RESULT_PATH_GAN)

        os.makedirs(DATA_PATH, exist_ok=True)
        os.makedirs(self.MODEL_PATH, exist_ok=True)
        os.makedirs(self.CKPT_PATH, exist_ok=True)
        os.makedirs(self.RESULT_PATH, exist_ok=True)

        self.models_config = utils.load_json(os.path.join(CONFIG_PATH, "models.json"))
        # config of actual model
        try:
            self.model_config = self.models_config[model_type]
        except NotImplementedError as e:
            print(e)

        self.mode = mode

        print(self.dataset_config)
        if mode == "nn":
            # dataset
            print("----------------Loading datasets-----------------")
            # self.train_set = dl.ImageFolder(
            #     config=self.dataset_config,
            #     file_path=self.dataset_config["train_file"],
            #     mode="train",
            # )
            # self.test_set = dl.ImageFolder(
            #     config=self.dataset_config,
            #     file_path=self.dataset_config["test_file"],
            #     mode="test",
            # )

            self.train_set = datasets.MNIST(
                root="./data/mnist",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.Grayscale(num_output_channels=3),
                        transforms.ToTensor(),
                        transforms.Resize((64, 64)),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    ]
                ),
            )

            self.test_set = datasets.MNIST(
                root="./data/mnist",
                train=False,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.Grayscale(num_output_channels=3),
                        transforms.ToTensor(),
                        transforms.Resize((64, 64)),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    ]
                ),
            )

            print("---------------Loading dataloader----------------")
            self.trainloader = dl.init_dataloader(self.model_config, self.train_set)
            self.testloader = dl.init_dataloader(self.model_config, self.test_set)

            del self.train_set
            del self.test_set

            # print(self.trainloader)
            # print(self.testloader)

            # neural Network
            self.model = torch.nn.DataParallel(self.load_model(model_type)).to(
                utils.get_device()
            )
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            # print(self.model)

            # print(self.model)

            # Optimizer
            self.optimizer = self.load_optimizer()
            self.criterion = nn.CrossEntropyLoss().cuda()

            # trigger training
            # self.train()
        elif mode == "gan":
            # load dataset
            print("----------------Loading datasets-----------------")
            # self.train_set = dl.ImageFolder(
            #     config=self.dataset_config,
            #     file_path=self.dataset_config["gan_file"],
            #     mode="gan",
            # )
            self.train_set = datasets.MNIST(
                root="./data/mnist",
                download=True,
                transform=transforms.Compose(
                    [
                        # transforms.Grayscale(num_output_channels=3),
                        transforms.Resize((64, 64)),
                        transforms.ToTensor(),
                        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                        transforms.Normalize(mean=[0.5], std=[0.5])
                    ]
                ),
            )

            self.trainloader = data.DataLoader(self.train_set, batch_size=self.model_config["batch_size"], shuffle=True)

            # self.trainloader = dl.init_dataloader(self.model_config, self.train_set)

            # del self.train_set

            # Generator variables
            self.generator, self.discriminator = self.load_model(model_type=model_type)
            self.generator = torch.nn.DataParallel(self.generator).cuda()
            self.discriminator = torch.nn.DataParallel(self.discriminator).cuda()

            # Optimizer
            self.optim_gen = torch.optim.Adam(
                self.generator.parameters(),
                lr=self.model_config["lr"],
                betas=(0.5, 0.999),
            )
            self.optim_dis = torch.optim.Adam(
                self.discriminator.parameters(),
                lr=self.model_config["lr"],
                betas=(0.5, 0.999),
            )

            # trigger training
            # self.train_gan()
        elif mode == "dpnn":
            # dataset
            print("----------------Loading datasets-----------------")
            # self.train_set = dl.ImageFolder(config=self.dataset_config, file_path=self.dataset_config['train_file'], mode="train")
            # self.test_set = dl.ImageFolder(config=self.dataset_config, file_path=self.dataset_config['test_file'], mode="test")

            #######################
            # only for test

            self.train_set = datasets.MNIST(
                root="./data/mnist",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.Grayscale(num_output_channels=3),
                        transforms.ToTensor(),
                        transforms.Resize((64, 64)),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    ]
                ),
            )

            self.test_set = datasets.MNIST(
                root="./data/mnist",
                train=False,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.Grayscale(num_output_channels=3),
                        transforms.ToTensor(),
                        transforms.Resize((64, 64)),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    ]
                ),
            )

            print("---------------Loading dataloader----------------")
            self.trainloader = dl.init_dataloader(self.model_config, self.train_set)
            self.testloader = dl.init_dataloader(self.model_config, self.test_set)

            del self.train_set
            del self.test_set

            # neural Network
            self.model = torch.nn.DataParallel(self.load_model(model_type)).to(
                utils.get_device()
            )

            # self.model = VGG16(1000)
            # path_T = "data/model_data/VGG16/dp_VGG16.tar"
            # self.model = torch.nn.DataParallel(self.model).cuda()
            # self.model.load_state_dict(torch.load(path_T)['state_dict'], strict=False)

            # Optimizer
            self.optimizer = self.load_optimizer()
            self.criterion = nn.CrossEntropyLoss().cuda()

        else:
            assert False, "training method not implemented yet"

    def train(self):
        if self.mode == "nn":
            self.train_nn()
        elif self.mode == "gan":
            self.train_gan()
        elif self.mode == "dpnn":
            self.train_dp()
        else:
            print(f"train mode for {self.mode} not implemented yet")

    def train_dp(self):
        print(f"Training-Process started! -> {self.model_config['epochs']} epochs")

        self.privacy_engine = PrivacyEngine()

        (
            self.model,
            self.optimizer,
            self.trainloader,
        ) = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.trainloader,
            epochs=self.model_config["epochs"],
            target_epsilon=EPSILON,
            target_delta=DELTA,
            max_grad_norm=MAX_GRAD_NORM,
        )

        best_ACC = 0.0

        for epoch in range(self.model_config["epochs"]):
            tf = time.time()
            self.model.train()
            losses = []
            with BatchMemoryManager(
                data_loader=self.trainloader,
                max_physical_batch_size=128,
                optimizer=self.optimizer,
            ) as memory_safe_data_loader:
                for _batch_idx, (data, target) in enumerate(memory_safe_data_loader):
                    data, target = data.to(utils.get_device()), target.to(
                        utils.get_device()
                    )
                    self.optimizer.zero_grad()
                    feat, output = self.model(data)
                    loss = self.criterion(output, target)
                    loss.backward()
                    self.optimizer.step()
                    losses.append(loss.item())

                test_acc = self.test()

                epsilon = self.privacy_engine.accountant.get_epsilon(delta=1e-5)
                print(
                    f"Train Epoch: {epoch} \t"
                    f"Loss: {np.mean(losses):.6f} "
                    f"(ε = {epsilon:.2f}, δ = {1e-5})"
                    f"test acc: {test_acc}"
                )
        torch.save(
            {"state_dict": self.model.state_dict()},
            os.path.join(self.MODEL_PATH, "dp_{}.tar").format(
                self.dataset_config["model_name"]
            ),
        )

    def train_nn(self):
        print("Training-Process started!")
        best_ACC = 0.0

        for epoch in range(self.model_config["epochs"]):
            tf = time.time()
            ACC, cnt, loss_tot = 0, 0, 0.0

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

            if (epoch + 1) % 10 == 0:
                torch.save(
                    {"state_dict": self.model.state_dict()},
                    os.path.join(self.CKPT_PATH, "ckpt_epoch{}.tar").format(epoch),
                )

            print(
                "Epoch:{}\tTime:{:.2f}\tTrain Loss:{:.2f}\tTrain Acc:{:.2f}\tTest Acc{:.2f}".format(
                    epoch, interval, train_loss, train_acc, test_acc
                )
            )
        torch.save(
            {"state_dict": self.model.state_dict()},
            os.path.join(self.MODEL_PATH, "{}.tar").format(
                self.dataset_config["model_name"]
            ),
        )

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

    def train_gan(self):
        print("Training-Process started!")
        step = 0

        for epoch in range(self.model_config["epochs"]):
            start = time.time()
            for i, (imgs, label) in enumerate(self.trainloader):
                step += 1
                imgs = imgs.cuda()
                bs = imgs.size(0)

                utils.freeze(self.generator)
                utils.unfreeze(self.discriminator)

                z = torch.randn(bs, self.model_config["z_dim"]).cuda()
                f_imgs = self.generator(z)

                r_logit = self.discriminator(imgs)
                f_logit = self.discriminator(f_imgs)

                wd = r_logit.mean() - f_logit.mean()  # Wasserstein-1 Distance
                gp = utils.gradient_penalty(
                    imgs.data, f_imgs.data, DG=self.discriminator
                )
                dg_loss = -wd + gp * 10.0

                self.optim_dis.zero_grad()
                dg_loss.backward()
                self.optim_dis.step()

                # train Generator
                if step % self.model_config["n_critic"] == 0:
                    utils.freeze(self.discriminator)
                    utils.unfreeze(self.generator)
                    z = torch.randn(bs, self.model_config["z_dim"]).cuda()
                    f_imgs = self.generator(z)
                    logit_dg = self.discriminator(f_imgs)

                    # calculate loss
                    g_loss = -logit_dg.mean()

                    self.optim_gen.zero_grad()
                    g_loss.backward()
                    self.optim_gen.step()

            end = time.time()
            interval = end - start

            print(f"Epoch:{epoch} \t Time:{interval} \t Generator loss: {g_loss}")
            if (epoch + 1) % 10 == 0:
                z = torch.randn(32, self.model_config["z_dim"]).cuda()
                fake_image = self.generator(z)
                utils.save_tensor_image(
                    fake_image.detach(),
                    os.path.join(self.RESULT_PATH, f"result_image_{epoch}.png"),
                    nrow=8,
                )
                torch.save(
                    {"state_dict": self.generator.state_dict()},
                    os.path.join(self.MODEL_PATH, f"ckpts/Generator{epoch}.tar"),
                )

            torch.save(
                {"state_dict": self.generator.state_dict()},
                os.path.join(self.MODEL_PATH, "Generator.tar"),
            )

    def validate_module(self):
        errors = ModuleValidator.validate(self.model, strict=False)
        print(errors)

    def correct_module(self):
        self.model = ModuleValidator.fix(self.model)
        self.validate_module()
        self.optimizer = self.load_optimizer()

    def get_dataset(self, dataset: str, image_size: int = 64):
        dataset_config = self.data_config[dataset]
        if os.path.exists(os.path.join(DATA_PATH, dataset)):
            print("load dataset from directory")
            if dataset_config["torch"] == True:
                # each dataset downloaded from torch existing in ./data
                if dataset_config["gray"]:
                    print("load grayscale dataset \n\n")
                    return utils.download_torchvision(
                        dataset=dataset,
                        dataset_root=os.path.join(DATA_PATH, dataset),
                        transform=transforms.Compose(
                            [
                                transforms.Grayscale(num_output_channels=3),
                                transforms.ToTensor(),
                                transforms.Resize((image_size, image_size)),
                                transforms.Normalize(
                                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                                ),
                            ]
                        ),
                    )
                else:
                    print("load colored dataset \n\n")
                    return utils.download_torchvision(
                        dataset=dataset,
                        dataset_root=os.path.join(DATA_PATH, dataset),
                        transform=transforms.Compose(
                            [
                                transforms.ToTensor(),
                                transforms.Resize((image_size, image_size)),
                                transforms.Normalize(
                                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                                ),
                            ]
                        ),
                    )
            else:
                # each dataset downloaded as directory/zip existing in ./data
                if dataset_config["gray"]:
                    print("load grayscale dataset \n\n")
                    return utils.load_ImageSet(
                        os.path.join(DATA_PATH, dataset),
                        transforms.Compose(
                            [
                                transforms.Grayscale(num_output_channesl=3),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                                ),
                            ]
                        ),
                    )
                else:
                    print("load colored dataset \n\n")
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

    # def get_dataset(self, dataset: str, image_size: int = 64):
    #     dataset_config = self.data_config[dataset]
    #     if os.path.exists(os.path.join(DATA_PATH, dataset)):
    #         print("load dataset from directory")
    #         if dataset_config["torch"] == True:
    #             # each dataset downloaded from torch existing in ./data
    #             if dataset_config["gray"]:
    #                 print("load grayscale dataset \n\n")
    #                 return utils.download_torchvision(
    #                     dataset=dataset,
    #                     dataset_root=os.path.join(DATA_PATH, dataset),
    #                     transform=transforms.Compose(
    #                         [
    #                             transforms.Grayscale(num_output_channels=3),
    #                             transforms.ToTensor(),
    #                             transforms.Resize((image_size, image_size)),
    #                             transforms.Normalize(
    #                                 mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    #                             ),
    #                         ]
    #                     ),
    #                 )
    #             else:
    #                 print("load colored dataset \n\n")
    #                 return utils.download_torchvision(
    #                     dataset=dataset,
    #                     dataset_root=os.path.join(DATA_PATH, dataset),
    #                     transform=transforms.Compose(
    #                         [
    #                             transforms.ToTensor(),
    #                             transforms.Resize((image_size, image_size)),
    #                             transforms.Normalize(
    #                                 mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    #                             ),
    #                         ]
    #                     ),
    #                 )
    #         else:
    #             # each dataset downloaded as directory/zip existing in ./data
    #             if dataset_config['gray']:
    #                 print("load grayscale dataset \n\n")
    #                 return utils.load_ImageSet(
    #                     os.path.join(DATA_PATH, dataset),
    #                     transforms.Compose(
    #                         [
    #                             transforms.Grayscale(num_output_channesl = 3),
    #                             transforms.ToTensor(),
    #                             transforms.Normalize(
    #                                 mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    #                             ),
    #                         ]
    #                     ),
    #                 )
    #             else:
    #                 print("load colored dataset \n\n")
    #                 return utils.load_ImageSet(
    #                     os.path.join(DATA_PATH, dataset),
    #                     transforms.Compose(
    #                         [
    #                             transforms.ToTensor(),
    #                             transforms.Normalize(
    #                                 mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    #                             ),
    #                         ]
    #                     ),
    #                 )

    #     else:
    #         if dataset_config["torch"] == True:
    #             print(" -------- download dataset from torchvision -------- ")
    #             return utils.download_torchvision(
    #                 dataset=dataset,
    #                 dataset_root=os.path.join(DATA_PATH, dataset),
    #                 transform=transforms.Compose(
    #                     [
    #                         transforms.ToTensor(),
    #                         transforms.Normalize(
    #                             mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    #                         ),
    #                     ]
    #                 ),
    #             )
    #         else:
    #             assert dataset_config[
    #                 "torch"
    #             ], f" -------- Please download the dataset manually to data directory (position: {dataset}) and run the script again -------- "

    #     print(dataset_config)

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
        if model_type == "VGG16":
            # return VGG16WithoutBatchNorm(num_classes=self.dataset_config['num_classes'])
            return VGG16(n_classes=self.dataset_config["num_classes"])
        elif model_type == "gan":
            # return Generator(self.model_config["z_dim"]), DGWGAN(3)
            return Generator_MNIST(self.model_config["z_dim"]), DGWGAN_MNIST(1)
        elif model_type == "FaceNet":
            return FaceNet(num_classes=self.dataset_config["num_classes"])
        return None

    def load_optimizer(self):
        return torch.optim.SGD(
            params=self.model.parameters(),
            lr=self.model_config["lr"],
            momentum=self.model_config["momentum"],
            weight_decay=self.model_config["weight_decay"],
        )
