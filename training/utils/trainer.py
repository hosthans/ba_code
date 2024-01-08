import torch
import torch.utils.data as data
import os
import training.utils.helper as utils
import training.utils.dataloader as dl
from torchvision import transforms
from torchvision import datasets
import time
from training.model.classifier import *
from training.model.gan import *
from training.model.facenet import *
import random
from copy import deepcopy
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.utils.batch_memory_manager import BatchMemoryManager
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
from evaluation.evaluator import Plot

CONFIG_PATH = "config/training"
DATA_PATH = "datasets"
MODEL_DATA_PATH = "checkpoints"
RESULT_PATH_GAN = "attack_results"
GRAPH_PATH = "graphs"

MAX_GRAD_NORM = 1.2
EPSILON = 50.0
DELTA = 1e-5


class Trainer:
    """Trainer class:
    Running trainig-loop for adjusting params of neural networks.
    """

    def __init__(self, dataset: str = "mnist", mode: str = "nn"):
        """Initialize neceserray values

        Args:
            model_type (str, optional): Insert the Model type implemented in models.json and implemented in python-code. Defaults to "VGG16".
            dataset (str, optional): Insert the Dataset-Name which should be used for training purposes. Defaults to None.
            mode (str, optional): Insert the training mode (gan or nn) for special training purposes
        """
        # initialize summary_writer for tensorboard
        self.writer = SummaryWriter(f"torchlogs/{mode}_{dataset}")

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
        self.RESULT_PATH = os.path.join(self.MODEL_PATH, "results")
        self.GRAPH_PATH = os.path.join(GRAPH_PATH, mode)

        os.makedirs(DATA_PATH, exist_ok=True)
        os.makedirs(self.MODEL_PATH, exist_ok=True)
        os.makedirs(self.CKPT_PATH, exist_ok=True)
        os.makedirs(self.RESULT_PATH, exist_ok=True)
        os.makedirs(self.GRAPH_PATH, exist_ok=True)

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
            self.train_set, self.test_set = self.get_dataset(
                dataset=dataset, image_size=64
            )
            print(self.train_set)

            print("---------------Loading dataloader----------------")
            self.trainloader = dl.init_dataloader(self.model_config, self.train_set)
            self.testloader = dl.init_dataloader(self.model_config, self.test_set)

            del self.train_set
            del self.test_set

            # neural Network
            self.model = torch.nn.DataParallel(self.load_model(model_type)).to(
                utils.get_device()
            )

            # Optimizer
            self.optimizer = self.load_optimizer()
            self.criterion = nn.CrossEntropyLoss().cuda()

            print("Draw Graph in Tensorboard")
            images, labels = next(iter(self.trainloader))
            grid = torchvision.utils.make_grid(images)
            self.writer.add_image("images", grid, 0)
            self.writer.add_graph(self.model, images)

        elif mode == "gan":
            # load dataset
            print("----------------Loading datasets-----------------")

            self.train_set = self.get_dataset(dataset=dataset, image_size=64, gan=True)
            print(self.train_set)

            self.trainloader = data.DataLoader(
                self.train_set, batch_size=self.model_config["batch_size"], shuffle=True
            )

            del self.train_set

            # # Generator variables
            self.generator, self.discriminator = self.load_model(model_type=model_type)
            self.generator = torch.nn.DataParallel(self.generator).cuda()
            self.discriminator = torch.nn.DataParallel(self.discriminator).cuda()
            # self.generator = Generator(self.model_config['z_dim'])
            # path_G = "checkpoints/gan/Generatorceleba.tar"
            # self.generator = torch.nn.DataParallel(self.generator).cuda()
            # self.generator.load_state_dict(torch.load(path_G)['state_dict'], strict=False)

            # self.discriminator = DGWGAN(3)
            # path_D = "checkpoints/gan/Discriminatorceleba.tar"
            # self.discriminator = torch.nn.DataParallel(self.discriminator).cuda()
            # self.discriminator.load_state_dict(torch.load(path_D)['state_dict'], strict=False)

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

        elif mode == "dpnn":
            # dataset
            print("----------------Loading datasets-----------------")
            self.train_set, self.test_set = self.get_dataset(
                dataset=dataset, image_size=64, gan=False
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

            # Optimizer
            self.optimizer = self.load_optimizer()
            self.criterion = nn.CrossEntropyLoss().cuda()

            print("Draw Graph in Tensorboard")
            images, labels = next(iter(self.trainloader))
            grid = torchvision.utils.make_grid(images)
            self.writer.add_image("images", grid, 0)
            self.writer.add_graph(self.model, images)

        else:
            assert False, "training method not implemented yet"

    def train(self):
        if self.mode == "nn":
            loss, acc, loss_t, acc_t = self.train_nn()
            return loss, acc, loss_t, acc_t
        elif self.mode == "gan":
            if self.dataset_config['name'] in ["mnist", "qmnist"]:
                self.train_gan_mnist()
            else:
                self.train_gan()
        elif self.mode == "dpnn":
            loss, acc, loss_t, acc_t, eps = self.train_dp()
            return loss, acc, loss_t, acc_t, eps
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

        train_acc_list = []
        train_loss_list = []
        test_acc_list = []
        test_loss_list = []
        epsilon_list = []
        epochs = np.arange(1, self.model_config["epochs"]+1)

        for epoch in range(self.model_config["epochs"]):
            tf = time.time()
            ACC, cnt, loss_tot = 0, 0, 0.0
            self.model.train()
            with BatchMemoryManager(
                data_loader=self.trainloader,
                max_physical_batch_size=32,
                optimizer=self.optimizer,
            ) as memory_safe_data_loader:
                for _batch_idx, (img, iden) in enumerate(memory_safe_data_loader):
                    img, iden = img.to(utils.get_device()), iden.to(
                        utils.get_device()
                    )
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
                    loss_tot += loss.item()*bs
                    cnt += bs
                    
                train_loss, train_acc = loss_tot * 1.0 / cnt, ACC * 100.0 / cnt
                test_loss, test_acc = self.test()

                epsilon = self.privacy_engine.accountant.get_epsilon(delta=1e-5)

                train_acc_list.append(train_acc)
                train_loss_list.append(train_loss)
                test_loss_list.append(test_loss)
                test_acc_list.append(test_acc)
                epsilon_list.append(epsilon)

                # self.writer.add_scalar(
                #     "train/lr", self.model_config["lr"], global_step=epoch
                # )
                #self.writer.add_scalar("train/loss", np.mean(losses), global_step=epoch)
                # self.writer.add_scalar("test/confidence", test_acc, global_step=epoch)
            
            if test_acc > best_ACC:
                best_ACC = test_acc

            if (epoch + 1) % 10 == 0:
                torch.save(
                    {"state_dict": self.model.state_dict()},
                    os.path.join(self.CKPT_PATH, "dp_ckpt_epoch{}.tar").format(epoch),
                )

            interval = time.time() - tf
            print(
                "Epoch:{}\tTime:{:.2f}\tTrain Loss:{:.2f}\tTrain Acc:{:.2f}\tTest Acc{:.2f}".format(
                    epoch, interval, train_loss, train_acc, test_acc
                )
            )
        
        train_loss_summary = ["VGG16-DP - loss", epochs, train_loss_list]
        train_acc_summary = ["VGG16-DP - accuracy", epochs, train_acc_list]
        test_loss_summary = ["VGG16 - loss (test)", epochs, test_loss_list]
        test_acc_summary = ["VGG16-DP - accuracy (test)", epochs, test_acc_list]
        epsilon_summary = ["DP-Epsilons", epochs, epsilon_list]

        torch.save(
            {"state_dict": self.model.state_dict()},
            os.path.join(self.MODEL_PATH, "dp_{}{}.tar").format(
                self.dataset_config["model_name"], self.dataset_config["name"]
            ),
        )

        return train_loss_summary, train_acc_summary, test_loss_summary, test_acc_summary, epsilon_summary

    def train_nn(self):
        print("Training-Process started!")
        best_ACC = 0.0
        train_acc_list = []
        train_loss_list = []
        test_acc_list = []
        test_loss_list = []
        epochs = np.arange(1, self.model_config["epochs"]+1)

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

                # global_step = epoch * len(self.trainloader) * self.model_config['batch_size'] + i

            train_loss, train_acc = loss_tot * 1.0 / cnt, ACC * 100.0 / cnt

            train_acc_list.append(train_acc)
            train_loss_list.append(train_loss)

            # self.writer.add_scalar(
            #     "train/lr", self.model_config["lr"], global_step=epoch
            # )
            # self.writer.add_scalar("train/loss", loss.item(), global_step=epoch)
            # self.writer.add_scalar("train/confidence", train_acc, global_step=epoch)

            test_loss, test_acc = self.test()
            test_acc_list.append(test_acc)
            test_loss_list.append(test_loss)

            # self.writer.add_scalar("test/confidence", test_acc, global_step=epoch)

            interval = time.time() - tf
            if test_acc > best_ACC:
                best_ACC = test_acc

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
        
        train_loss_summary = ["VGG16 - loss", epochs, train_loss_list]
        test_loss_summary = ["VGG16 - loss (test)", epochs, test_loss_list]
        train_acc_summary = ["VGG16 - accuracy", epochs, train_acc_list]
        test_acc_summary = ["VGG16 - accuracy (test)", epochs, test_acc_list]

        torch.save(
            {"state_dict": self.model.state_dict()},
            os.path.join(self.MODEL_PATH, "nn_{}{}.tar").format(
                self.dataset_config["model_name"], self.dataset_config["name"]
            ),
        )

        return train_loss_summary, train_acc_summary, test_loss_summary, test_acc_summary

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

            # calculate loss
            batch_loss = F.cross_entropy(out_prob, iden, reduction='sum')
            loss += batch_loss.item()

        avg_loss = loss/cnt

        return avg_loss, ACC * 100.0 / cnt

    def train_gan_mnist(self):
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
            if (epoch + 1) % 5 == 0:
                z = torch.randn(32, self.model_config["z_dim"]).cuda()
                fake_image = self.generator(z)
                utils.save_tensor_image(
                    fake_image.detach(),
                    os.path.join(self.RESULT_PATH, f"result_image_{epoch}.png"),
                    nrow=8,
                )
                torch.save(
                    {"state_dict": self.generator.state_dict()},
                    os.path.join(
                        self.MODEL_PATH,
                        f"ckpts/Generator{self.dataset_config['name']}_epoch{epoch}.tar",
                    ),
                )
                grid = torchvision.utils.make_grid(fake_image.detach())
                self.writer.add_image(f"images_epoch_{epoch}", grid, 0)

            self.writer.add_scalar(
                "generator/train/lr", self.model_config["lr"], global_step=epoch
            )
            self.writer.add_scalar("generator/train/loss", g_loss, global_step=epoch)

        torch.save(
            {"state_dict": self.generator.state_dict()},
            os.path.join(
                self.MODEL_PATH,
                f"Generator{self.dataset_config['name']}.tar",
            ),
        )

        torch.save(
            {"state_dict": self.discriminator.state_dict()},
            os.path.join(
                self.MODEL_PATH,
                f"Discriminator{self.dataset_config['name']}.tar",
            ),
        )

    def train_gan(self):
        print("Training-Process started!")
        step = 0

        for epoch in range(self.model_config["epochs"]):
            start = time.time()
            for i, imgs in enumerate(self.trainloader):
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
            if (epoch + 1) % 5 == 0:
                z = torch.randn(32, self.model_config["z_dim"]).cuda()
                fake_image = self.generator(z)
                utils.save_tensor_image(
                    fake_image.detach(),
                    os.path.join(self.RESULT_PATH, f"result_image_{epoch}.png"),
                    nrow=8,
                )
                torch.save(
                    {"state_dict": self.generator.state_dict()},
                    os.path.join(
                        self.MODEL_PATH,
                        f"ckpts/Generator{self.dataset_config['name']}_epoch{epoch}.tar",
                    ),
                )
                grid = torchvision.utils.make_grid(fake_image.detach())
                self.writer.add_image(f"images_epoch_{epoch}", grid, 0)

            self.writer.add_scalar(
                "generator/train/lr", self.model_config["lr"], global_step=epoch
            )
            self.writer.add_scalar("generator/train/loss", g_loss, global_step=epoch)

        torch.save(
            {"state_dict": self.generator.state_dict()},
            os.path.join(
                self.MODEL_PATH,
                f"Generator{self.dataset_config['name']}.tar",
            ),
        )

        torch.save(
            {"state_dict": self.discriminator.state_dict()},
            os.path.join(
                self.MODEL_PATH,
                f"Discriminator{self.dataset_config['name']}.tar",
            ),
        )

    def validate_module(self):
        errors = ModuleValidator.validate(self.model, strict=False)
        print(errors)

    def correct_module(self):
        self.model = ModuleValidator.fix(self.model)
        self.validate_module()
        self.optimizer = self.load_optimizer()

    def get_dataset(self, dataset: str, image_size: int = 64, gan: bool = False):
        dataset_config = self.data_config[dataset]
        if dataset == "mnist" and gan == False:
            train_set = datasets.MNIST(
                root="datasets/mnist",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.Grayscale(num_output_channels=3),
                        transforms.ToTensor(),
                        transforms.Resize((image_size, image_size)),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    ]
                ),
            )

            test_set = datasets.MNIST(
                root="datasets/mnist",
                train=False,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.Grayscale(num_output_channels=3),
                        transforms.ToTensor(),
                        transforms.Resize((image_size, image_size)),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    ]
                ),
            )
            return train_set, test_set

        elif dataset == "mnist" and gan == True:
            train_set = datasets.MNIST(
                root="datasets/mnist",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.Resize((image_size, image_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5], std=[0.5]),
                    ]
                ),
            )
            return train_set
        
        elif dataset == "qmnist" and gan == False:
            train_set = datasets.QMNIST(
                root="datasets/qmnist",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.Grayscale(num_output_channels=3),
                        transforms.ToTensor(),
                        transforms.Resize((image_size, image_size)),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    ]
                ),
            )

            test_set = datasets.QMNIST(
                root="datasets/qmnist",
                train=False,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.Grayscale(num_output_channels=3),
                        transforms.ToTensor(),
                        transforms.Resize((image_size, image_size)),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    ]
                ),
            )
            return train_set, test_set

        elif dataset == "qmnist" and gan == True:
            train_set = datasets.QMNIST(
                root="datasets/qmnist",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.Resize((image_size, image_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5], std=[0.5]),
                    ]
                ),
            )
            return train_set

        elif dataset == "celeba" and gan == False:
            train_set = dl.ImageFolder(
                config=self.dataset_config,
                file_path=self.dataset_config["train_file"],
                mode="train",
            )
            test_set = dl.ImageFolder(
                config=self.dataset_config,
                file_path=self.dataset_config["test_file"],
                mode="test",
            )

            return train_set, test_set

        elif dataset == "celeba" and gan == True:
            gan_set = dl.ImageFolder(
                config=self.dataset_config,
                file_path=self.dataset_config["gan_file"],
                mode="gan",
            )

            return gan_set

        else:
            print("Dataset not configured yet")
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
        if model_type == "VGG16":
            # return VGG16WithoutBatchNorm(num_classes=self.dataset_config['num_classes'])
            return VGG16(n_classes=self.dataset_config["num_classes"])
        elif model_type == "VGG16_celeba":
            # return VGG16WithoutBatchNorm(num_classes=self.dataset_config['num_classes'])
            return VGG16(n_classes=self.dataset_config["num_classes"])
        elif model_type == "gan":
            # return Generator(self.model_config["z_dim"]), DGWGAN(3)
            if self.dataset_config['name'] in ["mnist", "qmnist"]:
                return Generator_MNIST(self.model_config["z_dim"]), DGWGAN_MNIST(1)
            else:
                return Generator(self.model_config["z_dim"]), DGWGAN(3)
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
