import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pickle

from sparselandtools.dictionaries import DCTDictionary
import os, sys
from tqdm import tqdm
from datetime import datetime
from sacred import Experiment

from sacred import SETTINGS
from tensorboardX import SummaryWriter

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

from scipy.special import expit
from pytorch_msssim import MS_SSIM

sys.path.append("src/")

import model, generator, r_trainer, utils, r_conf

from conf import config_ingredient

import warnings

warnings.filterwarnings("ignore")

ex = Experiment("train", ingredients=[config_ingredient])

@ex.automain
def run(cfg):
    hyp = cfg["hyp"]

    random_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    # PATH = "../results/{}/{}".format(hyp["experiment_name"], random_date)
    PATH = "../results/{}".format(random_date)
    os.makedirs(PATH)
    writer = SummaryWriter(PATH)

    print("load data.")
    if hyp["dataset"] == "MNIST":
        train_loader, test_loader = generator.get_MNIST_loaders(
            hyp["batch_size"],
            shuffle=hyp["shuffle"]
        )
    elif hyp["dataset"] == "Image":
        train_loader = generator.get_path_loader(
        batch_size=hyp["batch_size"],
        image_path="../test_img/",
        shuffle=hyp["shuffle"],
        crop_dim=hyp["crop_dim"]
        )
    else:
        print("dataset is not implemented.")

    # Visualizations
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    print("images.shape = ", images.shape)

    # create grid of images
    img_grid = torchvision.utils.make_grid(images)

    # show images
    # utils.matplotlib_imshow(img_grid)

    # write original images to tensorboard
    writer.add_image('mnist_images', img_grid)

    H_init = None

    Phi_init = torch.eye(
        784, device='cpu',
        #dimension of data to smaller dimension
    )

    print("create model.")
    net = model.RandNet2(hyp, H_init, Phi_init)

    torch.save(net, os.path.join(PATH, "model_init.pt"))

    r, r_hat, y_hat, x_new, lam = net(images)

    print("model ran.")

    r_hat = r_hat.view(-1, 1, 28, 28)
    print(r_hat.shape)
    # print(y_hat.shape)
    # print(r_utils.find_maximum(y_hat))

    r = r.view(-1, 1, 28, 28)

    img_grid = torchvision.utils.make_grid(r)

    # utils.matplotlib_imshow(img_grid)
    # plt.show()

    writer.add_image('r', img_grid)

    print("add first image.")

    # plt.imshow(r[0][0].detach().numpy(), cmap='gray')
    # plt.show()
    #
    # plt.imshow(r_hat[0][0].detach().numpy(), cmap='gray')
    # plt.show()

    print("image shows")

    img_grid = torchvision.utils.make_grid(r_hat)

    # utils.matplotlib_imshow(img_grid)
    writer.add_image('r_hat', img_grid)

    img_grid = torchvision.utils.make_grid(y_hat)

    # utils.matplotlib_imshow(img_grid)
    writer.add_image('y_hat', img_grid)

    # writer.add_graph(net, images)

    writer.close()

    sys.exit()

    print("tensorboard --logdir={} --host localhost --port 8088".format(PATH))

    # different loss functions?
    # use r - r_hat to learn Phi and y - y_hat to learn H? Would need 2 optimizers.
    # try y - y_hat with MSELoss to learn both H and Phi

    if hyp["loss"] == "MSE":
        criterion = torch.nn.MSELoss()
    elif hyp["loss"] == "L1":
        criterion = torch.nn.L1Loss()
    elif hyp["loss"] == "MSSSIM_l1":
        criterion = utils.MSSSIM_l1()

    optimizer = optim.Adam(net.parameters(), lr=hyp["lr"], eps=1e-3)

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=hyp["lr_step"], gamma=hyp["lr_decay"]
    )

    print("train auto-encoder.")
    print("assume that the image is square.")

    net = r_trainer.train_ae(
        net,
        train_loader,
        hyp,
        criterion,
        optimizer,
        scheduler,
        PATH,
        test_loader,
        epoch_start=0,
        epoch_end=hyp["num_epochs"],
    )

    print("training finished!")
