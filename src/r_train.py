import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import pickle

from sparselandtools.dictionaries import DCTDictionary
import os
from tqdm import tqdm
from datetime import datetime
from sacred import Experiment

from sacred import SETTINGS

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

from scipy.special import expit
from pytorch_msssim import MS_SSIM

import sys

sys.path.append("src/")

import r_model, r_generator, r_trainer, r_utils, r_conf

from r_conf import config_ingredient

import warnings

warnings.filterwarnings("ignore")

ex = Experiment("train", ingredients=[config_ingredient])

@ex.automain
def run(cfg):

    hyp = cfg["hyp"]

    print(hyp)

    random_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    PATH = "../results/{}/{}".format(hyp["experiment_name"], random_date)
    os.makedirs(PATH)

    filename = os.path.join(PATH, "hyp.pickle")
    with open(filename, "wb") as file:
        pickle.dump(hyp, file)

    print("load data.")
    if hyp["dataset"] == "MNIST":
        train_loader, test_loader = r_generator.get_MNIST_loaders(
            hyp["batch_size"],
            shuffle=hyp["shuffle"]
            # Not putting train_batch or test_batch options here
        )
    # elif hyp["dataset"] == "simulated":
    #     dataset = r_dataset.SimulatedDataset1D(hyp)
    #     train_loader = DataLoader(
    #         dataset, shuffle=hyp["shuffle"], batch_size=hyp["batch_size"]
    #     )
    else:
        print("dataset is not implemented.")

    # Omitting the init with DCTDictionary - not sure what it does
    H_init = None
    Phi_init = None

    print("create model.")
    net = r_model.RandNet2(hyp, H_init, Phi_init)

    torch.save(net, os.path.join(PATH, "model_init.pt"))


    # different loss functions?
    # use r - r_hat to learn Phi and y - y_hat to learn H? Would need 2 optimizers.
    # try y - y_hat with MSELoss to learn both H and Phi

    if hyp["loss"] == "MSE":
        criterion = torch.nn.MSELoss()
    elif hyp["loss"] == "L1":
        criterion = torch.nn.L1Loss()
    elif hyp["loss"] == "MSSSIM_l1":
        criterion = utils.MSSSIM_l1()
    # for p in net.parameters():
    #     print(p.shape)
    # sys.exit()
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
