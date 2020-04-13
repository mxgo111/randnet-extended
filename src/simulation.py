"""
Copyright (c) 2019 CRISP

train

:author: Bahareh Tolooshams
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pickle
import os
from datetime import datetime

import sys

sys.path.append("src/")

import model, generator, trainer, utils, conf

from conf import config_ingredient

import warnings

warnings.filterwarnings("ignore")



#@title Generate Sparse Samples Function { run: "auto", display-mode: "form" }
def generate_sparse_samples(n, num_filters, dim, s, device, seed=None, unif=True):
    samples = torch.zeros((n, num_filters, dim), device=device)
    if random == True:
      torch.manual_seed(seed)
      np.random.seed(seed)
    for i in range(n):
      for j in range(num_filters):
        ind = np.random.choice(dim, s, replace=False)
        if unif:
            # draws amplitude from [-2,-1] U [1,2] uniformly
            samples[i][j][ind] = (torch.rand(s, device=device) + 1) * (
                torch.randint(low=0, high=2, size=(1, 1), device=device).float() * 2 - 1
            )
        else:
            # amplitude is 1 or -1 .5 prob of each
            samples[i][j][ind] = torch.ones(s, device=device) * (
                torch.randint(low=0, high=2, size=(1, 1), device=device).float() * 2 - 1
            )
    return samples
    # return samples.T


    hyp = cfg["hyp"]

    print(hyp)

    # random_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    #
    # PATH = "../results/{}/{}".format(hyp["experiment_name"], random_date)
    # os.makedirs(PATH)

    num_examples = 5
    num_filters = 2
    minibatch = num_examples
    dict_dim = 50
    y_dim = 500
    x_dim = y_dim - dict_dim + 1
    sparsity = 2
    seed = 100
    random = False
    L = 50
    num_iters = 100
    sigma = 0.1
    epoch_start = 1
    epoch_end = 10

    print("load data.")
        #@title Generate and Plot Sparse Samples { run: "auto", display-mode: "form" }
    # Generate Sparse Samples
    x_torch = generate_sparse_samples(num_examples, num_filters, x_dim, sparsity, device=device, seed=seed)
    # x_torch.requires_grad = True
    x_np = x_torch.numpy()
    print("x_torch.size() = " + str(x_torch.size()))

    # Plots Sparse Codes
    x_axis = np.arange(x_dim)
    # for j in range(num_examples):
    #   for i in range(num_filters):
    #     plt.scatter(x_axis, x_np[j][i])

    #@title Generate and Plot Data from Sparse Samples { run: "auto", display-mode: "form" }
    # Generates y via y = Hx
    y = F.conv_transpose1d(x, weights)

    print(y.size())

    # Plots resulting convolution
    for i in range(2):
      print("example " + str(i+1) + ":")
      plt.plot(np.arange(y.size()[-1]), y.numpy()[i][0])
      plt.show()
    #@title Define Data Generation Parameters { run: "auto" }


    #@title Hyperparameters { run: "auto", display-mode: "form" }
    hyp = {
      # "experiment_name": "test_mnist",
      # "dataset": "MNIST",
      # "network": "CRsAE2D",
      "dictionary_dim": dict_dim,
      "stride": 1,
      "num_conv": num_filters,
      "L": L,
      "num_iters": num_iters,

      # "batch_size": 1,
      # "num_epochs": 500,
      # "zero_mean_filters": False,
      "normalize": False,
      "lr": 100,
      "lr_decay": 1,
      "lr_step": 30,
      # "lr_lam": 0.001,
      # "noiseSTD": 20,
      # "shuffle": True,
      # "test_path": "../data/test_img/",
      "info_period": 20,
      # "denoising": True,
      # "supervised": True,
      # "crop_dim": (28, 28),
      # "init_with_DCT": True,

      "sigma": sigma,

      # "lam": 0.1,

      "twosided": True,

      "loss": "MSE",
      # "use_lam": False,
      # "delta": 100,
      # "trainable_bias": False,
      # "cyclic": False,

      "device": device,
    }

    H_init = None

    net = model.CRsAE1D(hyp, H_init)


    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=hyp["lr"], eps=1e-3)

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=hyp["lr_step"], gamma=hyp["lr_decay"]
    )

    print("train auto-encoder.")
    for idx, epoch in tqdm(enumerate(range(epoch_start, epoch_end))):
        # scheduler.step()
        loss_all = 0
        for i in range(y.size()[0]):
            optimizer.zero_grad()
            y_hat = net.forward(y)[0][i] # change this
            loss = criterion(y_hat, y[i])
              # print("y = ",torch.sum(abs(y[i])))
              # print("y_hat = ", torch.sum(abs(y_hat)))
                    # y_hat = net(y)[0][i] # change this
              # loss = criterion(y[i], y_hat)

            a = list(net.parameters())[0].clone()
            print("grads not zero? ", list(net.parameters())[0].grad)
            loss.backward()
            optimizer.step()
            b = list(net.parameters())[0].clone()
            print("are they equal? ", torch.equal(a.data, b.data))


        # loss.backward()
        #   # for param in net.parameters():
        #   #   print(param.grad.data.sum())

        #   # start debugger
        #   # import pdb; pdb.set_trace()
        # for param in net.parameters():
        #   print("H before optimizer.step() =", param[0][0][:5])
        # optimizer.step()
        # for param in net.parameters():
        #   print("H after optimizer.step() =", param[0][0][:5])
        # print("loss after optimizer.step() = ", loss.item())
          # loss_all += float(loss.item())

          # backward
          # optimizer.zero_grad()

        if idx % info_period == 0:
          print("loss:{:.8f} ".format(loss.item()))
        torch.cuda.empty_cache()

        for p in net.parameters():
            print(p.size())
        # if p.grad is not None:
        #     print(p.grad.data)

    print("training finished!")
