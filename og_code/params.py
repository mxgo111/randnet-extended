import torch

#@title Define Data Generation Parameters { run: "auto" }
num_examples = 5
num_filters = 2
minibatch = num_examples
dict_dim = 50
y_dim = 500
x_dim = y_dim - dict_dim + 1
sparsity = 2
seed = 100
random = False
L = 100
num_iters = 70
sigma = 0.1
epoch_start = 0
epoch_end = 78

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
device = torch.device("cpu")

# #@title Hyperparameters { run: "auto", display-mode: "form" }
hyp = {
  # "experiment_name": "test_mnist",
  # "dataset": "MNIST",
  # "network": "CRsAE2D",
  "dictionary_dim": dict_dim,
  "stride": 1,
  "num_conv": num_filters,
  "L": L,
  "lambda": 1,
  "num_iters": num_iters,
  # "batch_size": 1,
  # "num_epochs": 500,
  # "zero_mean_filters": False,
  "normalize": False,
  "lr": 0.03,

  # usually for images, lr = 10 ** (-2 to -4)

  "lr_decay": 0.99,
  "lr_step": 30,

  # after lr_step epochs, decrease lr by multiplying it by lr_decay

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
