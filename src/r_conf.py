import torch

from sacred import Experiment, Ingredient

config_ingredient = Ingredient("cfg")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@config_ingredient.config
def cfg():
    dict_dim = 5
    y_dim = 784
    r_dim = 356 # What value should this be?
    epoch_start = 0
    epoch_end = 20
    hyp = {
        "experiment_name": "test",
        "dataset": "MNIST",
        "network": "RandNet2",
        "device": device,
        "dictionary_dim": dict_dim,
        "num_examples": 5,
        # "minibatch": 5,
        "num_conv": 32,
        "x_dim": y_dim - dict_dim + 1,
        "y_dim": y_dim,
        "r_dim": r_dim,
        "sparsity": 2,
        "seed": 100,
        "random": False,
        "L": 100,
        "num_iters": 50,
        "sigma": 0.03,
        "stride": 1,
        "normalize": True,
        "lr": 0.005,
        "lr_decay": 0.7,
        "lr_step": 10,
        "lr_lam": 0.001,
        "twosided": True,
        "loss": "MSE",
        "shuffle": True,
        "batch_size": 32,
        "num_epochs": 100, # 500
        "use_lam": False,
        "info_period": 100,
        "denoising": True,
        "supervised": True,
        "trainistrue": True,
    }
