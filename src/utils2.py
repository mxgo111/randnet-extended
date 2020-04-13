import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#@title Generate Sparse Samples Function { run: "auto", display-mode: "form" }
def generate_sparse_samples(n, num_filters, dim, s, device, random=True, seed=None, unif=True):
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

            # samples[i][j][ind] = (torch.rand(s, device=device) + 1) * (
            #     torch.randint(low=0, high=2, size=(1, 1), device=device).float() * 2 - 1
            # )

        else:
            # amplitude is 1 or -1 .5 prob of each
            samples[i][j][ind] = torch.ones(s, device=device) * (
                torch.randint(low=0, high=2, size=(1, 1), device=device).float() * 2 - 1
            )
    return samples
    # return samples.T
