import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np

#@title Generate Sparse Samples Function { run: "auto", display-mode: "form" }
def generate_sparse_samples(n, num_conv, dim, s, device, random=True, seed=None, unif=True):
    samples = torch.zeros((n, num_conv, dim), device=device)
    if random == True:
      torch.manual_seed(seed)
      np.random.seed(seed)
    for i in range(n):
      for j in range(num_conv):
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

class SimulatedDataset1D(Dataset):
    def __init__(self, hyp, H=None):
        self.num_examples = hyp["num_examples"]
        # self.minibatch = hyp["minibatch"]
        self.x_dim = hyp["x_dim"]
        self.y_dim = hyp["y_dim"]
        self.H_dim = hyp["dictionary_dim"]
        self.num_conv = hyp["num_conv"]
        self.device = hyp["device"]
        self.random = hyp["random"]
        self.seed = hyp["seed"]
        x = generate_sparse_samples(
            self.num_examples, self.num_conv, self.x_dim, self.sparsity,
            device=self.device, random=self.random, seed=self.seed
        )
        if H is None:
            H = torch.randn((self.num_conv, 1, self.dictionary_dim), device=self.device)
        else:
            self.H = H.to(self.device)
        self.H = F.normalize(self.H, p=2, dim=-1)
        weights = torch.zeros(self.num_conv, 1, self.dictionary_dim)

        for i in range(self.num_conv):
          weights[i,0,:] = H[i]

        self.y = F.conv_transpose1d(x, weights)

    def __len__(self):
        return self.num_examples

    def __getitem(self, idx):
        with torch.no_grad():
            x = self.x[idx]
            y = self.y[idx]
        return y, x
