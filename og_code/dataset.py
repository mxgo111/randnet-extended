class SimulatedDataset1D(Dataset):
    def __init__(self, hyp, A1=None, A2=None):
        self.num_data = hyp[“num_data”]
        self.x1_dim = hyp[“x1_dim”]
        self.x2_dim = hyp[“x2_dim”]
        self.y_dim = hyp[“y_dim”]
        self.device = hyp[“device”]
        self.A1_col_s = hyp[“A1_col_s”]
        self.x1_s = hyp[“x1_s”]
        self.seed = hyp[“seed”]
        self.A1_dim = (self.x2_dim, self.x1_dim)
        self.A2_dim = (self.y_dim, self.x2_dim)
        self.x1 = generate_sparse_samples(
            self.num_data, self.x1_dim, self.x1_s, device=self.device, seed=self.seed
        )
        if A1 is None:
            self.A1 = generate_sparse_dict(
                self.A1_dim, self.A1_col_s, device=self.device, seed=self.seed
            )
            self.A1 /= torch.sqrt(torch.zeros(1, device=self.device) + self.A1.shape[0])
        else:
            self.A1 = A1.to(self.device)
        if A2 is None:
            self.A2 = torch.randn(self.A2_dim, device=self.device).float()
            self.A2 /= torch.sqrt(torch.zeros(1, device=self.device) + self.A2.shape[0])
        else:
            self.A2 = A2
        self.A1 = F.normalize(self.A1, p=2, dim=0)
        self.A2 = F.normalize(self.A2, p=2, dim=0)

        # generate y in __init__
        # anything you create here should be attributed to object (self.y)
        # self.H = ...
        # can make a dataset object and call (name).H

    def __len__(self):
        return self.num_data
    def __getitem__(self, idx):

        # return the idx index of the generated data

        with torch.no_grad():
            x1 = self.x1[:, idx].reshape(1, -1, 1)
            x2 = torch.matmul(self.A1, x1)
            y = torch.matmul(self.A2, x2)
        return y, x2, x1

        # return y[idx]
