"""
Copyright (c) 2019 CRISP

crsae model

:author: Bahareh Tolooshams
"""

import torch
import torch.nn.functional as F
import numpy as np

import utils


class RELUTwosided(torch.nn.Module):
    def __init__(self, num_conv, lam=1e-3, L=100, sigma=1, device=None):
        super(RELUTwosided, self).__init__()
        self.L = L
        self.lam = torch.nn.Parameter(
            lam * torch.ones(1, num_conv, 1, 1, device=device)
        )
        self.sigma = sigma
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        lam = self.relu(self.lam) * (self.sigma ** 2)
        mask1 = (x > (lam / self.L)).type_as(x)
        mask2 = (x < -(lam / self.L)).type_as(x)
        out = mask1 * (x - (lam / self.L))
        out += mask2 * (x + (lam / self.L))
        return out


class CRsAE1D(torch.nn.Module):
    def __init__(self, hyp, H=None):
        super(CRsAE1D, self).__init__()

        self.T = hyp["num_iters"]
        self.L = hyp["L"]
        self.num_conv = hyp["num_conv"]
        self.dictionary_dim = hyp["dictionary_dim"]
        self.device = hyp["device"]
        self.sigma = hyp["sigma"]
        self.stride = hyp["stride"]
        self.twosided = hyp["twosided"]
        self.lam = hyp["lambda"]

        if H is None:
            H = torch.randn((self.num_conv, 1, self.dictionary_dim), device=self.device)
            H = F.normalize(H, p=2, dim=-1)
        self.register_parameter("H", torch.nn.Parameter(H))

        self.relu = torch.nn.ReLU()

    def get_param(self, name):
        return self.state_dict(keep_vars=True)[name]

    def normalize(self):
        self.get_param("H").data = F.normalize(self.get_param("H").data, p=2, dim=-1)

    def zero_mean(self):

        self.get_param("H").data -= torch.mean(self.get_param("H").data, dim=-0)

    def forward(self, x):
        num_batches = x.shape[0]

        D_in = x.shape[2]
        D_enc = F.conv1d(x, self.get_param("H"), stride=self.stride).shape[-1]

        self.lam = self.sigma * torch.sqrt(
            2 * torch.log(torch.zeros(1, device=self.device) + (self.num_conv * D_enc))
        )

        # the above line <- see paper from Bahareh
        # estimate lambda depending on dimension of code (D_enc) and background noise (sigma is standard deviation of background noise)
        # sigma is large -> we want a larger lambda (the more noise, larger the sigma, should have larger lambda)
        # if code is very long, then we should enforce more sparsity
        # if code is very long, then l1 norm is large so we want lambda to be larger

        # set lambda to be 10, run the forward, if code estimate is 0 then lambda should be smaller.
        # when you want to tune lambda, set T to be large (large number of iterations)
        # lambda example: 0.1

        x_old = torch.zeros(num_batches, self.num_conv, D_enc, device=self.device)
        yk = torch.zeros(num_batches, self.num_conv, D_enc, device=self.device)
        x_new = torch.zeros(num_batches, self.num_conv, D_enc, device=self.device)
        t_old = torch.tensor(1, device=self.device).float()
        for t in range(self.T):
            H_wt = x - F.conv_transpose1d(yk, self.get_param("H"), stride=self.stride)
            x_new = (
                yk + F.conv1d(H_wt, self.get_param("H"), stride=self.stride) / self.L
            )

            # above line <- division by L : if L is very large, then you are not changing x much from the previous iteration
            # if L is large, you get a small step size : algorithm will converge slowly
            # which implies you need a larger T
            # for example, larger T and still bad reconstruction then L needs to be smaller
            # Minimium value of L is 1
            # If you get nans or infinity, then L is too small <- need to increase L
            # L = 10 is an example, or L = 100 to 500 (general range)

            if self.twosided:
                x_new = self.relu(torch.abs(x_new) - self.lam / self.L) * torch.sign(
                    x_new
                )
            else:
                x_new = self.relu(x_new - self.lam / self.L)

            t_new = (1 + torch.sqrt(1 + 4 * t_old * t_old)) / 2
            yk = x_new + (t_old - 1) / t_new * (x_new - x_old)

            x_old = x_new
            t_old = t_new

        z = F.conv_transpose1d(x_new, self.get_param("H"), stride=self.stride)

        return z, x_new, self.lam


class CRsAE2D(torch.nn.Module):
    def __init__(self, hyp, H=None, Phi=None):
        super(CRsAE2D, self).__init__()

        self.T = hyp["num_iters"]
        self.L = hyp["L"]
        self.num_conv = hyp["num_conv"]
        self.dictionary_dim = hyp["dictionary_dim"]
        self.device = hyp["device"]
        self.zero_mean_filters = hyp["zero_mean_filters"]
        self.sigma = hyp["sigma"]
        self.stride = hyp["stride"]
        self.twosided = hyp["twosided"]
        self.use_lam = hyp["use_lam"]

        if self.use_lam:
            self.lam = hyp["lam"]

        if H is None:
            H = torch.randn(
                (self.num_conv, 1, self.dictionary_dim, self.dictionary_dim),
                device=self.device,
            )
            H = F.normalize(H, p="fro", dim=(-1, -2))
        self.register_parameter("H", torch.nn.Parameter(H))

        if Phi is None:
            Phi = torch.randn(
                (self.r_dim, self.y_dim), device=self.device,
                #dimension of data to smaller dimension
            )
            Phi = F.normalize(Phi, p=2, dim=(-1))
        self.register_parameter("Phi", torch.nn.Parameter(Phi))

        self.relu = torch.nn.ReLU()

    def get_param(self, name):
        return self.state_dict(keep_vars=True)[name]

    def normalize(self):
        self.get_param("H").data = F.normalize(
            self.get_param("H").data, p="fro", dim=(-1, -2)
        )

    def zero_mean(self):
        self.get_param("H").data -= torch.mean(self.get_param("H").data, dim=0)

    def split_image(self, x):
        if self.stride == 1:
            return x, torch.ones_like(x)
        left_pad, right_pad, top_pad, bot_pad = utils.calc_pad_sizes(
            x, self.dictionary_dim, self.stride
        )
        x_batched_padded = torch.zeros(
            x.shape[0],
            self.stride ** 2,
            x.shape[1],
            top_pad + x.shape[2] + bot_pad,
            left_pad + x.shape[3] + right_pad,
            device=self.device,
        ).type_as(x)
        valids_batched = torch.zeros_like(x_batched_padded)
        for num, (row_shift, col_shift) in enumerate(
            [(i, j) for i in range(self.stride) for j in range(self.stride)]
        ):
            x_padded = F.pad(
                x,
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="reflect",
            )
            valids = F.pad(
                torch.ones_like(x),
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="constant",
            )
            x_batched_padded[:, num, :, :, :] = x_padded
            valids_batched[:, num, :, :, :] = valids
        x_batched_padded = x_batched_padded.reshape(-1, *x_batched_padded.shape[2:])
        valids_batched = valids_batched.reshape(-1, *valids_batched.shape[2:])
        return x_batched_padded, valids_batched

    def forward(self, x):
        x_batched_padded, valids_batched = self.split_image(x)

        num_batches = x_batched_padded.shape[0]

        D_enc1 = F.conv2d(
            x_batched_padded, self.get_param("H"), stride=self.stride
        ).shape[2]
        D_enc2 = F.conv2d(
            x_batched_padded, self.get_param("H"), stride=self.stride
        ).shape[3]

        if not self.use_lam:
            self.lam = self.sigma * torch.sqrt(
                2
                * torch.log(
                    torch.tensor(
                        self.num_conv * D_enc1 * D_enc2, device=self.device
                    ).float()
                )
            )

        x_old = torch.zeros(
            num_batches, self.num_conv, D_enc1, D_enc2, device=self.device
        )
        yk = torch.zeros(num_batches, self.num_conv, D_enc1, D_enc2, device=self.device)
        x_new = torch.zeros(
            num_batches, self.num_conv, D_enc1, D_enc2, device=self.device
        )

        del D_enc1
        del D_enc2
        del num_batches

        t_old = torch.tensor(1, device=self.device).float()

        for t in range(self.T):
            Hyk = F.conv_transpose2d(yk, self.get_param("H"), stride=self.stride)
            # This line finds H yk, yk is in the input
            # We want Phi H yk
            # add a line that finds Phi H yk by multiplication (recall H is a convolution, Phi is a matrix)

            x_tilda = x_batched_padded - Hyk

            # to implement Phi
            # wherever you have conv_transpose2d we need to change H to

            x_new = (
                yk + F.conv2d(x_tilda, self.get_param("H"), stride=self.stride) / self.L
            )
            # This line finds H_transpose x_tilda
            # We want to find Phi-transpose
            # before the above line, we should first apply Phi-transpose x_tilda, then the above line on the output

            if self.twosided:
                x_new = (x_new > (self.lam / self.L)).float() * (
                    x_new - (self.lam / self.L)
                ) + (x_new < -(self.lam / self.L)).float() * (
                    x_new + (self.lam / self.L)
                )
            else:
                x_new = (x_new > (self.lam / self.L)).float() * (
                    x_new - (self.lam / self.L)
                )

            t_new = (1 + torch.sqrt(1 + 4 * t_old * t_old)) / 2
            yk = x_new + ((t_old - 1) / t_new) * (x_new - x_old)

            x_old = x_new
            t_old = t_new

        z = (
            torch.masked_select(
                F.conv_transpose2d(x_new, self.get_param("H"), stride=self.stride),
                valids_batched.byte(),
            ).reshape(x.shape[0], self.stride ** 2, *x.shape[1:])

            # do the same thing for conv_transpose above
            # to get y_hat, need to get H x_new
            # to get r_hat, find phi H x_new
            # output r_hat, y_hat, and x_new, and (lambda)

            # challenge: reshaping the input to Phi, because the image is 2d but need 1d for matrix multiplication - how to turn back to 2d?

        ).mean(dim=1, keepdim=False)

        return z, x_new, self.lam


class CRsAE2DTrainableBias(torch.nn.Module):
    def __init__(self, hyp, H=None):
        super(CRsAE2DTrainableBias, self).__init__()

        self.T = hyp["num_iters"]
        self.L = hyp["L"]
        self.num_conv = hyp["num_conv"]
        self.dictionary_dim = hyp["dictionary_dim"]
        self.device = hyp["device"]
        self.zero_mean_filters = hyp["zero_mean_filters"]
        self.lam = hyp["lam"]
        self.stride = hyp["stride"]
        self.twosided = hyp["twosided"]
        self.sigma = hyp["sigma"]

        if H is None:
            H = torch.randn(
                (self.num_conv, 1, self.dictionary_dim, self.dictionary_dim),
                device=self.device,
            )
            H = F.normalize(H, p="fro", dim=(-1, -2))
        self.register_parameter("H", torch.nn.Parameter(H))

        self.relu = RELUTwosided(
            self.num_conv, self.lam, self.L, self.sigma, self.device
        )

    def get_param(self, name):
        return self.state_dict(keep_vars=True)[name]

    def normalize(self):
        self.get_param("H").data = F.normalize(
            self.get_param("H").data, p="fro", dim=(-1, -2)
        )

    def zero_mean(self):
        self.get_param("H").data -= torch.mean(self.get_param("H").data, dim=0)

    def split_image(self, x):
        if self.stride == 1:
            return x, torch.ones_like(x)
        left_pad, right_pad, top_pad, bot_pad = utils.calc_pad_sizes(
            x, self.dictionary_dim, self.stride
        )
        x_batched_padded = torch.zeros(
            x.shape[0],
            self.stride ** 2,
            x.shape[1],
            top_pad + x.shape[2] + bot_pad,
            left_pad + x.shape[3] + right_pad,
            device=self.device,
        ).type_as(x)
        valids_batched = torch.zeros_like(x_batched_padded)
        for num, (row_shift, col_shift) in enumerate(
            [(i, j) for i in range(self.stride) for j in range(self.stride)]
        ):
            x_padded = F.pad(
                x,
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="reflect",
            )
            valids = F.pad(
                torch.ones_like(I),
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="constant",
            )
            x_batched_padded[:, num, :, :, :] = x_padded
            valids_batched[:, num, :, :, :] = valids
        x_batched_padded = x_batched_padded.reshape(-1, *x_batched_padded.shape[2:])
        valids_batched = valids_batched.reshape(-1, *valids_batched.shape[2:])
        return x_batched_padded, valids_batched

    def forward(self, x):
        x_batched_padded, valids_batched = self.split_image(x)

        num_batches = x_batched_padded.shape[0]

        D_enc1 = F.conv2d(
            x_batched_padded, self.get_param("H"), stride=self.stride
        ).shape[2]
        D_enc2 = F.conv2d(
            x_batched_padded, self.get_param("H"), stride=self.stride
        ).shape[3]

        x_old = torch.zeros(
            num_batches, self.num_conv, D_enc1, D_enc2, device=self.device
        ).type_as(x_batched_padded)
        yk = torch.zeros(
            num_batches, self.num_conv, D_enc1, D_enc2, device=self.device
        ).type_as(x_batched_padded)
        x_new = torch.zeros(
            num_batches, self.num_conv, D_enc1, D_enc2, device=self.device
        ).type_as(x_batched_padded)

        del D_enc1
        del D_enc2
        del num_batches

        t_old = torch.tensor(1, device=self.device).float()

        for t in range(self.T):
            Hyk = F.conv_transpose2d(yk, self.get_param("H"), stride=self.stride)
            x_tilda = x_batched_padded - Hyk

            x_new = (
                yk + F.conv2d(x_tilda, self.get_param("H"), stride=self.stride) / self.L
            )

            x_new = self.relu(x_new)

            t_new = (1 + torch.sqrt(1 + 4 * t_old * t_old)) / 2
            yk = x_new + ((t_old - 1) / t_new) * (x_new - x_old)

            x_old = x_new
            t_old = t_new

        z = (
            torch.masked_select(
                F.conv_transpose2d(x_new, self.get_param("H"), stride=self.stride),
                valids_batched.byte(),
            ).reshape(x.shape[0], self.stride ** 2, *x.shape[1:])
        ).mean(dim=1, keepdim=False)

        return z, x_new, self.relu.lam
