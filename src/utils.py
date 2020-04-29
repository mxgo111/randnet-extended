"""
Copyright (c) 2019 CRISP

utils

:author: Bahareh Tolooshams
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class LambdaLoss2D(torch.nn.Module):
    def __init__(self):
        super(LambdaLoss2D, self).__init__()

    def forward(self, target, hyp):
        enc_hat = target[0]
        lam = target[1]

        num_conv = hyp["num_conv"]
        delta = hyp["delta"]
        lam_init = hyp["lam"]

        r = delta * lam_init

        Ne = enc_hat.shape[-1] * enc_hat.shape[-2]

        lam_enc_l1 = torch.mean(torch.sum(torch.abs(lam * enc_hat), dim=(-1, -2, -3)))
        return (
            lam_enc_l1
            + torch.sum(lam * delta)
            - (Ne + r - 1) * torch.sum(torch.log(lam))
        )


def normalize1d(x):
    return F.normalize(x, dim=-1)


def normalize2d(x):
    return F.normalize(x, dim=(-1, -2))


def err1d_H(H, H_hat):
    H = H.detach().clone().cpu().numpy()
    H_hat = H_hat.detach().clone().cpu().numpy()

    H /= np.linalg.norm(H, axis=-1, keepdims=True)
    H_hat /= np.linalg.norm(H_hat, axis=-1, keepdims=True)

    num_conv = H.shape[0]

    err = []
    for conv in range(num_conv):
        corr = np.sum(H[conv, 0, :] * H_hat[conv, 0, :])
        err.append(np.sqrt(1 - corr ** 2))
    return err


def err2d_H(H, H_hat):
    H = H.detach().cpu().numpy()
    H_hat = H_hat.detach().cpu().numpy()

    num_conv = H.shape[0]

    err = []

    for conv in range(num_conv):
        corr = np.sum(H[conv, 0, :, :] * H_hat[conv, 0, :, :])
        err.append(np.sqrt(1 - corr ** 2))
    return err


def PSNR(x, x_hat):
    mse = np.mean((x - x_hat) ** 2)
    max_x = np.max(x)
    return 20 * np.log10(max_x) - 10 * np.log10(mse)


def calc_pad_sizes(y, dictionary_dim=8, stride=1):
    right_pad = (stride - ((x.shape[3] - dictionary_dim) % stride)) % stride + stride
    bot_pad = (stride - ((x.shape[2] - dictionary_dim) % stride)) % stride + stride
    return stride, right_pad, stride, bot_pad

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.detach().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def find_maximum(img):
    batch_size = img.shape[0]
    ans = np.zeros(batch_size)
    for k in range(batch_size):
        for i in range(img.shape[-2]):
            for j in range(img.shape[-1]):
                if img[k,0,i,j] > ans[k]:
                    ans[k] = img[k,0,i,j]
    return ans
