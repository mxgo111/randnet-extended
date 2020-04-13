"""
Copyright (c) 2019 CRISP

crsae model

:author: Bahareh Tolooshams
"""

import torch
import torch.nn.functional as F
import numpy as np

import r_utils

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

class RandNet2(torch.nn.Module):
    def __init__(self, hyp, H=None, Phi=None):
        super(RandNet2, self).__init__()

        self.T = hyp["num_iters"]
        self.L = hyp["L"]
        self.num_conv = hyp["num_conv"]
        self.dictionary_dim = hyp["dictionary_dim"]
        self.device = hyp["device"]
        self.sigma = hyp["sigma"]
        self.stride = hyp["stride"]
        self.twosided = hyp["twosided"]
        self.use_lam = hyp["use_lam"]
        self.r_dim = hyp["r_dim"]
        self.y_dim = hyp["y_dim"]
        self.trainistrue = hyp["trainistrue"]

        if self.use_lam:
            self.lam = hyp["lam"]

        if H is None:
            H = torch.randn(
                (self.num_conv, 1, self.dictionary_dim, self.dictionary_dim),
                device=self.device,
            )
            H /= self.dictionary_dim
            H = F.normalize(H, p="fro", dim=(-1, -2))
        self.register_parameter("H", torch.nn.Parameter(H))

        if Phi is None:
            Phi = torch.randn(
                (self.r_dim, self.y_dim), device=self.device,
                #dimension of data to smaller dimension
            )
            Phi = F.normalize(Phi, dim=-1)

            # think about the normalization above later (TODO)
            # we want the norm of each column to be 1
            # y_dim is number of columns, r_dim is number of rows in Phi

        self.register_parameter("Phi", torch.nn.Parameter(Phi))

        self.relu = torch.nn.ReLU()

    def get_param(self, name):
        return self.state_dict(keep_vars=True)[name]

    def normalize(self):
        self.get_param("H").data = F.normalize(
            self.get_param("H").data, p="fro", dim=(-1, -2)
        )
        self.get_param("Phi").data = F.normalize(
            self.get_param("Phi").data, p="fro", dim=(-1, -2)
        )

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

    def forward(self, i):

        if self.trainistrue:
            # Note : pass in r later for testing
            y_batched_padded, valids_batched = self.split_image(i)

            num_batches = y_batched_padded.shape[0]

            D_enc1 = F.conv2d(
                y_batched_padded, self.get_param("H"), stride=self.stride
            ).shape[2]
            D_enc2 = F.conv2d(
                y_batched_padded, self.get_param("H"), stride=self.stride
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

            # optimization for the future: find HT y then multiply by 0

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
                # print("Hyk:", Hyk.shape)
                flatten = Hyk.view(-1,Hyk.shape[-2]* Hyk.shape[-1], 1) # TODO check this later
                # print("flatten:",flatten.shape)
                # print("Phi dimension:", self.get_param('Phi').shape)
                PhiHyk = torch.matmul(self.get_param("Phi"), flatten)

                # print("PhiHyk:", PhiHyk.shape)
                # This line finds H yk, yk is in the input
                # We want Phi H yk
                # add a line that finds Phi H yk by multiplication (recall H is a convolution, Phi is a matrix)

                flatten = y_batched_padded.view(-1, y_batched_padded.shape[-2]*y_batched_padded.shape[-1], 1)
                r_batched_padded = torch.matmul(self.get_param("Phi"), flatten)

                # print("r_batched_padded:", r_batched_padded.shape)

                x_tilda = r_batched_padded - PhiHyk

                # to implement Phi
                # wherever you have conv_transpose2d we need to change H to


                phi_t_x_tilda1d = torch.matmul(torch.t(self.get_param("Phi")), x_tilda)
                # Check the shape of this (TODO)
                # Insert more print statements to check sizes (TODO)

                phi_t_x_tilda2d = phi_t_x_tilda1d.view(-1, 1, Hyk.shape[-2], Hyk.shape[-1])

                x_new = (
                    yk + F.conv2d(phi_t_x_tilda2d, self.get_param("H"), stride=self.stride) / self.L
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

            Hx = F.conv_transpose2d(x_new, self.get_param("H"), stride=self.stride)
            y_hat = Hx
            flatten2 = Hx.view(-1, Hx.shape[-2]* Hx.shape[-1], 1) # TODO check this later
            PhiHx = torch.matmul(self.get_param("Phi"), flatten2)

            r_hat = PhiHx
            # (
            #     torch.masked_select( # look into this (TODO), check if dimensions match (TODO)
            #         PhiHx,
            #         valids_batched.byte(),
            #     ).reshape(y.shape[0], self.stride ** 2, *y.shape[1:])

            # if this line fails, set z = PhiHx and stride = 1

                # do the same thing for conv_transpose above
                # to get y_hat, need to get H x_new
                # to get r_hat, find phi H x_new
                # output r_hat, y_hat, and x_new, and (lambda)

                # challenge: reshaping the input to Phi, because the image is 2d but need 1d for matrix multiplication - how to turn back to 2d?

            # ).mean(dim=1, keepdim=False)

            return r_hat, y_hat, x_new, self.lam

        else: # Note : pass in r later for testing
            # y_batched_padded, valids_batched = self.split_image(i)
            r1d = i
            flat_phiTr = torch.t(self.get_param("Phi"), r1d)
            # assume the data is square
            phiTr = flat_phiTr.view(-1, 1, sqrt(self.y_dim), sqrt(self.y_dim))

            num_batches = phiTr.shape[0]

            D_enc1 = F.conv2d(
                phiTr, self.get_param("H"), stride=self.stride
            ).shape[2]
            D_enc2 = F.conv2d(
                phiTr, self.get_param("H"), stride=self.stride
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
                # print("Hyk:", Hyk.shape)
                flatten = Hyk.view(-1,Hyk.shape[-2]* Hyk.shape[-1], 1) # TODO check this later
                # print("flatten:",flatten.shape)
                # print("Phi dimension:", self.get_param('Phi').shape)
                PhiHyk = torch.matmul(self.get_param("Phi"), flatten)

                # print("PhiHyk:", PhiHyk.shape)
                # This line finds H yk, yk is in the input
                # We want Phi H yk
                # add a line that finds Phi H yk by multiplication (recall H is a convolution, Phi is a matrix)

                # flatten = y_batched_padded.view(-1, y_batched_padded.shape[-2]*y_batched_padded.shape[-1], 1)
                # r_batched_padded = torch.matmul(self.get_param("Phi"), flatten)

                # print("r_batched_padded:", r_batched_padded.shape)

                x_tilda = r1d - PhiHyk

                # to implement Phi
                # wherever you have conv_transpose2d we need to change H to


                phi_t_x_tilda1d = torch.matmul(torch.t(self.get_param("Phi")), x_tilda)
                # Check the shape of this (TODO)
                # Insert more print statements to check sizes (TODO)

                phi_t_x_tilda2d = phi_t_x_tilda1d.view(-1, 1, Hyk.shape[-2], Hyk.shape[-1])

                x_new = (
                    yk + F.conv2d(phi_t_x_tilda2d, self.get_param("H"), stride=self.stride) / self.L
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

            Hx = F.conv_transpose2d(x_new, self.get_param("H"), stride=self.stride)
            y_hat = Hx
            flatten2 = Hx.view(-1, Hx.shape[-2]* Hx.shape[-1], 1) # TODO check this later
            PhiHx = torch.matmul(self.get_param("Phi"), flatten2)

            r_hat = PhiHx
            # (
            #     torch.masked_select( # look into this (TODO), check if dimensions match (TODO)
            #         PhiHx,
            #         valids_batched.byte(),
            #     ).reshape(y.shape[0], self.stride ** 2, *y.shape[1:])

            # if this line fails, set z = PhiHx and stride = 1

                # do the same thing for conv_transpose above
                # to get y_hat, need to get H x_new
                # to get r_hat, find phi H x_new
                # output r_hat, y_hat, and x_new, and (lambda)

                # challenge: reshaping the input to Phi, because the image is 2d but need 1d for matrix multiplication - how to turn back to 2d?

            # ).mean(dim=1, keepdim=False)

            return r_hat, y_hat, x_new, self.lam
