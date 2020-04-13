import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np

import gc
import r_utils

def train_ae(
    net,
    data_loader,
    hyp,
    criterion,
    optimizer,
    scheduler,
    PATH="",
    test_loader=None,
    epoch_start=0,
    epoch_end=1,
):

    info_period = hyp["info_period"]
    # noiseSTD = hyp["noiseSTD"]
    device = hyp["device"]
    # zero_mean_filters = hyp["zero_mean_filters"]
    normalize = hyp["normalize"]
    supervised = hyp["supervised"]

    if normalize:
        net.normalize()

    for epoch in tqdm(range(epoch_start, epoch_end)):
        scheduler.step()
        loss_all = 0
        for idx, (img, _) in tqdm(enumerate(data_loader)):
            optimizer.zero_grad()

            img = img.to(device)
            # noise = noiseSTD / 255 * torch.randn(img.shape).to(device)
            r_hat, img_hat, code, lam = net(img)
            loss = criterion(img, img_hat)

            loss_all += float(loss.item())
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if normalize:
                net.normalize()

            if idx % info_period == 0:
                print("loss:{:.8f} ".format(loss.item()))

            torch.cuda.empty_cache()

        # ===================log========================

        torch.save(loss_all, os.path.join(PATH, "loss_epoch{}.pt".format(epoch)))

        torch.save(net, os.path.join(PATH, "model_epoch{}.pt".format(epoch)))

        print(
            "epoch [{}/{}], loss:{:.8f} ".format(
                epoch + 1, hyp["num_epochs"], loss.item()
            )
        )

    return net
