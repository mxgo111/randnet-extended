# currently this is a script
# can just put hyperparameters in the beginning


import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import model

from utils2 import generate_sparse_samples
from params import *

#@title Generate and Plot Sparse Samples { run: "auto", display-mode: "form" }
# Generate Sparse Samples
x_torch = generate_sparse_samples(num_examples, num_filters, x_dim, sparsity, device=device, seed=seed)
x_np = x_torch.cpu().detach().numpy()
print("x_torch.size() = " + str(x_torch.size()))
x_axis = np.arange(x_dim)

#@title Example Convolutional Filters { run: "auto", display-mode: "form" }
# Defines Example Convolutional Filters
upper = int(np.ceil(dict_dim/2))
lower = int(np.floor(dict_dim/2))
mid = dict_dim/2

# First example convolutional filter
h1 = np.append(np.ones(upper), -np.ones(lower))
h1 /= np.linalg.norm(h1)
# plt.plot(np.arange(h1.size), h1)
# plt.show()

# Second example convolutional filter
h2 = np.abs(np.arange(-lower, upper)) - mid/2
h2 /= np.linalg.norm(h2)
# plt.plot(np.arange(h2.size), h2)
# plt.show()

# Converts everything to right form/shape
h1_torch = torch.from_numpy(h1)
h2_torch = torch.from_numpy(h2)

H_torch = torch.zeros(num_filters, dict_dim)
H_torch[0, :] = h1_torch
H_torch[1, :] = h2_torch

x = torch.zeros(minibatch, num_filters, x_torch.size()[2])
weights = torch.zeros(num_filters, 1, dict_dim)

for i in range(num_filters):
  for j in range(num_examples):
    x[j,i,:] = x_torch[j][i]
  weights[i,0,:] = H_torch[i]

#@title Generate and Plot Data from Sparse Samples { run: "auto", display-mode: "form" }
# Generates y via y = Hx
y = F.conv_transpose1d(x, weights)

# for 2-d image shape should be [num_examples, num_channels=1 (greyscale), ydim, ydim]

print(y.size())

#@title Initial H { run: "auto" }
H_init = weights.clone() + torch.randn(weights.size()) * 0.1
# H_init = None

#@title Run CRsAE { run: "auto", display-mode: "form" }
def train_ae(
  net,
  hyp,
  criterion,
  optimizer,
  scheduler,
  test_loader=None,
  epoch_start=epoch_start,
  epoch_end=epoch_end,
):
  loss_all = []
  for idx, epoch in tqdm(enumerate(range(epoch_start, epoch_end))):
    # for i in range(y.size()[0]):
    optimizer.zero_grad()
    y_hat = net.forward(y)[0] # change this
    loss = criterion(y_hat, y)
    loss.backward()
    optimizer.step()
    scheduler.step()
    loss_all.append(loss.item())
    if idx % info_period == 0:
      print("loss:{:.8f} ".format(loss.item()))
    torch.cuda.empty_cache()
  return net, loss_all

#@title Run BackProp Function
net = model.CRsAE1D(hyp, H_init)

criterion = torch.nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=hyp["lr"], eps=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=hyp["lr_step"], gamma=hyp["lr_decay"])

info_period = 1

H = (net.get_param("H")).clone()
net, loss_all = train_ae(net, hyp, criterion, optimizer, scheduler)
H_new = net.get_param("H")
z_1, x_new_1 = net.forward(y)[:2]

#@title Plot y, y_hat { run: "auto", display-mode: "form" }

for i in range(2):
  print("example " + str(i + 1) + ":")
  plt.plot(np.arange(y.size()[-1]), y.detach().numpy()[i][0])
  plt.plot(np.arange(z_1.size()[-1]), z_1.detach().numpy()[i][0])
  plt.show()

#@title Plot x, x_hat { run: "auto", display-mode: "form" }
# print("x")
fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.scatter(x_axis, x_np[0][0])
ax1.scatter(x_axis, x_new_1.detach().numpy()[0][0])
ax2.scatter(x_axis, x_np[0][1])
ax2.scatter(x_axis, x_new_1.detach().numpy()[0][1])
plt.show()

plt.plot(np.arange(len(loss_all)), np.asarray(loss_all))
plt.title("Loss while training")
plt.show()

#@title Print Generative, Original, and Final Filters { run: "auto", display-mode: "form" }
fig, ax = plt.subplots(2, 3)

for i in range(num_filters):
  ax[i][0].plot(np.arange(dict_dim), H_torch[i])
ax[0][0].set_title("Generative filters")

for i in range(num_filters):
  ax[i][1].plot(np.arange(dict_dim), H.detach().numpy()[i,0,:])
ax[0][1].set_title("Initial filters")

for i in range(num_filters):
  ax[i][2].plot(np.arange(dict_dim), H_new.detach().numpy()[i,0,:])
ax[0][2].set_title("Learned filters")
plt.show()
