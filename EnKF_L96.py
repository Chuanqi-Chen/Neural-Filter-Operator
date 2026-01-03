# Lorenz 96 System
# System States: [x1, x2, ...x40]
# Observations: Noisy [x2, x4, .., x38]

import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


mpl.use("Qt5Agg")
plt.rc("text", usetex=True)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

# device = "cpu"
torch.manual_seed(0)
np.random.seed(0)

####################################################
################# Data Preparation #################
####################################################

F = 8
sigma = 0.5

I = 40
Lt = 500
dt = 0.001
Nt = int(Lt/dt)
t = np.linspace(0, Lt, Nt)
u = np.zeros((Nt, I))
# for n in range(Nt-1):
#     if n % (1/dt) == 0:
#         print(int(n*dt))
#     for i in range(I):
#         u_dot = -u[n, i] + u[n,(i+1)%I]*u[n,i-1] - u[n,i-2]*u[n,i-1] + F
#         u[n+1, i] = u[n, i] + u_dot*dt + sigma*np.sqrt(dt)*np.random.randn()
# u = u[::50]
dt_obs = 0.05

# Measurement Noise
# u = u + 1.*np.random.randn(*u.shape)
u = np.load("./data/L96_data.npy")

# Train/Test & State/Observation
Ntrain = 8000
Ntest = 2000
train_x = torch.tensor(u[:Ntrain], dtype=torch.float)
test_x = torch.tensor(u[-Ntest:], dtype=torch.float)
train_y = torch.tensor(u[:Ntrain, ::2], dtype=torch.float)
test_y = torch.tensor(u[-Ntest:, ::2], dtype=torch.float)

##########################################################
################# Ensemble Kalman Filter #################
##########################################################

def cross_cov(X, Y):
    n = X.shape[0]
    assert n == Y.shape[0]
    X_centered = X - torch.mean(X, dim=0)
    Y_centered = Y - torch.mean(Y, dim=0)
    cross_cov_matrix = X_centered.T @ Y_centered / (n - 1)
    return cross_cov_matrix

J = 10000
x_ens = torch.randn(J, Ntest+1, I)
x_ens_prior = torch.zeros(J, I)
for nt in range(Ntest):
    print(nt)
    # Forecast
    for i in range(I):
        x_ens_dot = -x_ens[:, nt, i] + x_ens[:, nt,(i+1)%I]*x_ens[:, nt, i-1] - x_ens[:, nt, i-2]*x_ens[:, nt,i-1] + F
        x_ens_prior[:, i] = x_ens[:, nt, i] + x_ens_dot*dt_obs + sigma*dt_obs**0.5*torch.randn_like(x_ens[:, nt+1, i])
    y_ens_prior = x_ens_prior[..., ::2] + 1.*torch.randn_like(x_ens_prior[..., ::2])
    # Analysis
    cov_yy = torch.cov(y_ens_prior.T)
    ccov_xy = cross_cov(x_ens_prior, y_ens_prior)
    K = ccov_xy @ torch.inverse(cov_yy)
    x_ens[:, nt+1] = x_ens_prior + (test_y[nt] - y_ens_prior)@K.T
x_ens = x_ens[:, 1:]

x_ens_mean = x_ens.mean(dim=0)
nnF.mse_loss(test_x, x_ens_mean)


si = 800
ei = 1000
t = np.arange(si, ei)

fig = plt.figure()
axs = fig.subplots(3, 1)
x1_true = test_x[si:ei, 0]
x1_mean = x_ens_mean[si:ei, 0]
# x1_std = torch.abs(test_x_est["cov"][0, si:ei, 0])**0.5
axs[0].plot(t, x1_true, color="black")
axs[0].plot(t, x1_mean, color="red")
# axs[0].fill_between(t, x1_mean+2*x1_std, x1_mean-2*x1_std, color="red", alpha=0.2, label="95% Conf")

x2_true = test_x[si:ei, 1]
x2_mean = x_ens_mean[si:ei, 1]
# x2_std = torch.abs(test_x_est["cov"][0, si:ei, 1])**0.5
axs[1].plot(t, x2_true, color="black")
axs[1].plot(t, x2_mean, color="red")
# axs[1].fill_between(t, x2_mean+2*x2_std, x2_mean-2*x2_std, color="red", alpha=0.2, label="95% Conf")

x3_true = test_x[si:ei, 2]
x3_mean = x_ens_mean[si:ei, 2]
# x3_std = torch.abs(test_x_est["cov"][0, si:ei, 2])**0.5
axs[2].plot(t, x3_true, color="black")
axs[2].plot(t, x3_mean, color="red")
# axs[2].fill_between(t, x3_mean+2*x3_std, x3_mean-2*x3_std, color="red", alpha=0.2, label="95% Conf")
fig.tight_layout()
