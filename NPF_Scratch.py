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
Lt = 5000
dt = 0.001
Nt = int(Lt/dt)
t = np.linspace(0, Lt, Nt)
u = np.zeros((Nt, I))
for n in range(Nt-1):
    if n % (1/dt) == 0:
        print(int(n*dt))
    for i in range(I):
        u_dot = -u[n, i] + u[n,(i+1)%I]*u[n,i-1] - u[n,i-2]*u[n,i-1] + F
        u[n+1, i] = u[n, i] + u_dot*dt + sigma*np.sqrt(dt)*np.random.randn()
u = u[::50] # dt_obs = 0.1


# Measurement Noise
sigma_obs = 1.
u = u + sigma_obs*np.random.randn(*u.shape)
# u = np.load("./data/L96_data.npy")

# Train/Test & State/Observation
Ntrain = int(len(u)*0.8)
Ntest = int(len(u)*0.2)
train_x = torch.tensor(u[:Ntrain], dtype=torch.float)
test_x = torch.tensor(u[-Ntest:], dtype=torch.float)
train_y = torch.tensor(u[:Ntrain, ::2], dtype=torch.float)
test_y = torch.tensor(u[-Ntest:, ::2], dtype=torch.float)

####################################################
################# Model Architecture ###############
####################################################

class NPF(nn.Module):
    def __init__(self, state_size, obs_size, hidden_size, num_layers=1, diag_cov=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.state_size = state_size
        self.diag_cov = diag_cov
        self.mean_output_size = state_size
        if diag_cov:
            self.cov_output_size = state_size * 2
        else:
            self.cov_output_size = int(state_size + state_size*(state_size+1) / 2)

        self.rnn = nn.RNN(obs_size, hidden_size, num_layers, batch_first=True)
        self.mlp_mean = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SiLU(),
                                      nn.Linear(hidden_size, hidden_size), nn.SiLU(),
                                      nn.Linear(hidden_size, hidden_size), nn.SiLU(),
                                      nn.Linear(hidden_size, hidden_size), nn.SiLU(),
                                      nn.Linear(hidden_size, hidden_size), nn.SiLU(),
                                      nn.Linear(hidden_size, hidden_size), nn.SiLU(),
                                      nn.Linear(hidden_size, self.mean_output_size))

        self.mlp_cov = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SiLU(),
                                     nn.Linear(hidden_size, hidden_size), nn.SiLU(),
                                     nn.Linear(hidden_size, hidden_size), nn.SiLU(),
                                     nn.Linear(hidden_size, hidden_size), nn.SiLU(),
                                     nn.Linear(hidden_size, hidden_size), nn.SiLU(),
                                     nn.Linear(hidden_size, hidden_size), nn.SiLU(),
                                     nn.Linear(hidden_size, self.cov_output_size))

    def forward(self, y, p0=None):
        # y: (N, T, S); p0: (Num_Layers, N, hidden_size)
        if p0 is None:
            p0 = torch.zeros(self.num_layers, y.shape[0], self.hidden_size, device=y.device)
            p0[..., 1] = 1.
        p, _ = self.rnn(y, p0) # out: (N, T, hidden_size)
        mu = self.mlp_mean(p)
        cov_root = self.mlp_cov(p)
        if self.diag_cov:
            cov = cov_root[..., self.state_size:]**2
        else:
            cov = self.vec_to_cov(cov_root[..., self.state_size:])
        x_est = {"mean": mu, "cov": cov}
        return x_est

    def vec_to_cov(self, v):
        d = self.state_size
        L = torch.zeros(*v.shape[:-1], d, d, device=v.device, dtype=v.dtype)
        rows, cols = torch.tril_indices(d, d, device=v.device)
        L[..., rows, cols] = v
        cov = L @ L.mT
        return cov


##################################################
################# Model Training #################
##################################################

# Stage 1: Train NPF with MSE
batch_size = 10
batch_step = 100
Niterations = 5000
train_mse_history = []

npf = NPF(state_size=40, obs_size=20, hidden_size=256, num_layers=1, diag_cov=True)
optimizer = torch.optim.Adam(npf.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Niterations)
for niter in range(Niterations):
    indices = np.random.choice(Ntrain-batch_step, size=batch_size, replace=False)
    batch_x = torch.stack([train_x[idx:idx+batch_step] for idx in indices])
    batch_y = torch.stack([train_y[idx:idx+batch_step] for idx in indices])
    batch_x_est = npf(batch_y)
    loss = nnF.mse_loss(batch_x, batch_x_est["mean"])
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(npf.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    train_mse_history.append(loss.item())
    print(f"niter {niter} | MSE {loss.item():.4f}")


# Stage 2: Train NPF with NLL
def NLL(x, est, eps=1e-6):
    # x: (N, T, S)
    # est: {(N, T, S), (N, T, S) or (N, T, S, S)}
    mu = est["mean"]
    if est["cov"].dim() == 3:
        var = est["cov"]
        var = nnF.softplus(var) + eps
        nll = ( (x-mu)**2/var + torch.log(var) ).sum(dim=-1)
    else:
        x = x.unsqueeze(-1)
        mu = mu.unsqueeze(-1)
        cov = est["cov"]
        d = cov.diagonal(dim1=-2, dim2=-1)
        target_d = nnF.softplus(d) + eps
        cov = cov + torch.diag_embed(target_d - d, dim1=-2, dim2=-1)
        wmse = ((x-mu).mT @ torch.inverse(cov) @  (x-mu)).squeeze((-1, -2))
        nll = wmse + torch.log(torch.det(cov))
    return nll.mean()

batch_size = 10
batch_step = 100
Niterations = 5000
train_nll_history = []

npf.rnn.requires_grad_(False)
npf.mlp_mean.requires_grad_(False)
optimizer = torch.optim.Adam(npf.mlp_cov.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Niterations)
for niter in range(Niterations):
    indices = np.random.choice(Ntrain-batch_step, size=batch_size, replace=False)
    batch_x = torch.stack([train_x[idx:idx+batch_step] for idx in indices])
    batch_y = torch.stack([train_y[idx:idx+batch_step] for idx in indices])
    batch_x_est = npf(batch_y)
    loss = NLL(batch_x, batch_x_est)

    loss.backward()
    # torch.nn.utils.clip_grad_norm_(npf.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    train_nll_history.append(loss.item())
    print(f"niter {niter} | NLL {loss.item():.4f}")


###################################################
################# Model Inference #################
###################################################

with torch.no_grad():
    train_x_est = npf(train_y.unsqueeze(0))

# MSE
nnF.mse_loss(train_x.unsqueeze(0), train_x_est["mean"])
# NLL
NLL(train_x, train_x_est)


si = 800
ei = 1000
t = np.arange(si, ei)

fig = plt.figure()
axs = fig.subplots(3, 1)
x1_true = train_x[si:ei, 0]
x1_mean = train_x_est["mean"][0, si:ei, 0]
x1_std = torch.abs(train_x_est["cov"][0, si:ei, 0])**0.5
axs[0].plot(t, x1_true, color="black")
axs[0].plot(t, x1_mean, color="red")
axs[0].fill_between(t, x1_mean+2*x1_std, x1_mean-2*x1_std, color="red", alpha=0.2, label="95% Conf")

x2_true = train_x[si:ei, 1]
x2_mean = train_x_est["mean"][0, si:ei, 1]
x2_std = torch.abs(train_x_est["cov"][0, si:ei, 1])**0.5
axs[1].plot(t, x2_true, color="black")
axs[1].plot(t, x2_mean, color="red")
axs[1].fill_between(t, x2_mean+2*x2_std, x2_mean-2*x2_std, color="red", alpha=0.2, label="95% Conf")

x3_true = train_x[si:ei, 2]
x3_mean = train_x_est["mean"][0, si:ei, 2]
x3_std = torch.abs(train_x_est["cov"][0, si:ei, 2])**0.5
axs[2].plot(t, x3_true, color="black")
axs[2].plot(t, x3_mean, color="red")
axs[2].fill_between(t, x3_mean+2*x3_std, x3_mean-2*x3_std, color="red", alpha=0.2, label="95% Conf")
fig.tight_layout()

