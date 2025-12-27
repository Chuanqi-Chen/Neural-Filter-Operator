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

# X, Y ---> Z

# Simulation Settings: Lt=1000, dt=0.001
sigma, rho, beta = 10, 28, 8/3
Lt = 1000
dt = 0.001
Nt = int(Lt/dt)
u = np.zeros((Nt, 3))
for n in range(Nt-1):
    if n % (1/dt) == 0:
        print(n*dt)
    u[n+1, 0] = u[n, 0] + (sigma*(u[n, 1] - u[n, 0]))*dt
    u[n+1, 1] = u[n, 1] + (u[n, 0]*(rho-u[n, 2])-u[n, 1])*dt
    u[n+1, 2] = u[n, 2] + (u[n, 0]*u[n, 1] - beta*u[n, 2])*dt
    u[n+1, :] = u[n+1, :] + dt**0.5*np.random.randn(*u[n+1, :].shape)

u = u[::100]
u = u + 1.*np.random.randn(*u.shape)

Ntrain = 8000
Ntest = 2000
train_u = torch.tensor(u[:Ntrain], dtype=torch.float)
test_u = torch.tensor(u[-Ntest:], dtype=torch.float)


####################################################
################# Model Architecture ###############
####################################################

class NFO(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(4, 16), nn.SiLU(),
                                 nn.Linear(16, 32), nn.SiLU(),
                                 nn.Linear(32, 16), nn.SiLU(),
                                 nn.Linear(16, 2))

    def forward(self, obs, est):
        cat = torch.cat([est, obs], dim=-1)
        est = self.mlp(cat)
        return est

    def filter(self, obs, est0=None):
        # obs: (N, T, C); est0: (N, C)
        N, Nt = obs.shape[:2]
        if est0 is None:
            est0 = torch.zeros(N, 2)
            est0[:, 1] = 1.
        est = torch.zeros(N, Nt+1, 2)
        est[:, 0] = est0
        for nt in range(Nt):
            est[:, nt+1] = self(est[:, nt], obs[:, nt])
        return est

##################################################
################# Model Training #################
##################################################

def NLL(z, est, eps=1e-6):
    # z shape (N, T, S), est shape (N, T, [mean, cov])
    mu = est[..., 0:1]
    s  = est[..., 1:2]
    var = s**2 + eps
    nll = (z-mu)**2/var + torch.log(var)
    return nll.mean()


batch_size = 10
batch_step = 100
Niterations = 5000
train_loss_history = []
warmup_iter = 1000

nfo = NFO()
optimizer = torch.optim.Adam(nfo.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Niterations)
for niter in range(Niterations):
    indices = np.random.choice(Ntrain-batch_step, size=batch_size, replace=False)
    batch_obs = torch.stack([train_u[idx:idx+batch_step] for idx in indices])
    batch_est = nfo.filter(batch_obs[..., :2])
    batch_est = batch_est[:, 1:]  # Cut off the initial point
    if niter < warmup_iter:
        loss = nnF.mse_loss(batch_obs[..., 2:], batch_est[..., :1])
    else:
        loss = NLL(batch_obs[..., 2:], batch_est)
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(nfo.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    train_loss_history.append(loss.item())
    print(f"niter {niter} | loss {loss.item():.4f}")



###################################################
################# Model Inference #################
###################################################

sample_u = test_u[1000:1100].unsqueeze(0)
sample_z = sample_u[..., 2:]
with torch.no_grad():
    sample_est = nfo.filter(sample_u[..., :2])

nnF.mse_loss(sample_z, sample_est[:, 1:, :1])
sample_est[:, 1:, 1:]

fig = plt.figure()
ax = fig.subplots()
ax.plot(np.arange(1, 101), sample_z.flatten())
ax.plot(np.arange(101), sample_est[..., :1].flatten())
ax.fill_between(np.arange(0, 101), sample_est[..., :1].flatten() - 2*torch.abs(sample_est[..., 1:].flatten()), sample_est[..., :1].flatten() + 2*torch.abs(sample_est[..., 1:].flatten()), alpha=0.2)

