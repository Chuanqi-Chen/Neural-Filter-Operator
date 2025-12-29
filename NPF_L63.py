# Lorenz 63 System
# Observations: x
# States: [x, y, z]

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
u = u[::100] # dt_obs = 0.1

# Measurement Noise
u = u + 1.*np.random.randn(*u.shape)

# Train/Test & State/Observation
Ntrain = 8000
Ntest = 2000
train_x = torch.tensor(u[:Ntrain], dtype=torch.float)
test_x = torch.tensor(u[-Ntest:], dtype=torch.float)
train_y = torch.tensor(u[:Ntrain, :1], dtype=torch.float)
test_y = torch.tensor(u[-Ntest:, :1], dtype=torch.float)


####################################################
################# Model Architecture ###############
####################################################

class NPF(nn.Module):
    def __init__(self, input_size, hidden_size, state_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.state_size = state_size
        self.output_size = int(state_size + state_size*(state_size+1) / 2)

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SiLU(),
                                 nn.Linear(hidden_size, hidden_size), nn.SiLU(),
                                 nn.Linear(hidden_size, self.output_size))

    def forward(self, y, p0=None):
        # y: (N, T, S); p0: (Num_Layers, N, hidden_size)
        if p0 is None:
            p0 = torch.zeros(self.num_layers, y.shape[0], self.hidden_size, device=y.device)
            p0[..., 1] = 1.
        p, _ = self.rnn(y, p0) # out: (N, T, hidden_size)
        m = self.mlp(p) # (N, T, output_size)
        mean = m[..., :self.state_size]
        cov = self.vec_to_cov(m[..., self.state_size:])
        return [mean, cov]

    def vec_to_cov(self, v):
        d = self.state_size
        L = torch.zeros(*v.shape[:-1], d, d, device=v.device, dtype=v.dtype)
        rows, cols = torch.tril_indices(d, d, device=v.device)
        L[..., rows, cols] = v

        diag_mask = torch.eye(d, device=v.device, dtype=torch.bool)
        L[..., diag_mask] = nnF.softplus(L[..., diag_mask])
        cov = L @ L.mT
        return cov


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

