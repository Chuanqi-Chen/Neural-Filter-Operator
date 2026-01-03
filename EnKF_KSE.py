# K-S Equation
# System States: 128 u
# Observations: Noisy 8 u

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
sigma_obs = 0.2
u = np.load(r"./data/KSE_Data(Noisy).npy")


# Train/Test & State/Observation
Nt_obs, state_size = u.shape
Ntrain = int(Nt_obs*0.8)
Ntest = int(Nt_obs*0.2)
train_x = torch.tensor(u[:Ntrain], dtype=torch.float)
test_x = torch.tensor(u[-Ntest:], dtype=torch.float)
train_y = torch.tensor(u[:Ntrain, ::16], dtype=torch.float)
test_y = torch.tensor(u[-Ntest:, ::16], dtype=torch.float)



##########################################################
################# Ensemble Kalman Filter #################
##########################################################

train_x = train_x.numpy()
train_y = train_y.numpy()
test_x = test_x.numpy()
test_y = test_y.numpy()

def cross_cov(X, Y):
    n = X.shape[0]
    assert n == Y.shape[0]
    X_centered = X - np.mean(X, axis=0)
    Y_centered = Y - np.mean(Y, axis=0)
    cross_cov_matrix = X_centered.T @ Y_centered / (n - 1)
    return cross_cov_matrix

def KSE_forecast(u, dt_obs=1.):
    # u: (N, C)
    J, Nx = u.shape
    Lx = 22
    dt = 0.01
    Nt = int(dt_obs/dt)

    kx = np.hstack([np.arange(0,Nx/2,1), np.array([0]), np.arange(-Nx/2+1,0,1)])  # integer wavenumbers: exp(2*pi*i*kx*x/L)
    alpha = 2.*np.pi*kx/Lx              # real wavenumbers:    exp(i*alpha*x)
    D = 1j*alpha                   # D = d/dx operator in Fourier space
    L = alpha**2 - alpha**4        # linear operator -D^2 - D^4 in Fourier space
    G = -0.5*D                      # -1/2 D operator in Fourier space
    A =  np.ones(Nx) + dt/2*L
    B = 1/(np.ones(Nx) - dt/2*L)

    Nn  = sp.fftpack.fft(u**2)*G  # -1/2 d/dx(u^2) = -u u_x, collocation calculation
    u = sp.fftpack.fft(u) # transform data to spectral coeffs
    for n in range(Nt):
        Nn1 = Nn     # shift N^{n-1} <- N^n
        Nn  = G * sp.fftpack.fft(np.real(sp.fftpack.ifft(u)) ** 2) # compute N^n = -u u_x
        u = B * (A * u + 3/2*dt * Nn - dt/2 * Nn1) # CNAB2 formula
    u = np.real(sp.fftpack.ifft(u))
    return u


def gaspari_cohn(r):
    """
    Gaspari–Cohn localization (compactly supported correlation).
    Input r is normalized distance: r = d / loc_radius (dimensionless).
    Output rho in [0,1], with rho(r)=0 for r>=2.
    """
    r = np.abs(r).astype(float)
    rho = np.zeros_like(r)

    m1 = (r <= 1.0)
    m2 = (r > 1.0) & (r <= 2.0)

    r1 = r[m1]
    r2 = r[m2]

    # 0 <= r <= 1
    rho[m1] = (
        1.0
        - (5.0/3.0) * r1**2
        + (5.0/8.0) * r1**3
        + 0.5 * r1**4
        - 0.25 * r1**5
    )

    # 1 < r <= 2  (CORRECT branch)
    rho[m2] = (
        4.0
        - 5.0 * r2
        + (5.0/3.0) * r2**2
        + (5.0/8.0) * r2**3
        - 0.5 * r2**4
        + (1.0/12.0) * r2**5
        - 2.0 / (3.0 * r2)
    )

    # r > 2 already zero
    return rho

def periodic_dist(x, y, L):
    d = np.abs(x - y)
    return np.minimum(d, L - d)

def gc_rho_state_obs(x_state, x_obs, loc_radius, L):
    # rho_xy: (n_state, n_obs)
    d = periodic_dist(x_state[:, None], x_obs[None, :], L)
    return gaspari_cohn(d / loc_radius)

# Domain
L = 22.0

# State grid
n_state = 128
dx = L / n_state
x_state = np.arange(n_state) * dx

# Observations (8 evenly spaced grid points)
n_obs = 8
obs_idx = np.arange(0, n_state, n_state // n_obs)   # [0,16,32,48,64,80,96,112]
x_obs = x_state[obs_idx]

# Localization radius (physical units)
loc_radius = 1.375

rho_xy = gc_rho_state_obs(
    x_state=x_state,
    x_obs=x_obs,
    loc_radius=loc_radius,
    L=L
)



J = 100
x_ens = np.random.randn(J, Ntest+1, state_size)
init_indices = np.random.choice(Ntrain, J, replace=False)
x_ens[:, 0] = train_x[init_indices]
x_ens_prior = np.zeros((J, state_size))
for nt in range(Ntest):
    print(nt)
    # Forecast
    x_ens_prior = KSE_forecast(x_ens[:, nt])
    y_ens_prior = x_ens_prior[..., ::16] + sigma_obs*np.random.randn(*x_ens_prior[:, ::16].shape)
    # Analysis
    cov_yy = np.cov(y_ens_prior.T)
    ccov_xy = rho_xy * cross_cov(x_ens_prior, y_ens_prior)
    K = ccov_xy @ np.linalg.inv(cov_yy)
    x_ens[:, nt+1] = x_ens_prior + (test_y[nt] - y_ens_prior)@K.T

x_ens = x_ens[:, 1:]

x_ens_mean = x_ens.mean(axis=0)
np.mean((test_x - x_ens_mean)**2)


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
