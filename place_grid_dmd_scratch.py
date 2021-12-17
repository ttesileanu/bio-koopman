# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: grid-cell-dmd
#     language: python
#     name: grid-cell-dmd
# ---

# %% [markdown] id="PCVwyhynz309"
# # Experiments with dynamic mode decomposition (DMD) interpretation of place and grid cells

# %% id="JNWUwVFTz-zF"
# %load_ext autoreload
# %autoreload 2

import matplotlib.pyplot as plt
import seaborn as sns
import pydove as dv

import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm

from training import train, test
from utils import StepwiseScheduler
from bump_simulator import PlaceGridMotionSimulator
from naive_model import PlaceGridSystemNonBio
from naive_model_cplx import PlaceGridSystemNonBioCplx

# %% [markdown] id="n1dz8V341gMS" tags=[]
# ## Test dataset generation

# %% colab={"base_uri": "https://localhost:8080/", "height": 80} id="kstusppd5HNF" outputId="68856bcd-d61a-4182-e9fe-31cd54df580a"
torch.manual_seed(0)
simulator = PlaceGridMotionSimulator(32, sigma=2)

fig, ax = plt.subplots(constrained_layout=True)
ax.imshow(simulator()[None, :])

# %% colab={"base_uri": "https://localhost:8080/", "height": 313} id="E1oNEWnO5TRP" outputId="bccdc90d-10a6-4782-88b8-0720f2572a6a"
torch.manual_seed(0)
simulator = PlaceGridMotionSimulator(32, sigma=2)

crt_s = torch.normal(torch.zeros(30), 5.0)
crt_batch = simulator.batch(crt_s)

fig, (ax1, ax2) = plt.subplots(
    1, 2, sharey=True, constrained_layout=True, gridspec_kw={"width_ratios": (3, 1)}
)
ax1.imshow(crt_batch)
ax1.set_ylabel("time")
ax1.set_xlabel("position")

ax2.barh(width=crt_s, y=np.arange(len(crt_s)))
ax2.axvline(0, ls=":", c="gray")
crt_xl = max(ax2.get_xlim())
ax2.set_xlim(-crt_xl, crt_xl)
ax2.set_ylim(ax1.get_ylim())
ax2.invert_yaxis()
ax2.set_xlabel("shift")
sns.despine(ax=ax2, left=True)

# %% [markdown] id="N8SBrBPfPLrv"
# ## Generate a dataset with periodic boundary conditions

# %% id="BHxRVaEkMNyf"
torch.manual_seed(0)

n = 32
simulator = PlaceGridMotionSimulator(n, sigma=2)

n_samples = 150_000
s = torch.normal(torch.zeros(n_samples), 5.0)
trajectory = simulator.batch(s)

dataset_full = [(trajectory[i], trajectory[i + 1], s[i]) for i in range(n_samples - 1)]

test_size = 500
dataset_train = dataset_full[:-test_size]
dataset_test = dataset_full[-test_size:]

# %% colab={"base_uri": "https://localhost:8080/", "height": 313} id="9yehqta96sDy" outputId="1b286f8d-7eca-4a1f-fef3-f0fce4595ce7"
crt_n = 4 * n
crt_step = 1

fig, (ax1, ax2) = plt.subplots(
    1, 2, sharey=True, constrained_layout=True, gridspec_kw={"width_ratios": (3, 1)}
)
ax1.imshow(trajectory[:crt_n:crt_step])
ax1.set_ylabel("time")
ax1.set_xlabel("position")

ax2.barh(width=s[:crt_n:crt_step], y=np.arange(0, crt_n // crt_step))
ax2.axvline(0, ls=":", c="gray")
crt_xl = max(ax2.get_xlim())
ax2.set_xlim(-crt_xl, crt_xl)
ax2.set_ylim(ax1.get_ylim())
ax2.invert_yaxis()
ax2.set_xlabel("shift")
sns.despine(ax=ax2, left=True)

# %% colab={"base_uri": "https://localhost:8080/", "height": 288} id="1WkvdWxh7EHR" outputId="53e25e5f-4ebe-4101-ec58-37e69e6df687"
fig, ax = plt.subplots()
ax.plot(torch.mean(trajectory, dim=0))
ax.set_ylim(0, None)

ax.set_xlabel("position")
ax.set_ylabel("average activation")
sns.despine(ax=ax, offset=10)

# %% [markdown] id="An9kiTssb1ZN" tags=[] jp-MarkdownHeadingCollapsed=true
# ### Check that the generated patterns exactly match Fourier translations

# %% id="lfaJ_lShdR-W"
crt_fourier_U = np.array(
    [
        [1 / np.sqrt(n) * np.exp(2 * np.pi * k * l / n * 1j) for k in range(n)]
        for l in range(n)
    ]
)
crt_fourier_V = crt_fourier_U.conj().T

# %% colab={"base_uri": "https://localhost:8080/"} id="KNA_b1J8dR-X" outputId="31d9b45a-78a4-4a53-cb1f-8c5df1ac11e7"
(
    np.max(np.abs(crt_fourier_U @ crt_fourier_V - np.eye(n))),
    np.max(np.abs(crt_fourier_V @ crt_fourier_U - np.eye(n))),
)


# %% id="W4tI7mFPdR-Y"
def fourier_translate(x: np.ndarray, s: float) -> np.ndarray:
    n = len(x)
    # turns out the ** operator has the right branch cut to maket this work
    lbd = np.exp(-(2j * np.pi / n) * np.arange(n)) ** s

    # need to zero out highest frequency mode if n is even
    if n % 2 == 0:
        lbd[n // 2] = 0

    x_f = crt_fourier_V @ x
    x_f_shifted = lbd * x_f
    x_shifted = crt_fourier_U @ x_f_shifted
    
    assert np.max(np.abs(x_shifted.imag)) < 1e-6

    return x_shifted.real


# %% id="rHtzhAULdRSi"
crt_errors = []
for x, y, s in dataset_full:
    x = x.detach().numpy()
    y = y.detach().numpy()
    s = s.item()

    y_pred = fourier_translate(x, s)
    crt_errors.append(np.linalg.norm(y - y_pred))

crt_errors = np.array(crt_errors)

# %% colab={"base_uri": "https://localhost:8080/"} id="OU4KUyNkd_M6" outputId="2935fadb-f9cf-4a33-ee1d-b08abf83edef"
np.quantile(crt_errors, [0.005, 0.025, 0.500, 0.975, 0.995])

# %% colab={"base_uri": "https://localhost:8080/", "height": 96} id="47K9nTAU-fco" outputId="b678e18c-c2d0-4dd2-fd19-da3ba7360ebd"
plt.imshow(np.vstack((x, y, y_pred)))

# %% [markdown]
# ## Check that there is an exact solution for the simulated patterns using our model class

# %% [markdown] tags=[]
# ### Complex-valued version

# %% [markdown] id="pQU-ZrMTb45r"
# #### Check that a model initialized at Fourier solution yields (almost) zero loss

# %% id="-aND72Nj-MGy"
test_system = PlaceGridSystemNonBioCplx(n, n - 1)
sys_U_torch = torch.from_numpy(np.copy(crt_fourier_V.T)).type(torch.complex64)
sys_V_torch = torch.from_numpy(np.copy(crt_fourier_U.T)).type(torch.complex64)
test_system.U = torch.hstack((sys_U_torch[:, : n // 2], sys_U_torch[:, n // 2 + 1 :]))
test_system.V = torch.vstack((sys_V_torch[: n // 2, :], sys_V_torch[n // 2 + 1 :, :]))

crt_lbd = torch.exp(-(2j * np.pi / n) * torch.arange(n))
test_system.lbd = torch.hstack((crt_lbd[: n // 2], crt_lbd[n // 2 + 1 :]))

# this doesn't really matter for the math, but for consistency: let's put c.c. pairs
# next to each other
crt_order = torch.hstack(
    (
        torch.IntTensor([0]),
        torch.stack(
            (torch.arange(1, n // 2), torch.arange(n - 2, n // 2 - 1, - 1))
        ).T.ravel(),
    )
)
test_system.U = test_system.U[:, crt_order]
test_system.V = test_system.V[crt_order, :]
test_system.lbd = test_system.lbd[crt_order]

crt_scores = []
for x, y, s in dataset_full:
    crt_loss = test_system.loss(x[None, :], y[None, :], s[None]).item()
    crt_scores.append(crt_loss)

crt_scores = np.array(crt_scores)

# %% colab={"base_uri": "https://localhost:8080/"} id="mi2xY8L0Cmi6" outputId="1a41b50c-de27-44b1-fb3d-b0b2ef21eaed"
np.quantile(crt_scores, [0.005, 0.025, 0.500, 0.975, 0.995])

# %% colab={"base_uri": "https://localhost:8080/", "height": 96} id="zHRIjS1rTeXN" outputId="fe508623-fafc-456d-d63f-5362d0498867"
y_pred = test_system.propagate_place(x[None, :], s[None])
plt.imshow(np.vstack((x, y, y_pred)))

# %% colab={"base_uri": "https://localhost:8080/", "height": 282} id="5A7UgcZVTZYs" outputId="80b51c77-07b3-4a7b-87cc-b77be6521b23"
plt.imshow(test_system.U.real)

# %% [markdown] id="Hf3o2Y8RZY7r"
# #### Sanity check: projection to valid parameters does not spoil global optimum

# %% id="i50nRq2ZZcZ8"
crt_old_U = torch.clone(test_system.U)
crt_old_V = torch.clone(test_system.V)
crt_old_lbd = torch.clone(test_system.lbd)

test_system.project_to_real()

assert torch.allclose(crt_old_U, test_system.U)
assert torch.allclose(crt_old_V, test_system.V)
assert torch.allclose(crt_old_lbd, test_system.lbd)

# %% [markdown] id="ULf2KRNTYjtr"
# #### Sanity check: gradient (almost) zero at global optimum

# %% id="N3NeMljAYof_"
test_system.U.requires_grad = True
test_system.V.requires_grad = True
test_system.lbd.requires_grad = True

test_system.train()

crt_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=200)
crt_iter = iter(crt_dataloader)
crt_data, crt_target, crt_shift = next(crt_iter)

test_system.zero_grad()
crt_loss = test_system.loss(crt_data, crt_target, crt_shift)
crt_loss.backward()

assert torch.max(torch.abs(crt_loss)) < 1e-9

assert torch.max(torch.abs(test_system.U.grad)) < 1e-9
assert torch.max(torch.abs(test_system.V.grad)) < 1e-9
assert torch.max(torch.abs(test_system.lbd.grad)) < 1e-6

# %% [markdown] id="8KEXDbWbWkOG"
# #### Sanity check: SGD optimization does not lead us away from global optimum

# %% colab={"base_uri": "https://localhost:8080/"} id="OTKdAHfiWsir" outputId="0d308b51-99ef-48a9-b0ab-91b55c25fa7b"
batch_size = 200
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size)

test_system.U.requires_grad = True
test_system.V.requires_grad = True
test_system.lbd.requires_grad = True

test_optimizer = torch.optim.SGD(test_system.parameters(), lr=0.001)
test_scheduler = torch.optim.lr_scheduler.StepLR(
    test_optimizer, step_size=10, gamma=0.995
)
test_train_results = train(
    test_system,
    "cpu",
    dataloader_train,
    test_optimizer,
    test_set=dataloader_test,
    test_every=200,
    scheduler=test_scheduler,
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 282} id="-akQFzogXZvA" outputId="9609669b-9adc-4bc8-8b25-198eeb90d957"
plt.imshow(test_system.U.real.detach())

# %% colab={"base_uri": "https://localhost:8080/", "height": 288} id="ctd4bkpvYZOZ" outputId="2cc16bd8-0298-4ef5-979e-26e4ef90f040"
fig, ax = plt.subplots()
ax.semilogy(test_train_results.train_loss, lw=0.5, label="train")
ax.semilogy(
    test_train_results.test_idxs, test_train_results.test_loss, lw=1.0, label="test"
)
ax.set_xlabel("batch")
ax.set_ylabel("loss")

ax.axhline(
    test_train_results.test_loss[-1],
    lw=2.0,
    ls="--",
    c="C1",
    label=f"final test loss: {test_train_results.test_loss[-1]:.2g}"
)

ax.legend(frameon=False)

sns.despine(ax=ax, offset=10)

# %% [markdown]
# ### TODO: Checks for real-valued version

# %% [markdown]
# ## Try learning

# %% [markdown] id="K9I6NTxOb-dK"
# ### Complex-valued version

# %% colab={"base_uri": "https://localhost:8080/"} id="4ybXgIwOPgJ-" outputId="a6cfbd62-6da4-4602-8b32-83ffb3658a3c"
torch.manual_seed(0)

m = n - 1

system = PlaceGridSystemNonBioCplx(n, m)

original_U = torch.clone(system.U).detach()
original_V = torch.clone(system.V).detach()
original_lbd = torch.clone(system.lbd).detach()

optimizer = torch.optim.Adagrad(system.parameters(), lr=0.05)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.995)
scheduler = StepwiseScheduler(optimizer, [(50, 0.05), (100, 0.2), (100, 0.3), (250, 0.1)])

train_results = train(
    system,
    "cpu",
    dataloader_train,
    optimizer,
    test_set=dataloader_test,
    test_every=50,
    scheduler=scheduler,
)

# %% colab={"base_uri": "https://localhost:8080/"} id="hHwjBKEhvbBY" outputId="6e976c16-8863-47c5-c091-edfd870a3673"
scheduler.get_last_lr()

# %% colab={"base_uri": "https://localhost:8080/"} id="g5Tje2Pj4FRF" outputId="97834b66-02a8-4a1a-cf29-659a23977ed8"
(
    torch.median(torch.abs(system.U - original_U)),
    torch.median(torch.abs(system.V - original_V)),
    torch.median(torch.abs(system.lbd - original_lbd)),
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 288} id="z4y94yW-3aTD" outputId="46649305-1cc4-44ab-83d1-078b11cdb4fa"
fig, ax = plt.subplots()
ax.semilogy(train_results.train_loss, lw=0.5, label="train")
ax.semilogy(train_results.test_idxs, train_results.test_loss, lw=1.0, label="test")
ax.set_xlabel("batch")
ax.set_ylabel("loss")

ax.axhline(
    train_results.test_loss[-1],
    lw=2.0,
    ls="--",
    c="C1",
    label=f"final test loss: {train_results.test_loss[-1]:.2g}"
)

ax.legend(frameon=False)

sns.despine(ax=ax, offset=10)

# %% colab={"base_uri": "https://localhost:8080/", "height": 282} id="4_FpjTeD2OI3" outputId="365616d6-9602-46f3-c09a-c74b877a8f01"
crt_tensor = torch.real(system.U @ system.V).detach().numpy()
crt_lim = np.max(np.abs(crt_tensor))
plt.imshow(crt_tensor, cmap="RdBu", vmin=-crt_lim, vmax=crt_lim)
plt.colorbar()

# %% colab={"base_uri": "https://localhost:8080/", "height": 291} id="NkhI_uk68Fla" outputId="28944db8-c6ca-4c58-b020-01aa97855955"
fig, ax = plt.subplots()
crt_v = system.lbd.detach().numpy()

ax.axhline(0, ls=":", lw=1, c="gray")
ax.axvline(0, ls=":", lw=1, c="gray")

ax.scatter(crt_v.real, crt_v.imag)
ax.set_aspect(1)
ax.set_xlabel("Re($\\lambda$)")
ax.set_ylabel("Im($\\lambda$)")

sns.despine(ax=ax, offset=10)

# %% colab={"base_uri": "https://localhost:8080/"} id="o0J_lkEr6caL" outputId="182882fb-3368-4fed-9f43-8fc81904d8e4"
(
    torch.max(torch.abs(system.U)),
    torch.max(torch.abs(system.V)),
)

# %% [markdown] id="ZVoVwL8MDRTz"
# #### Try learned system on examples

# %% id="w7yv8vblDVIX"
torch.manual_seed(1)

test_simulator = PlaceGridMotionSimulator(n, sigma=2)

test_n_samples = 10
test_x = n * torch.rand(test_n_samples)
test_trajectory = []
for i in range(test_n_samples):
    test_simulator.x = test_x[i].item()
    test_trajectory.append(test_simulator().type(torch.float32))

test_trajectory = torch.stack(test_trajectory)

# %% id="zQ_B-mad_U5s"
test_moved = system.propagate_place(test_trajectory, 3 * torch.ones(test_n_samples))

# %% colab={"base_uri": "https://localhost:8080/", "height": 330} id="_sc18G5-eewh" outputId="c22cd697-7858-4c07-a504-3d360a9ac4ca"
fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True)
ax1.imshow(test_trajectory)
ax1.set_ylabel("time")
ax1.set_xlabel("position")

ax2.imshow(test_moved.detach().numpy())
ax2.set_ylabel("time")
ax2.set_xlabel("position")

# %% colab={"base_uri": "https://localhost:8080/"} id="LK5qBNhE4Y3o" outputId="e09dba93-21b4-495c-dc43-950e6e700031"
[torch.min(torch.abs(system.lbd)), torch.max(torch.abs(system.lbd))]

# %% colab={"base_uri": "https://localhost:8080/", "height": 622} id="1DkYy5qgengL" outputId="1f3ae84d-d483-4b7d-8813-be3e25a7e2e0"
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
crt_d = {"U": system.U, "V": system.V.T}
crt_ordering = np.argsort(np.abs(np.angle(system.lbd.detach().numpy())))
for i, crt_name in enumerate(crt_d):
    crt_mat = crt_d[crt_name].detach().numpy()

    crt_mat = crt_mat[:, crt_ordering]
    # crt_mat = crt_mat[crt_ordering, :]

    crt_lim = np.max(np.abs(crt_mat))

    ax_row = axs[i]
    ax_row[0].imshow(crt_mat.real, vmin=-crt_lim, vmax=crt_lim, cmap="RdBu")
    ax_row[1].imshow(crt_mat.imag, vmin=-crt_lim, vmax=crt_lim, cmap="RdBu")

    ax_row[0].set_ylabel(crt_name)

axs[1, 0].set_xlabel("Re")
axs[1, 1].set_xlabel("Im")

# %% [markdown] id="K9I6NTxOb-dK"
# ### Real-valued version

# %% colab={"base_uri": "https://localhost:8080/"} id="4ybXgIwOPgJ-" outputId="a6cfbd62-6da4-4602-8b32-83ffb3658a3c"
torch.manual_seed(0)

m = n - 1

system = PlaceGridSystemNonBio(n, m)

original_U = torch.clone(system.U).detach()
original_V = torch.clone(system.V).detach()
original_xi = torch.clone(system.xi).detach()
original_theta = torch.clone(system.theta).detach()

optimizer = torch.optim.Adagrad(system.parameters(), lr=0.05)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.995)
# scheduler = StepwiseScheduler(optimizer, [(50, 0.05), (100, 0.2), (100, 0.3), (250, 0.1)])
scheduler = None

train_results = train(
    system,
    "cpu",
    dataloader_train,
    optimizer,
    test_set=dataloader_test,
    test_every=50,
    scheduler=scheduler,
)

# %% colab={"base_uri": "https://localhost:8080/"} id="hHwjBKEhvbBY" outputId="6e976c16-8863-47c5-c091-edfd870a3673"
# scheduler.get_last_lr()

# %% colab={"base_uri": "https://localhost:8080/"} id="g5Tje2Pj4FRF" outputId="97834b66-02a8-4a1a-cf29-659a23977ed8"
(
    torch.max(torch.abs(system.U - original_U)),
    torch.max(torch.abs(system.V - original_V)),
    torch.max(torch.abs(system.xi - original_xi)),
    torch.max(torch.abs(system.theta - original_theta)),
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 288} id="z4y94yW-3aTD" outputId="46649305-1cc4-44ab-83d1-078b11cdb4fa"
fig, ax = plt.subplots()
ax.semilogy(train_results.train_loss, lw=0.5, label="train")
ax.semilogy(train_results.test_idxs, train_results.test_loss, lw=1.0, label="test")
ax.set_xlabel("batch")
ax.set_ylabel("loss")

ax.axhline(
    train_results.test_loss[-1],
    lw=2.0,
    ls="--",
    c="C1",
    label=f"final test loss: {train_results.test_loss[-1]:.2g}"
)

ax.legend(frameon=False)

sns.despine(ax=ax, offset=10)

# %% colab={"base_uri": "https://localhost:8080/", "height": 282} id="4_FpjTeD2OI3" outputId="365616d6-9602-46f3-c09a-c74b877a8f01"
crt_tensor = (system.U @ system.V).detach().numpy()
crt_lim = np.max(np.abs(crt_tensor))
plt.imshow(crt_tensor, cmap="RdBu", vmin=-crt_lim, vmax=crt_lim)
plt.colorbar()

# %% colab={"base_uri": "https://localhost:8080/", "height": 291} id="NkhI_uk68Fla" outputId="28944db8-c6ca-4c58-b020-01aa97855955"
fig, ax = plt.subplots()
crt_rho = (1 / torch.cosh(system.xi)).detach().numpy()
crt_theta = system.theta.detach().numpy()
crt_v = crt_rho[:len(crt_theta)] * np.exp(1j * crt_theta)
if len(crt_rho) > len(crt_theta):
    crt_v = np.hstack((crt_v, [crt_rho[-1]]))

ax.axhline(0, ls=":", lw=1, c="gray")
ax.axvline(0, ls=":", lw=1, c="gray")

ax.scatter(crt_v.real, crt_v.imag)
ax.set_aspect(1)
ax.set_xlabel("Re($\\lambda$)")
ax.set_ylabel("Im($\\lambda$)")

sns.despine(ax=ax, offset=10)

# %% colab={"base_uri": "https://localhost:8080/"} id="o0J_lkEr6caL" outputId="182882fb-3368-4fed-9f43-8fc81904d8e4"
(
    torch.max(torch.abs(system.U)),
    torch.max(torch.abs(system.V)),
)

# %% [markdown] id="ZVoVwL8MDRTz"
# #### Try learned system on examples

# %% id="w7yv8vblDVIX"
torch.manual_seed(1)

test_simulator = PlaceGridMotionSimulator(n, sigma=2)

test_n_samples = 10
test_x = n * torch.rand(test_n_samples)
test_trajectory = []
for i in range(test_n_samples):
    test_simulator.x = test_x[i].item()
    test_trajectory.append(test_simulator().type(torch.float32))

test_trajectory = torch.stack(test_trajectory)

# %% id="zQ_B-mad_U5s"
test_moved = system.propagate_place(test_trajectory, 3 * torch.ones(test_n_samples))

# %% colab={"base_uri": "https://localhost:8080/", "height": 330} id="_sc18G5-eewh" outputId="c22cd697-7858-4c07-a504-3d360a9ac4ca"
fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True)
ax1.imshow(test_trajectory)
ax1.set_ylabel("time")
ax1.set_xlabel("position")

ax2.imshow(test_moved.detach().numpy())
ax2.set_ylabel("time")
ax2.set_xlabel("position")

# %% colab={"base_uri": "https://localhost:8080/"} id="LK5qBNhE4Y3o" outputId="e09dba93-21b4-495c-dc43-950e6e700031"
[torch.min(torch.abs(system.xi)), torch.max(torch.abs(system.xi))]

# %% colab={"base_uri": "https://localhost:8080/", "height": 622} id="1DkYy5qgengL" outputId="1f3ae84d-d483-4b7d-8813-be3e25a7e2e0"
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
crt_d = {"$U$": system.U, "$V^\\top$": system.V.T}
# crt_ordering = np.argsort(np.abs(system.theta.detach().numpy()))
for i, crt_name in enumerate(crt_d):
    crt_mat = crt_d[crt_name].detach().numpy()

    # crt_mat = crt_mat[:, crt_ordering]
    # crt_mat = crt_mat[crt_ordering, :]

    crt_lim = np.max(np.abs(crt_mat))
    
    ax = axs[i]
    ax.imshow(crt_mat, vmin=-crt_lim, vmax=crt_lim, cmap="RdBu")

    ax.set_title(crt_name)

# %%
1 / np.cosh(system.xi.detach().numpy())

# %%
tmp = test_trajectory[0]
tmp_g = system.to_grid(tmp)
fig, ax = plt.subplots()
ax.imshow([tmp.numpy()], vmin=0)

fig, ax = plt.subplots()
crt_l = np.max(np.abs(tmp_g.detach().numpy()))
ax.imshow([tmp_g.detach().numpy()], vmin=-crt_l, vmax=crt_l, cmap="RdBu")

tmp_back = system.from_grid(tmp_g)
fig, ax = plt.subplots()
ax.imshow([tmp_back.detach().numpy()], vmin=0)

# %%
fig, ax = plt.subplots()
ax.plot(tmp.detach().numpy())
ax.plot(tmp_back.detach().numpy())

# %%
plt.imshow(system.U.detach().numpy())
plt.colorbar()

# %% [markdown] id="euyEHDjR1mOX"
# ## Test learning on smaller system

# %% id="HjrmlD231kC1"
torch.manual_seed(0)

n = 8
simulator = PlaceGridMotionSimulator(n, sigma=0.5)

n_samples = 500_000
s = torch.normal(torch.zeros(n_samples), 1.0)
trajectory = simulator.batch(s)

dataset_full = [(trajectory[i], trajectory[i + 1], s[i]) for i in range(n_samples - 1)]

test_size = 1000
dataset_train = dataset_full[:-test_size]
dataset_test = dataset_full[-test_size:]

batch_size = 200
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size)

# %% colab={"base_uri": "https://localhost:8080/", "height": 313} id="-tR-EcHK1kC1" outputId="1fcd5faf-b16b-4e69-eaed-d55a633a409b"
crt_n = 4 * n
crt_step = 1

fig, (ax1, ax2) = plt.subplots(
    1, 2, sharey=True, constrained_layout=True, gridspec_kw={"width_ratios": (3, 1)}
)
ax1.imshow(trajectory[:crt_n:crt_step])
ax1.set_ylabel("time")
ax1.set_xlabel("position")

ax2.barh(width=s[:crt_n:crt_step], y=np.arange(0, crt_n // crt_step))
ax2.axvline(0, ls=":", c="gray")
crt_xl = max(ax2.get_xlim())
ax2.set_xlim(-crt_xl, crt_xl)
ax2.set_ylim(ax1.get_ylim())
ax2.invert_yaxis()
ax2.set_xlabel("shift")
sns.despine(ax=ax2, left=True)

# %% colab={"base_uri": "https://localhost:8080/", "height": 288} id="rvSgTAtk1kC2" outputId="bee2a762-51c2-446d-d8c2-6532123224db"
fig, ax = plt.subplots()
ax.plot(torch.mean(trajectory, dim=0))
ax.set_ylim(0, None)

ax.set_xlabel("position")
ax.set_ylabel("average activation")
sns.despine(ax=ax, offset=10)

# %% [markdown]
# ### Complex-valued simulation

# %% colab={"base_uri": "https://localhost:8080/"} id="IOhW9zr21kC2" outputId="3afb29ae-c2a7-4137-fc7d-53b3df09b35e"
torch.manual_seed(0)

m = n - 1

system = PlaceGridSystemNonBioCplx(n, m)

original_U = torch.clone(system.U).detach()
original_V = torch.clone(system.V).detach()
original_lbd = torch.clone(system.lbd).detach()

optimizer = torch.optim.AdamW(system.parameters(), lr=0.01)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.995)
scheduler = StepwiseScheduler(
    optimizer,
    [(100, 0.01), (100, 0.05), (100, 0.03), (300, 0.02), (1600, 0.01), (100, 0.005)]
)
# scheduler = None
train_results = train(
    system,
    "cpu",
    dataloader_train,
    optimizer,
    test_set=dataloader_test,
    test_every=50,
    scheduler=scheduler,
)

# %% colab={"base_uri": "https://localhost:8080/"} id="JWLExrOc1kC2" outputId="dd0ea4c7-bed1-48fb-faa3-023829028047"
scheduler.get_last_lr()

# %% colab={"base_uri": "https://localhost:8080/"} id="a1K3zZ2F1kC2" outputId="1a708ae2-7716-4421-ed10-ba79dc90c237"
(
    torch.median(torch.abs(system.U - original_U)),
    torch.median(torch.abs(system.V - original_V)),
    torch.median(torch.abs(system.lbd - original_lbd)),
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 288} id="m6dh_6_31kC3" outputId="1b6bb86a-e4ab-4aea-b33f-19cfed2a70bc"
fig, ax = plt.subplots()
ax.semilogy(train_results.train_loss, lw=0.5, label="train")
ax.semilogy(train_results.test_idxs, train_results.test_loss, lw=1.0, label="test")
ax.set_xlabel("batch")
ax.set_ylabel("loss")

ax.axhline(
    train_results.test_loss[-1],
    lw=2.0,
    ls="--",
    c="C1",
    label=f"final test loss: {train_results.test_loss[-1]:.2g}"
)

ax.legend(frameon=False)

sns.despine(ax=ax, offset=10)

# %% colab={"base_uri": "https://localhost:8080/", "height": 282} id="yZNOnbKV1kC3" outputId="bdecdb0c-381d-4c4c-fcda-75b60678b7e9"
crt_tensor = torch.real(system.U @ system.V).detach().numpy()
crt_lim = np.max(np.abs(crt_tensor))
plt.imshow(crt_tensor, cmap="RdBu", vmin=-crt_lim, vmax=crt_lim)
plt.colorbar()

# %% colab={"base_uri": "https://localhost:8080/", "height": 289} id="3sxD_nl21kC3" outputId="729a9946-4aa2-4fb4-c48c-a7b511d1da29"
fig, ax = plt.subplots()
crt_v = system.lbd.detach().numpy()

ax.axhline(0, ls=":", lw=1, c="gray")
ax.axvline(0, ls=":", lw=1, c="gray")

ax.scatter(crt_v.real, crt_v.imag)
ax.set_aspect(1)
ax.set_xlabel("Re($\\lambda$)")
ax.set_ylabel("Im($\\lambda$)")

sns.despine(ax=ax, offset=10)

# %% colab={"base_uri": "https://localhost:8080/"} id="j-URPysj1kC3" outputId="f2033a74-e672-42ed-9f79-f390d3c29ede"
(
    torch.max(torch.abs(system.U)),
    torch.max(torch.abs(system.V)),
)

# %% [markdown] id="S_e7wGsl2uNv"
# ### Try learned system on examples

# %% id="fGJUwTIP2uNw"
torch.manual_seed(1)

test_simulator = PlaceGridMotionSimulator(n, sigma=0.5)

test_n_samples = 10
test_x = n * torch.rand(test_n_samples)
test_trajectory = []
for i in range(test_n_samples):
    test_simulator.x = test_x[i].item()
    test_trajectory.append(test_simulator().type(torch.float32))

test_trajectory = torch.stack(test_trajectory)

# %% id="vvgdMJbu2uNw"
test_moved = system.propagate_place(test_trajectory, 1 * torch.ones(test_n_samples))

# %% colab={"base_uri": "https://localhost:8080/", "height": 330} id="3Hgin79U2uNw" outputId="85e60a12-a4e4-4cfa-930b-71b3b9145f7a"
fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True)
ax1.imshow(test_trajectory)
ax1.set_ylabel("time")
ax1.set_xlabel("position")

ax2.imshow(test_moved.detach().numpy())
ax2.set_ylabel("time")
ax2.set_xlabel("position")

# %% colab={"base_uri": "https://localhost:8080/"} id="bXTAXQFV2uNw" outputId="244ea24d-04b0-4d65-c336-6724f5b7e7db"
[torch.min(torch.abs(system.lbd)), torch.max(torch.abs(system.lbd))]

# %% colab={"base_uri": "https://localhost:8080/", "height": 622} id="BAv-pYnS2uNw" outputId="99bdcf06-266d-4f66-ec69-61d46c41a3b8"
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
crt_d = {"U": system.U, "V": system.V.T}
crt_ordering = np.argsort(np.abs(np.angle(system.lbd.detach().numpy())))
for i, crt_name in enumerate(crt_d):
    crt_mat = crt_d[crt_name].detach().numpy()

    crt_mat = crt_mat[:, crt_ordering]
    # crt_mat = crt_mat[crt_ordering, :]

    crt_lim = np.max(np.abs(crt_mat))

    ax_row = axs[i]
    ax_row[0].imshow(crt_mat.real, vmin=-crt_lim, vmax=crt_lim, cmap="RdBu")
    ax_row[1].imshow(crt_mat.imag, vmin=-crt_lim, vmax=crt_lim, cmap="RdBu")

    ax_row[0].set_ylabel(crt_name)

axs[1, 0].set_xlabel("Re")
axs[1, 1].set_xlabel("Im")

# %% [markdown]
# ### Real-valued simulation

# %% colab={"base_uri": "https://localhost:8080/"} id="IOhW9zr21kC2" outputId="3afb29ae-c2a7-4137-fc7d-53b3df09b35e"
torch.manual_seed(0)

m = n - 1

system = PlaceGridSystemNonBio(n, m)

original_U = torch.clone(system.U).detach()
original_V = torch.clone(system.V).detach()
original_xi = torch.clone(system.xi).detach()
original_theta = torch.clone(system.theta).detach()

optimizer = torch.optim.AdamW(system.parameters(), lr=0.01)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.995)
# scheduler = StepwiseScheduler(
#     optimizer,
#     [(100, 0.01), (100, 0.05), (100, 0.03), (300, 0.02), (1600, 0.01), (100, 0.005)]
# )
scheduler = None
train_results = train(
    system,
    "cpu",
    dataloader_train,
    optimizer,
    test_set=dataloader_test,
    test_every=50,
    scheduler=scheduler,
)

# %% colab={"base_uri": "https://localhost:8080/"} id="JWLExrOc1kC2" outputId="dd0ea4c7-bed1-48fb-faa3-023829028047"
# scheduler.get_last_lr()

# %% colab={"base_uri": "https://localhost:8080/"} id="a1K3zZ2F1kC2" outputId="1a708ae2-7716-4421-ed10-ba79dc90c237"
(
    torch.median(torch.abs(system.U - original_U)),
    torch.median(torch.abs(system.V - original_V)),
    torch.median(torch.abs(system.xi - original_xi)),
    torch.median(torch.abs(system.theta - original_theta)),
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 288} id="m6dh_6_31kC3" outputId="1b6bb86a-e4ab-4aea-b33f-19cfed2a70bc"
fig, ax = plt.subplots()
ax.semilogy(train_results.train_loss, lw=0.5, label="train")
ax.semilogy(train_results.test_idxs, train_results.test_loss, lw=1.0, label="test")
ax.set_xlabel("batch")
ax.set_ylabel("loss")

ax.axhline(
    train_results.test_loss[-1],
    lw=2.0,
    ls="--",
    c="C1",
    label=f"final test loss: {train_results.test_loss[-1]:.2g}"
)

ax.legend(frameon=False)

sns.despine(ax=ax, offset=10)

# %% colab={"base_uri": "https://localhost:8080/", "height": 282} id="yZNOnbKV1kC3" outputId="bdecdb0c-381d-4c4c-fcda-75b60678b7e9"
crt_tensor = (system.U @ system.V).detach().numpy()
crt_lim = np.max(np.abs(crt_tensor))
plt.imshow(crt_tensor, cmap="RdBu", vmin=-crt_lim, vmax=crt_lim)
plt.colorbar()

# %% colab={"base_uri": "https://localhost:8080/", "height": 289} id="3sxD_nl21kC3" outputId="729a9946-4aa2-4fb4-c48c-a7b511d1da29"
fig, ax = plt.subplots()
crt_rho = (1 / torch.cosh(system.xi)).detach().numpy()
crt_theta = system.theta.detach().numpy()
crt_v = crt_rho[:len(crt_theta)] * np.exp(1j * crt_theta)
if len(crt_rho) > len(crt_theta):
    crt_v = np.hstack((crt_v, [crt_rho[-1]]))

ax.axhline(0, ls=":", lw=1, c="gray")
ax.axvline(0, ls=":", lw=1, c="gray")

ax.scatter(crt_v.real, crt_v.imag)
ax.set_aspect(1)
ax.set_xlabel("Re($\\lambda$)")
ax.set_ylabel("Im($\\lambda$)")

sns.despine(ax=ax, offset=10)

# %% colab={"base_uri": "https://localhost:8080/"} id="j-URPysj1kC3" outputId="f2033a74-e672-42ed-9f79-f390d3c29ede"
(
    torch.max(torch.abs(system.U)),
    torch.max(torch.abs(system.V)),
)

# %% [markdown] id="S_e7wGsl2uNv"
# ### Try learned system on examples

# %% id="fGJUwTIP2uNw"
torch.manual_seed(1)

test_simulator = PlaceGridMotionSimulator(n, sigma=0.5)

test_n_samples = 10
test_x = n * torch.rand(test_n_samples)
test_trajectory = []
for i in range(test_n_samples):
    test_simulator.x = test_x[i].item()
    test_trajectory.append(test_simulator().type(torch.float32))

test_trajectory = torch.stack(test_trajectory)

# %% id="vvgdMJbu2uNw"
test_moved = system.propagate_place(test_trajectory, 1 * torch.ones(test_n_samples))

# %% colab={"base_uri": "https://localhost:8080/", "height": 330} id="3Hgin79U2uNw" outputId="85e60a12-a4e4-4cfa-930b-71b3b9145f7a"
fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True)
ax1.imshow(test_trajectory)
ax1.set_ylabel("time")
ax1.set_xlabel("position")

ax2.imshow(test_moved.detach().numpy())
ax2.set_ylabel("time")
ax2.set_xlabel("position")

# %% colab={"base_uri": "https://localhost:8080/"} id="bXTAXQFV2uNw" outputId="244ea24d-04b0-4d65-c336-6724f5b7e7db"
[torch.min(torch.abs(system.xi)), torch.max(torch.abs(system.xi))]

# %% colab={"base_uri": "https://localhost:8080/", "height": 622} id="BAv-pYnS2uNw" outputId="99bdcf06-266d-4f66-ec69-61d46c41a3b8"
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
crt_d = {"$U$": system.U, "$V^\\top$": system.V.T}
# crt_ordering = np.argsort(np.abs(system.theta.detach().numpy()))
for i, crt_name in enumerate(crt_d):
    crt_mat = crt_d[crt_name].detach().numpy()

    # crt_mat = crt_mat[:, crt_ordering]
    # crt_mat = crt_mat[crt_ordering, :]

    crt_lim = np.max(np.abs(crt_mat))
    
    ax = axs[i]
    ax.imshow(crt_mat, vmin=-crt_lim, vmax=crt_lim, cmap="RdBu")

    ax.set_title(crt_name)

# %%
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
crt_d = {"$U$": system.U, "$V^\\top$": system.V.T}
for i, crt_name in enumerate(crt_d):
    ax = axs[i]
    crt_mat = crt_d[crt_name].detach().numpy()
    crt_lim = np.max(np.abs(crt_mat))
    for k in range(m):
        crt_v = crt_mat[:, k]
        ax.axhline(k, ls=":", c="gray", alpha=0.7)
        ax.plot(np.arange(n), k + 0.4 * crt_v / crt_lim)
    
    ax.set_title(crt_name)
    sns.despine(ax=ax, offset=10)

# %%
fig, ax = plt.subplots(figsize=(6, 6))
crt_u = system.U.detach().numpy()
crt_vt = system.V.T.detach().numpy()
crt_lim = max(np.max(np.abs(_)) for _ in [crt_u, crt_vt])
ax.plot([-crt_lim, crt_lim], [-crt_lim, crt_lim], c="gray", ls=":")
ax.scatter(crt_u, crt_vt)
ax.set_xlabel("elements of $U$")
ax.set_ylabel("elements of $V^\\top$")

ax.set_aspect(1)
sns.despine(ax=ax, offset=10)

# %% [markdown] id="euyEHDjR1mOX"
# ## Test learning with non-Fourier bump movements

# %% id="HjrmlD231kC1"
torch.manual_seed(0)

n = 8
simulator = PlaceGridMotionSimulator(n, sigma=0.5, fourier=False)

n_samples = 500_000
s = torch.normal(torch.zeros(n_samples), 1.0)
trajectory = simulator.batch(s)

dataset_full = [(trajectory[i], trajectory[i + 1], s[i]) for i in range(n_samples - 1)]

test_size = 1000
dataset_train = dataset_full[:-test_size]
dataset_test = dataset_full[-test_size:]

batch_size = 200
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size)

# %% colab={"base_uri": "https://localhost:8080/", "height": 313} id="-tR-EcHK1kC1" outputId="1fcd5faf-b16b-4e69-eaed-d55a633a409b"
crt_n = 4 * n
crt_step = 1

fig, (ax1, ax2) = plt.subplots(
    1, 2, sharey=True, constrained_layout=True, gridspec_kw={"width_ratios": (3, 1)}
)
ax1.imshow(trajectory[:crt_n:crt_step])
ax1.set_ylabel("time")
ax1.set_xlabel("position")

ax2.barh(width=s[:crt_n:crt_step], y=np.arange(0, crt_n // crt_step))
ax2.axvline(0, ls=":", c="gray")
crt_xl = max(ax2.get_xlim())
ax2.set_xlim(-crt_xl, crt_xl)
ax2.set_ylim(ax1.get_ylim())
ax2.invert_yaxis()
ax2.set_xlabel("shift")
sns.despine(ax=ax2, left=True)

# %% colab={"base_uri": "https://localhost:8080/", "height": 288} id="rvSgTAtk1kC2" outputId="bee2a762-51c2-446d-d8c2-6532123224db"
fig, ax = plt.subplots()
ax.plot(torch.mean(trajectory, dim=0))
ax.set_ylim(0, None)

ax.set_xlabel("position")
ax.set_ylabel("average activation")
sns.despine(ax=ax, offset=10)

# %% [markdown]
# ### Real-valued simulation

# %% colab={"base_uri": "https://localhost:8080/"} id="IOhW9zr21kC2" outputId="3afb29ae-c2a7-4137-fc7d-53b3df09b35e"
torch.manual_seed(0)

m = n - 1

system = PlaceGridSystemNonBio(n, m)

original_U = torch.clone(system.U).detach()
original_V = torch.clone(system.V).detach()
original_xi = torch.clone(system.xi).detach()
original_theta = torch.clone(system.theta).detach()

optimizer = torch.optim.AdamW(system.parameters(), lr=0.01)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.995)
# scheduler = StepwiseScheduler(
#     optimizer,
#     [(100, 0.01), (100, 0.05), (100, 0.03), (300, 0.02), (1600, 0.01), (100, 0.005)]
# )
scheduler = None
train_results = train(
    system,
    "cpu",
    dataloader_train,
    optimizer,
    test_set=dataloader_test,
    test_every=50,
    scheduler=scheduler,
)

# %% colab={"base_uri": "https://localhost:8080/"} id="JWLExrOc1kC2" outputId="dd0ea4c7-bed1-48fb-faa3-023829028047"
# scheduler.get_last_lr()

# %% colab={"base_uri": "https://localhost:8080/"} id="a1K3zZ2F1kC2" outputId="1a708ae2-7716-4421-ed10-ba79dc90c237"
(
    torch.median(torch.abs(system.U - original_U)),
    torch.median(torch.abs(system.V - original_V)),
    torch.median(torch.abs(system.xi - original_xi)),
    torch.median(torch.abs(system.theta - original_theta)),
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 288} id="m6dh_6_31kC3" outputId="1b6bb86a-e4ab-4aea-b33f-19cfed2a70bc"
fig, ax = plt.subplots()
ax.semilogy(train_results.train_loss, lw=0.5, label="train")
ax.semilogy(train_results.test_idxs, train_results.test_loss, lw=1.0, label="test")
ax.set_xlabel("batch")
ax.set_ylabel("loss")

ax.axhline(
    train_results.test_loss[-1],
    lw=2.0,
    ls="--",
    c="C1",
    label=f"final test loss: {train_results.test_loss[-1]:.2g}"
)

ax.legend(frameon=False)

sns.despine(ax=ax, offset=10)

# %% colab={"base_uri": "https://localhost:8080/", "height": 282} id="yZNOnbKV1kC3" outputId="bdecdb0c-381d-4c4c-fcda-75b60678b7e9"
crt_tensor = (system.U @ system.V).detach().numpy()
crt_lim = np.max(np.abs(crt_tensor))
plt.imshow(crt_tensor, cmap="RdBu", vmin=-crt_lim, vmax=crt_lim)
plt.colorbar()

# %% colab={"base_uri": "https://localhost:8080/", "height": 289} id="3sxD_nl21kC3" outputId="729a9946-4aa2-4fb4-c48c-a7b511d1da29"
fig, ax = plt.subplots()
crt_rho = (1 / torch.cosh(system.xi)).detach().numpy()
crt_theta = system.theta.detach().numpy()
crt_v = crt_rho[:len(crt_theta)] * np.exp(1j * crt_theta)
if len(crt_rho) > len(crt_theta):
    crt_v = np.hstack((crt_v, [crt_rho[-1]]))

ax.axhline(0, ls=":", lw=1, c="gray")
ax.axvline(0, ls=":", lw=1, c="gray")

ax.scatter(crt_v.real, crt_v.imag)
ax.set_aspect(1)
ax.set_xlabel("Re($\\lambda$)")
ax.set_ylabel("Im($\\lambda$)")

sns.despine(ax=ax, offset=10)

# %% colab={"base_uri": "https://localhost:8080/"} id="j-URPysj1kC3" outputId="f2033a74-e672-42ed-9f79-f390d3c29ede"
(
    torch.max(torch.abs(system.U)),
    torch.max(torch.abs(system.V)),
)

# %% [markdown] id="S_e7wGsl2uNv"
# ### Try learned system on examples

# %% id="fGJUwTIP2uNw"
torch.manual_seed(1)

test_simulator = PlaceGridMotionSimulator(n, sigma=0.5)

test_n_samples = 10
test_x = n * torch.rand(test_n_samples)
test_trajectory = []
for i in range(test_n_samples):
    test_simulator.x = test_x[i].item()
    test_trajectory.append(test_simulator().type(torch.float32))

test_trajectory = torch.stack(test_trajectory)

# %% id="vvgdMJbu2uNw"
test_moved = system.propagate_place(test_trajectory, 1 * torch.ones(test_n_samples))

# %% colab={"base_uri": "https://localhost:8080/", "height": 330} id="3Hgin79U2uNw" outputId="85e60a12-a4e4-4cfa-930b-71b3b9145f7a"
fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True)
ax1.imshow(test_trajectory)
ax1.set_ylabel("time")
ax1.set_xlabel("position")

ax2.imshow(test_moved.detach().numpy())
ax2.set_ylabel("time")
ax2.set_xlabel("position")

# %% colab={"base_uri": "https://localhost:8080/"} id="bXTAXQFV2uNw" outputId="244ea24d-04b0-4d65-c336-6724f5b7e7db"
[torch.min(torch.abs(system.xi)), torch.max(torch.abs(system.xi))]

# %% colab={"base_uri": "https://localhost:8080/", "height": 622} id="BAv-pYnS2uNw" outputId="99bdcf06-266d-4f66-ec69-61d46c41a3b8"
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
crt_d = {"$U$": system.U, "$V^\\top$": system.V.T}
# crt_ordering = np.argsort(np.abs(system.theta.detach().numpy()))
for i, crt_name in enumerate(crt_d):
    crt_mat = crt_d[crt_name].detach().numpy()

    # crt_mat = crt_mat[:, crt_ordering]
    # crt_mat = crt_mat[crt_ordering, :]

    crt_lim = np.max(np.abs(crt_mat))
    
    ax = axs[i]
    ax.imshow(crt_mat, vmin=-crt_lim, vmax=crt_lim, cmap="RdBu")

    ax.set_title(crt_name)

# %%
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
crt_d = {"$U$": system.U, "$V^\\top$": system.V.T}
for i, crt_name in enumerate(crt_d):
    ax = axs[i]
    crt_mat = crt_d[crt_name].detach().numpy()
    crt_lim = np.max(np.abs(crt_mat))
    for k in range(m):
        crt_v = crt_mat[:, k]
        ax.axhline(k, ls=":", c="gray", alpha=0.7)
        ax.plot(np.arange(n), k + 0.4 * crt_v / crt_lim)
    
    ax.set_title(crt_name)
    sns.despine(ax=ax, offset=10)

# %%
fig, ax = plt.subplots(figsize=(6, 6))
crt_u = system.U.detach().numpy()
crt_vt = system.V.T.detach().numpy()
crt_lim = max(np.max(np.abs(_)) for _ in [crt_u, crt_vt])
ax.plot([-crt_lim, crt_lim], [-crt_lim, crt_lim], c="gray", ls=":")
ax.scatter(crt_u, crt_vt)
ax.set_xlabel("elements of $U$")
ax.set_ylabel("elements of $V^\\top$")

ax.set_aspect(1)
sns.despine(ax=ax, offset=10)

# %% [markdown] id="euyEHDjR1mOX"
# ## Test learning with non-periodic bump movements

# %% id="HjrmlD231kC1"
torch.manual_seed(0)

n = 8
simulator = PlaceGridMotionSimulator(n, sigma=0.5, fourier=False, periodic=False)

n_samples = 500_000
s = torch.normal(torch.zeros(n_samples), 1.0)
trajectory = simulator.batch(s)

dataset_full = [(trajectory[i], trajectory[i + 1], s[i]) for i in range(n_samples - 1)]

test_size = 1000
dataset_train = dataset_full[:-test_size]
dataset_test = dataset_full[-test_size:]

batch_size = 200
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size)

# %% colab={"base_uri": "https://localhost:8080/", "height": 313} id="-tR-EcHK1kC1" outputId="1fcd5faf-b16b-4e69-eaed-d55a633a409b"
crt_n = 4 * n
crt_step = 1

fig, (ax1, ax2) = plt.subplots(
    1, 2, sharey=True, constrained_layout=True, gridspec_kw={"width_ratios": (3, 1)}
)
ax1.imshow(trajectory[:crt_n:crt_step])
ax1.set_ylabel("time")
ax1.set_xlabel("position")

ax2.barh(width=s[:crt_n:crt_step], y=np.arange(0, crt_n // crt_step))
ax2.axvline(0, ls=":", c="gray")
crt_xl = max(ax2.get_xlim())
ax2.set_xlim(-crt_xl, crt_xl)
ax2.set_ylim(ax1.get_ylim())
ax2.invert_yaxis()
ax2.set_xlabel("shift")
sns.despine(ax=ax2, left=True)

# %% colab={"base_uri": "https://localhost:8080/", "height": 288} id="rvSgTAtk1kC2" outputId="bee2a762-51c2-446d-d8c2-6532123224db"
fig, ax = plt.subplots()
ax.plot(torch.mean(trajectory, dim=0))
ax.set_ylim(0, None)

ax.set_xlabel("position")
ax.set_ylabel("average activation")
sns.despine(ax=ax, offset=10)

# %% [markdown]
# ### Real-valued simulation

# %% colab={"base_uri": "https://localhost:8080/"} id="IOhW9zr21kC2" outputId="3afb29ae-c2a7-4137-fc7d-53b3df09b35e"
torch.manual_seed(0)

m = n - 1

system = PlaceGridSystemNonBio(n, m)

original_U = torch.clone(system.U).detach()
original_V = torch.clone(system.V).detach()
original_xi = torch.clone(system.xi).detach()
original_theta = torch.clone(system.theta).detach()

optimizer = torch.optim.AdamW(system.parameters(), lr=0.01)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.995)
# scheduler = StepwiseScheduler(
#     optimizer,
#     [(100, 0.01), (100, 0.05), (100, 0.03), (300, 0.02), (1600, 0.01), (100, 0.005)]
# )
scheduler = None
train_results = train(
    system,
    "cpu",
    dataloader_train,
    optimizer,
    test_set=dataloader_test,
    test_every=50,
    scheduler=scheduler,
)

# %% colab={"base_uri": "https://localhost:8080/"} id="JWLExrOc1kC2" outputId="dd0ea4c7-bed1-48fb-faa3-023829028047"
# scheduler.get_last_lr()

# %% colab={"base_uri": "https://localhost:8080/"} id="a1K3zZ2F1kC2" outputId="1a708ae2-7716-4421-ed10-ba79dc90c237"
(
    torch.median(torch.abs(system.U - original_U)),
    torch.median(torch.abs(system.V - original_V)),
    torch.median(torch.abs(system.xi - original_xi)),
    torch.median(torch.abs(system.theta - original_theta)),
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 288} id="m6dh_6_31kC3" outputId="1b6bb86a-e4ab-4aea-b33f-19cfed2a70bc"
fig, ax = plt.subplots()
ax.semilogy(train_results.train_loss, lw=0.5, label="train")
ax.semilogy(train_results.test_idxs, train_results.test_loss, lw=1.0, label="test")
ax.set_xlabel("batch")
ax.set_ylabel("loss")

ax.axhline(
    train_results.test_loss[-1],
    lw=2.0,
    ls="--",
    c="C1",
    label=f"final test loss: {train_results.test_loss[-1]:.2g}"
)

ax.legend(frameon=False)

sns.despine(ax=ax, offset=10)

# %% colab={"base_uri": "https://localhost:8080/", "height": 282} id="yZNOnbKV1kC3" outputId="bdecdb0c-381d-4c4c-fcda-75b60678b7e9"
crt_tensor = (system.U @ system.V).detach().numpy()
crt_lim = np.max(np.abs(crt_tensor))
plt.imshow(crt_tensor, cmap="RdBu", vmin=-crt_lim, vmax=crt_lim)
plt.colorbar()

# %% colab={"base_uri": "https://localhost:8080/", "height": 289} id="3sxD_nl21kC3" outputId="729a9946-4aa2-4fb4-c48c-a7b511d1da29"
fig, ax = plt.subplots()
crt_rho = (1 / torch.cosh(system.xi)).detach().numpy()
crt_theta = system.theta.detach().numpy()
crt_v = crt_rho[:len(crt_theta)] * np.exp(1j * crt_theta)
if len(crt_rho) > len(crt_theta):
    crt_v = np.hstack((crt_v, [crt_rho[-1]]))

ax.axhline(0, ls=":", lw=1, c="gray")
ax.axvline(0, ls=":", lw=1, c="gray")

ax.scatter(crt_v.real, crt_v.imag)
ax.set_aspect(1)
ax.set_xlabel("Re($\\lambda$)")
ax.set_ylabel("Im($\\lambda$)")

sns.despine(ax=ax, offset=10)

# %% colab={"base_uri": "https://localhost:8080/"} id="j-URPysj1kC3" outputId="f2033a74-e672-42ed-9f79-f390d3c29ede"
(
    torch.max(torch.abs(system.U)),
    torch.max(torch.abs(system.V)),
)

# %% [markdown] id="S_e7wGsl2uNv"
# ### Try learned system on examples

# %% id="fGJUwTIP2uNw"
torch.manual_seed(1)

test_simulator = PlaceGridMotionSimulator(n, sigma=0.5)

test_n_samples = 10
test_x = n * torch.rand(test_n_samples)
test_trajectory = []
for i in range(test_n_samples):
    test_simulator.x = test_x[i].item()
    test_trajectory.append(test_simulator().type(torch.float32))

test_trajectory = torch.stack(test_trajectory)

# %% id="vvgdMJbu2uNw"
test_moved = system.propagate_place(test_trajectory, 2 * torch.ones(test_n_samples))

# %% colab={"base_uri": "https://localhost:8080/", "height": 330} id="3Hgin79U2uNw" outputId="85e60a12-a4e4-4cfa-930b-71b3b9145f7a"
fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True)
ax1.imshow(test_trajectory)
ax1.set_ylabel("time")
ax1.set_xlabel("position")

ax2.imshow(test_moved.detach().numpy())
ax2.set_ylabel("time")
ax2.set_xlabel("position")

# %% colab={"base_uri": "https://localhost:8080/"} id="bXTAXQFV2uNw" outputId="244ea24d-04b0-4d65-c336-6724f5b7e7db"
[torch.min(torch.abs(system.xi)), torch.max(torch.abs(system.xi))]

# %% colab={"base_uri": "https://localhost:8080/", "height": 622} id="BAv-pYnS2uNw" outputId="99bdcf06-266d-4f66-ec69-61d46c41a3b8"
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
crt_d = {"$U$": system.U, "$V^\\top$": system.V.T}
# crt_ordering = np.argsort(np.abs(system.theta.detach().numpy()))
for i, crt_name in enumerate(crt_d):
    crt_mat = crt_d[crt_name].detach().numpy()

    # crt_mat = crt_mat[:, crt_ordering]
    # crt_mat = crt_mat[crt_ordering, :]

    crt_lim = np.max(np.abs(crt_mat))
    
    ax = axs[i]
    ax.imshow(crt_mat, vmin=-crt_lim, vmax=crt_lim, cmap="RdBu")

    ax.set_title(crt_name)

# %%
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
crt_d = {"$U$": system.U, "$V^\\top$": system.V.T}
for i, crt_name in enumerate(crt_d):
    ax = axs[i]
    crt_mat = crt_d[crt_name].detach().numpy()
    crt_lim = np.max(np.abs(crt_mat))
    for k in range(m):
        crt_v = crt_mat[:, k]
        ax.axhline(k, ls=":", c="gray", alpha=0.7)
        ax.plot(np.arange(n), k + 0.4 * crt_v / crt_lim)
    
    ax.set_title(crt_name)
    sns.despine(ax=ax, offset=10)

# %%
fig, ax = plt.subplots(figsize=(6, 6))
crt_u = system.U.detach().numpy()
crt_vt = system.V.T.detach().numpy()
crt_lim = max(np.max(np.abs(_)) for _ in [crt_u, crt_vt])
ax.plot([-crt_lim, crt_lim], [-crt_lim, crt_lim], c="gray", ls=":")
ax.scatter(crt_u, crt_vt)
ax.set_xlabel("elements of $U$")
ax.set_ylabel("elements of $V^\\top$")

ax.set_aspect(1)
sns.despine(ax=ax, offset=10)

# %% [markdown] id="YiToQ7Nowdeo"
# ## Check Fourier translation on von-Mises

# %% id="3cXsC0k9whXz"
n = 8
fourier_U = np.array(
    [
        [1 / np.sqrt(n) * np.exp(2 * np.pi * k * l / n * 1j) for k in range(n)]
        for l in range(n)
    ]
)
fourier_V = fourier_U.conj().T

# %% colab={"base_uri": "https://localhost:8080/"} id="fcN6SHaaxzzd" outputId="d27c0358-a4e4-495e-cf60-6584fe9bd2ac"
(
    np.max(np.abs(fourier_U @ fourier_V - np.eye(n))),
    np.max(np.abs(fourier_V @ fourier_U - np.eye(n))),
)


# %% id="86S48q9lyAT-"
def fourier_translate(x, s: float) -> np.ndarray:
    n = len(x)
    # turns out the ** operator has the right branch cut to maket this work
    lbd = np.exp(-(2j * np.pi / n) * np.arange(n)) ** s

    # need to zero out highest frequency mode if n is even
    if n % 2 == 0:
        lbd[n // 2] = 0

    x_f = fourier_V @ x
    x_f_shifted = lbd * x_f
    x_shifted = fourier_U @ x_f_shifted
    
    assert np.max(np.abs(x_shifted.imag)) < 1e-6

    return x_shifted.real


# %% colab={"base_uri": "https://localhost:8080/", "height": 303} id="gjZg3c5AzH2y" outputId="1054ae31-0cec-4d42-a744-f3c2b4b67fa5"
crt_s = 1.7
crt_v0 = np.hstack(([1, 1], np.zeros(n - 2)))
crt_v1 = fourier_translate(crt_v0, crt_s)

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(np.arange(n + 1) / n * 2 * np.pi, 1 + np.hstack((crt_v0, crt_v0[[0]])), c="C0")
ax.plot(np.arange(n + 1) / n * 2 * np.pi, 1 + np.hstack((crt_v1, crt_v1[[0]])), c="C1")

# %% colab={"base_uri": "https://localhost:8080/", "height": 303} id="7aVArdCf4jnG" outputId="d37530e0-8d52-4e47-c07c-db1e7dc31a38"
crt_x0 = 2.7
crt_cos = np.cos(2 * np.pi * (np.arange(n) - crt_x0) / n)
crt_kappa = 1 / 0.5 ** 2
crt_v0 = 0.02 * np.exp(crt_kappa * crt_cos)

crt_s = 1.4
crt_v1 = fourier_translate(crt_v0, crt_s)

# what if we had shifted the von Mises?
crt_cos_tgt = np.cos(2 * np.pi * (np.arange(n) - crt_x0 - crt_s) / n)
crt_v1_tgt = 0.02 * np.exp(crt_kappa * crt_cos_tgt)

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(np.arange(n + 1) / n * 2 * np.pi, 1 + np.hstack((crt_v0, crt_v0[[0]])), c="C0")
ax.plot(np.arange(n + 1) / n * 2 * np.pi, 1 + np.hstack((crt_v1, crt_v1[[0]])), c="C1")
ax.plot(
    np.arange(n + 1) / n * 2 * np.pi,
    1 + np.hstack((crt_v1_tgt, crt_v1_tgt[[0]])),
    c="C1",
    ls="--",
)

# %% colab={"base_uri": "https://localhost:8080/"} id="QPBffJJ85flJ" outputId="3ca36901-1113-4515-cd9c-09c33e745e4b"
np.max(np.abs(crt_v1 - crt_v1_tgt))

# %%
with dv.FigureManager() as (_, ax):
    crt_x = np.linspace(-0.5, 16.0)
    crt_cos = 0.5 * np.cos(crt_x)
    crt_sin = 0.5 * np.sin(crt_x)
    
    for i in range(1, 8):
        ax.axhline(i, ls=":", lw=0.5, c="gray")
        
    ax.plot(crt_x, crt_cos + 7)
    ax.plot(crt_x, np.clip(crt_cos, 0, None) + 6)
    ax.plot(crt_x, -np.clip(crt_cos, None, 0) + 5)
    
    ax.plot(crt_x, crt_sin + 3)
    ax.plot(crt_x, np.clip(crt_sin, 0, None) + 2)
    ax.plot(crt_x, -np.clip(crt_sin, None, 0) + 1)

# %%
