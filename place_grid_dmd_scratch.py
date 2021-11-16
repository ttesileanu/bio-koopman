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
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import numpy as np

from types import SimpleNamespace
from tqdm import tqdm
from typing import Callable, Collection, Optional, Sequence
from contextlib import ExitStack


# %% [markdown] id="65E_RUocx1r1"
# ## Generic functions

# %% id="MoStKBIVyEzT"
def train(
    model: nn.Module,
    device: torch.device,
    loader: Collection,
    optimizer: torch.optim.Optimizer,
    progress: Optional[Callable] = tqdm,
    test_every: int = 1,
    test_set: Optional[Sequence] = None,
    scheduler: Optional[object] = None,
    scheduler_every: int = 1,
) -> SimpleNamespace:
    """A simple trainer for place/grid systems.

    This uses `model.loss` and automatic gradient calculation to optimize the model.

    :param model: the model to train
    :param device: device on which to train the model
    :param loader: data loader
        This can be any iterable returning triplets of `data`, `target`, and `shift`.
    :param optimizer: PyTorch optimizer for the model
    :param progress: progress bar callable -- must have the `tqdm` interface
    :param test_every: how often to call `test` function (in batches)
    :param test_set: data loader for test set
    :param scheduler: learning rate scheduler; must have `step` method
    :return: namespace containing:
        * train_loss:   loss curve on train set; one value for every batch
            (the following fields only if `test_set` is not `None`:)
        * test_loss:    loss curve on test set; one value every `test_every` batch
        * test_idxs:    batch indices where a test was performed
    """
    # send the model to the correct device
    model = model.to(device)

    # set the module in training mode
    model.train()

    # figure out total number of samples
    if hasattr(loader, "dataset"):
        n_total = len(loader.dataset)
        step = None
    else:
        n_total = len(loader)
        step = 1

    train_loss = []
    test_loss = []
    test_idxs = []

    batch_idx = 0

    with progress(
        total=n_total, postfix="loss: 0", mininterval=0.5
    ) if progress is not None else ExitStack() as pbar:
        for data, target, shift in loader:
            # ensure the tensors are on the right device
            data = data.to(device)
            target = target.to(device)

            # ensure parameters satisfy reality condition
            # XXX this is a bit hacky
            model.project_to_real()

            # start gradient calculation by first setting all of them to zero
            optimizer.zero_grad()

            # run the network and calculate the loss
            loss = model.loss(data, target, shift)
            train_loss.append(loss.item())

            # back-propagate to find gradients
            loss.backward()

            # update parameters
            optimizer.step()

            # update progress bar
            if progress is not None:
                pbar.postfix = f"train batch loss: {loss.item():.6f}"
                pbar.update(len(data) if step is None else step)
            
            # test on test set, if any given
            if test_set is not None and batch_idx % test_every == 0:
                l = test(model, device, test_set, progress=None)
                test_loss.append(l)
                test_idxs.append(batch_idx)
            
            # adjust step size, if scheduler given
            if scheduler is not None:
                scheduler.step()
            
            # keep batch index sync'd up
            batch_idx += 1

    res = SimpleNamespace(train_loss=train_loss)
    if test_set is not None:
        res.test_loss = test_loss
        res.test_idxs = test_idxs

    return res


# %% id="jySdY5-w1a14"
def test(
    model: nn.Module,
    device: torch.device,
    loader: Collection,
    progress: Optional[Callable] = tqdm,
) -> float:
    """A simple tester for place/grid systems.

    This uses `model.loss`.

    :param model: the model to test
    :param device: device on which to test the model
    :param loader: data loader
        This can be any iterable returning triplets of `data`, `target`, and `shift`.
    :param progress: progress bar callable -- must have the `tqdm` interface
    :return: mean loss per batch on test set
    """
    # send the model to the correct device
    model = model.to(device)

    # set the module in evaluation mode
    model.eval()

    # figure out total number of samples
    if hasattr(loader, "dataset"):
        n_total = len(loader.dataset)
        step = None
    else:
        n_total = len(loader)
        step = 1

    # initialize loss tracker
    total_loss = 0
    count = 0

    with torch.no_grad():
        with progress(
            total=n_total, postfix="loss: 0", mininterval=0.5
        ) if progress is not None else ExitStack() as pbar:
            for data, target, shift in loader:
                # ensure the tensors are on the right device
                data = data.to(device)
                target = target.to(device)

                # run the network and calculate the loss
                loss = model.loss(data, target, shift)

                # keep track of loss and total samples
                total_loss += loss.item()
                count += 1

                # update progress bar
                if progress is not None:
                    pbar.postfix = f"test batch loss: {loss.item():.6f}"
                    pbar.update(len(data) if step is None else step)

    avg_loss = total_loss / count

    return avg_loss


# %% [markdown] id="zP_r7TYb2BWx"
# ## Dynamical systems

# %% id="fk86s7W52GUw"
class PlaceGridMotionSimulator:
    """ A simple implementation of motion on a place grid with periodic boundary
    conditions.
    """
    def __init__(self, n: int, x0: Optional[float] = None, sigma: float = 2.0):
        """ Initialize the motion simulator.

        :param n: number of place cells
        :param x0: initial position; default: `n / 2`
        :param sigma: standard deviation of Gaussian place field
        """
        self.n = n
        self.x = x0 if x0 is not None else self.n / 2
        self.sigma = sigma

        # generate a zero-centered bump, using von Mises distribution
        self._kappa = (self.sigma * 2 * np.pi / self.n) ** (-2)
        self._centers = torch.linspace(0, 2 * np.pi, self.n) + np.pi / self.n
        self._prefactor = 1 / (2 * np.pi * torch.i0(torch.FloatTensor([self._kappa])))

        exponents = self._kappa * torch.cos(self._centers)
        self._bump = self._prefactor * torch.exp(exponents)

        # generate matrices for Fourier transform
        self._fourier_U = np.array(
            [
                [1 / np.sqrt(n) * np.exp(2 * np.pi * k * l / n * 1j) for k in range(n)]
                for l in range(n)
            ]
        )
        self._fourier_V = self._fourier_U.conj().T
    
    def __call__(self) -> torch.Tensor:
        """ Generate current place-cell activation vector. """
        # first move bump coarsely, according the integer part of x
        bump = torch.roll(self._bump, int(self.x))

        # then move finely, using Fourier transform
        bump = self._fourier_shift(bump, self.x - int(self.x))
        return bump
    
    def move(self, s: float):
        """ Shift current position by `s`. """
        self.x = (self.x + s) % self.n

    def batch(self, s: torch.Tensor):
        """ Generate place-cell activation vectors while moving according to a given
        sequence of shifts.

        :param s: sequence of shifts
        :return: tensor of activation vectors, shape `(len(s), self.n)`
        """
        res = torch.zeros(len(s), self.n)
        for i, crt_s in enumerate(s):
            res[i, :] = self()
            self.move(crt_s.item())
        
        return res

    def _fourier_shift(self, x: torch.Tensor, s: float) -> np.ndarray:
        """ Shift a vector using Fourier transform. """
        # turns out the ** operator has the right branch cut to make this work
        lbd = np.exp(-(2j * np.pi / self.n) * np.arange(self.n)) ** s

        # need to zero out highest frequency mode if n is even
        if self.n % 2 == 0:
            lbd[self.n // 2] = 0

        x_f = self._fourier_V @ x.detach().numpy()
        x_f_shifted = lbd * x_f
        x_shifted = self._fourier_U @ x_f_shifted
        
        assert np.max(np.abs(x_shifted.imag)) < 1e-6

        return torch.from_numpy(x_shifted.real)


# %% [markdown] id="f-yoM2yFx4Rm"
# ## Network models

# %% id="gqliVXkh0SwA"
class PlaceGridSystemNonBioCplx:
    """ A complex-number based, non-biological implementation of our place&grid cell
    system.
    """
    def __init__(self, n: int, m: int):
        """ Initialize the system.
    
        :param n: number of place cells
        :param m: number of grid cells; if it is even, all eigenvalues and eigenvectors
                  come in complex-conjugate pairs; otherwise, the first eigenvalue and
                  eigenvector are real and non-negative, and real, respectively.
        """
        # assert m % 2 == 0

        self.n = n
        self.m = m

        rnd_scale = 0.1

        m_half = m // 2
        U_half = (
            torch.eye(n, m_half) + 
            rnd_scale * torch.view_as_complex(torch.normal(torch.zeros(n, m_half, 2)))
        )
        V_half = (
            torch.eye(m_half, n) +
            rnd_scale * torch.view_as_complex(torch.normal(torch.zeros(m_half, n, 2)))
        )
        lbd_half = (
            1 +
            rnd_scale * torch.view_as_complex(torch.normal(torch.zeros(m_half, 2)))
        )

        # the largest even number smaller equal to m
        me = 2 * m_half
        self.U = torch.stack((U_half, U_half.conj()), dim=2).view(n, me).contiguous()
        self.V = torch.stack((V_half, V_half.conj()), dim=1).view(me, n).contiguous()
        self.lbd = torch.stack((lbd_half, lbd_half.conj()), dim=1).view(me).contiguous()
        
        if m % 2 == 1:
            # add the real eigenvalue & eigenvector
            self.U = torch.hstack((torch.normal(torch.zeros(n, 1)), self.U))
            self.V = torch.vstack((torch.normal(torch.zeros(1, n)), self.V))
            self.lbd = torch.hstack((torch.FloatTensor([1.0]), self.lbd))

        self.U.requires_grad = True
        self.V.requires_grad = True
        self.lbd.requires_grad = True
    
    def to_grid(self, x: torch.Tensor) -> torch.Tensor:
        """ Convert batch of inputs to grid basis.

        This multiplies by `U`.

        :param x: tensor to transform; first dimension: batch index
        :return: transformed tensor
        """
        return (x + 0j) @ self.U

    def from_grid(
        self, z: torch.Tensor, atol=1e-4, rtol=1e-2, return_real: bool = True
    ) -> torch.Tensor:
        """ Convert batch of grid responses to place basis.

        This multiplies by `V`, ensures that the imaginary part is close to zero (and
        raises assertion otherwise), and converts to real.

        The imaginary part is checked to obey
            |imag| <= atol + rtol * |abs| ,
        where |abs| is the complex magnitude.

        :param z: tensor to transform; first dimension: batch index
        :param atol: absolute tolerance for imaginary part of output
        :param rtol: relative tolerance for imaginary part of output
        :param return_real: if `False`, it returns the full, complex-valued tensor
        :return: transformed tensor
        """
        y_pred = z @ self.V

        if return_real:
            max_abs_imag = torch.max(torch.abs(torch.imag(y_pred)))
            max_rel_imag = torch.max(
                torch.abs(torch.imag(y_pred)) / (torch.abs(y_pred) + 1e-12)
            )

            check_v = torch.imag(y_pred) <= atol + rtol * torch.abs(y_pred)
            if not torch.all(check_v):
                print(
                    f"max(abs(imag))={max_abs_imag:.2g}, "
                    f"max(rel(imag))={max_rel_imag:.2g}"
                )

            return y_pred.real
        else:
            return y_pred
    
    def propagate_grid(self, z: torch.Tensor, s: float) -> torch.Tensor:
        """ Propagate forward in time in grid basis.

        This multiplies by `diag(lbd ** s)`.

        :param z: tensor to propagate forward in time; first dimension: batch index
        :param s: amount by which to propagate
        :return: propagated tensor
        """
        return z * torch.tile(self.lbd, (len(s), 1)) ** s[:, None]
    
    def propagate_place(self, x: torch.Tensor, s: float) -> torch.Tensor:
        """ Propagate forward in time in place basis.

        :param x: tensor to propagate forward in time; first dimension: batch index
        :param s: amount by which to propagate
        :return: propagated tensor
        """
        return self.from_grid(self.propagate_grid(self.to_grid(x), s))
    
    def loss(self, x: torch.Tensor, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """ Calculate quadratic loss on a batch of input, outputs, and controls.

        First dimension: batch index.

        :param x: initial values in place basis
        :param y: final values in place basis
        :param s: amount of shift (control) for each sample.
        :return: mean losses over the batch
        """
        y_pred = self.propagate_place(x, s)
        return 0.5 * torch.mean((y - y_pred) ** 2)
    
    def parameters(self) -> list:
        """ Get list of parameters that are being optimized. """
        return [self.U, self.V, self.lbd]
    
    def zero_grad(self):
        """ Set gradients of all parameters to zero. """
        for param in self.parameters():
            if param.grad is not None:
                param.grad.zero_()
    
    def to(self, device: torch.device) -> "PlaceGridSystemNonBioCplx":
        """ Send all parameters to the given device. """
        for param in self.parameters():
            param.to(device)
        
        return self
    
    def train(self):
        """ Set model to training mode. (Does nothing for now). """
        pass

    def eval(self):
        """ Set model to evaluation (test) mode. (Does nothing for now). """
        pass    

    def project_to_real(self, atol=1.0):
        """ Ensure predictions stay real by enforcing conjugation symmetry among pairs
        of eigenvalues and eigenvectors.

        For `m` even, this ensures that each consecutive pair of eigenvalues, as well as
        each consecutive pair of eigenvectors (columns of `U` and rows of `V`) are
        complex conjugates of each other. After checking that these are within a given
        tolerance, the function then sets them equal to a consensus value obtained by
        averaging. That is, the pair of `(a, b)` is replaced by
            (0.5 * (a + b.conj()), 0.5 * (b + a.conj())) .

        If `m` is odd, the first eigenvalue and eigenvector are checked to be real,
        within a tolerance. Their imaginary part is then set to zero. The eigenvalue is
        also checked to be non-negative, within a tolerance, and is clipped from below
        at zero.
        
        :param atol: absolute tolerance for conjugacy
        """
        i0 = self.m % 2
        assert torch.allclose(
            self.U[:, i0 : : 2], self.U[:, i0 + 1 : : 2].conj(), atol=atol
        )
        assert torch.allclose(
            self.V[i0 : : 2, :], self.V[i0 + 1 : : 2, :].conj(), atol=atol
        )
        assert torch.allclose(
            self.lbd[i0 : : 2], self.lbd[i0 + 1 : : 2].conj(), atol=atol
        )

        with torch.no_grad():
            U_half = 0.5 * (self.U[:, i0 : : 2] + self.U[:, i0 + 1 : : 2].conj())
            V_half = 0.5 * (self.V[i0 : : 2, :] + self.V[i0 + 1 : : 2, :].conj())
            lbd_half = 0.5 * (self.lbd[i0 : : 2] + self.lbd[i0 + 1 : : 2].conj())

            me = 2 * (self.m // 2)

            self.U[:, i0:] = torch.stack((U_half, U_half.conj()), dim=2).view(n, me)
            self.V[i0:, :] = torch.stack((V_half, V_half.conj()), dim=1).view(me, n)
            self.lbd[i0:] = torch.stack((lbd_half, lbd_half.conj()), dim=1).view(me)

            if self.m % 2 == 1:
                assert torch.max(torch.abs(self.U[:, 0].imag)) <= atol
                assert torch.max(torch.abs(self.V[0, :].imag)) <= atol
                assert torch.abs(self.lbd[0].imag) <= atol

                self.U[:, 0] = self.U[:, 0].real.clone()
                self.V[0, :] = self.V[0, :].real.clone()
                self.lbd[0] = self.lbd[0].real.clone()

                if self.lbd[0].real < 0:
                    # assert self.lbd[0].real >= -atol
                    self.lbd[0] = 0


# %% id="sulIw-LCqKbk"
class PlaceGridSystemNonBio:
    """ A real-valued, non-biological implementation of our place&grid cell system.
    
    self.xi: list of `arcsech` of magnitudes for block-diagonal matrices
    self.theta: list of angles for block-diagonal matrices

    Thus, a 2x2 block is given by:
        sech(xi) * [[cos[theta], -sin[theta]],
                    [sin[theta],  cos[theta]]] .
    This ensures that the magnitude is always between 0 and 1, for any real `xi`.

    If `m % 2 == 1`, there is one additional element of `self.xi` compared to
    `self.theta`. This corresponds to a single diagonal entry.
    """
    def __init__(self, n: int, m: int):
        """ Initialize the system.
    
        :param n: number of place cells
        :param m: number of grid cells
        """
        self.n = n
        self.m = m

        rnd_scale = 0.1

        self.U = torch.eye(n, m) + rnd_scale * torch.randn(n, m)
        self.V = torch.eye(m, n) + rnd_scale * torch.randn(m, n)

        self.xi = rnd_scale * torch.randn((m + 1) // 2)
        self.theta = rnd_scale * torch.randn(m // 2)

        self.U.requires_grad = True
        self.V.requires_grad = True
        self.xi.requires_grad = True
        self.theta.requires_grad = True
    
    def to_grid(self, x: torch.Tensor) -> torch.Tensor:
        """ Convert batch of inputs to grid basis.

        This multiplies by `U`.

        :param x: tensor to transform; first dimension: batch index
        :return: transformed tensor
        """
        return x @ self.U

    def from_grid(self, z: torch.Tensor) -> torch.Tensor:
        """ Convert batch of grid responses to place basis.

        This multiplies by `V`.

        :param z: tensor to transform; first dimension: batch index
        :return: transformed tensor
        """
        y_pred = z @ self.V
        return y_pred
    
    def propagate_grid(self, z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """ Propagate forward in time in grid basis.

        This multiplies by `Lambda ** s`, where `Lambda ** s` is calculated by
        `self.get_lambda_s_matrix`.

        :param z: tensor to propagate forward in time; first dimension: batch index
        :param s: amount by which to propagate
        :return: propagated tensor
        """
        z_prop_lst = []
        for crt_z, crt_s in zip(z, s):
            crt_grid_prop = self.get_lambda_s_matrix(crt_s)
            crt_z_prop = crt_z @ crt_grid_prop
            z_prop_lst.append(crt_z_prop)

        return torch.vstack(z_prop_lst)
    
    def propagate_place(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """ Propagate forward in time in place basis.

        :param x: tensor to propagate forward in time; first dimension: batch index
        :param s: amount by which to propagate
        :return: propagated tensor
        """
        return self.from_grid(self.propagate_grid(self.to_grid(x), s))
    
    def loss(self, x: torch.Tensor, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """ Calculate quadratic loss on a batch of input, outputs, and controls.

        First dimension: batch index.

        :param x: initial values in place basis
        :param y: final values in place basis
        :param s: amount of shift (control) for each sample.
        :return: mean losses over the batch
        """
        y_pred = self.propagate_place(x, s)
        return 0.5 * torch.mean((y - y_pred) ** 2)
    
    def parameters(self) -> list:
        """ Get list of parameters that are being optimized. """
        return [self.U, self.V, self.xi, self.theta]
    
    def zero_grad(self):
        """ Set gradients of all parameters to zero. """
        for param in self.parameters():
            if param.grad is not None:
                param.grad.zero_()
    
    def to(self, device: torch.device) -> "PlaceGridSystemNonBioCplx":
        """ Send all parameters to the given device. """
        for param in self.parameters():
            param.to(device)
        
        return self
    
    def train(self):
        """ Set model to training mode. (Does nothing for now). """
        pass

    def eval(self):
        """ Set model to evaluation (test) mode. (Does nothing for now). """
        pass

    def project_to_real(self):
        pass
    
    @staticmethod
    def _get_lambda_from_parts(rho: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        blocks = []
        for crt_rho, crt_theta in zip(rho, theta):
            crt_a = crt_rho * torch.cos(crt_theta)
            crt_b = crt_rho * torch.sin(crt_theta)
            crt_block = torch.FloatTensor([[crt_a, -crt_b], [crt_b, crt_a]])
            blocks.append(crt_block)
        
        if len(rho) > len(theta):
            blocks.append(rho[-1])
        
        return torch.block_diag(*blocks)

    def get_lambda_s_matrix(self, s: float) -> torch.Tensor:
        """ Generate the matrix that propagates grid cells forward.
        
        If `self.m % 2 == 0`, this is a block-diagonal matrix made up of 2x2 blocks
            rho ** s * [[cos(s * theta), -sin(s * theta)],
                        [sin(s * theta),  cos(s * theta)]] .
        `rho` is `sech(self.xi)`.

        If `self.m % 2 == 1`, a single diagonal element is added at the bottom-right of
        the matrix, equal to `sech(self.xi[-1])`.
        """
        return self._get_lambda_from_parts(1 / torch.cosh(self.xi) ** s, s * self.theta)


# %% [markdown] id="himLi2WNQE6t"
# ## Unit tests for network models

# %% [markdown] id="lSbNwnZqX6EJ"
# ### Complex-valued version

# %% id="-igvvdX6JEC6"
def test_loss_averages_over_samples():
    torch.manual_seed(0)

    n = 10
    m = 6
    samples = 3

    x = torch.normal(torch.zeros(samples, n))
    y = x + 0.1 * torch.normal(torch.zeros(samples, n))
    s = torch.normal(torch.zeros(samples))

    system = PlaceGridSystemNonBioCplx(n, m)

    all_l = [system.loss(x[[i]], y[[i]], s[[i]]) for i in range(samples)]

    l = system.loss(x, y, s)

    assert torch.allclose(l, sum(all_l) / samples), "wrong multi-sample loss"
    print("ok")


# %% colab={"base_uri": "https://localhost:8080/"} id="_gDtovnXJQXH" outputId="fddbc79d-4a75-4f02-8502-e0ff5fd4d6f9"
test_loss_averages_over_samples()


# %% id="F7gz4gMb2_to"
def test_u_derivative():
    torch.manual_seed(0)

    n = 10
    m = 6
    samples = 1

    x = torch.normal(torch.zeros(samples, n))
    y = x + 0.1 * torch.normal(torch.zeros(samples, n))
    s = torch.normal(torch.zeros(samples))

    system = PlaceGridSystemNonBioCplx(n, m)

    system.zero_grad()
    l = system.loss(x, y, s)

    l.backward()

    # lbd_s_mat = torch.diag(system.lbd) ** s[:, None, None]
    lbd_s_mat = torch.stack([torch.diag(system.lbd ** _) for _ in s])
    y_pred = (x + 0j) @ system.U @ lbd_s_mat @ system.V
    eps = (y - y_pred)[0]

    assert torch.allclose(torch.imag(eps), torch.FloatTensor([0]), atol=1e-7), "eps complex?"

    exp_grad_u = -(x + 0j).T @ eps @ system.V.T.conj() @ lbd_s_mat[0].conj() / n

    assert torch.allclose(system.U.grad, exp_grad_u), "U.grad wrong"
    print("ok")


# %% colab={"base_uri": "https://localhost:8080/"} id="udyXs__t4zw9" outputId="a695435e-7bc6-446f-d0b9-07dd6c124ac2"
test_u_derivative()


# %% id="bbWZvDbdMmC9"
def test_v_derivative():
    torch.manual_seed(1)

    n = 10
    m = 6
    samples = 1

    x = torch.normal(torch.zeros(samples, n))
    y = x + 0.1 * torch.normal(torch.zeros(samples, n))
    s = torch.normal(torch.zeros(samples))

    system = PlaceGridSystemNonBioCplx(n, m)

    system.zero_grad()
    l = system.loss(x, y, s)

    l.backward()

    # lbd_s_mat = torch.diag(system.lbd) ** s[:, None, None]
    lbd_s_mat = torch.stack([torch.diag(system.lbd ** _) for _ in s])
    y_pred = (x + 0j) @ system.U @ lbd_s_mat @ system.V
    eps = (y - y_pred)[0]

    assert torch.allclose(torch.imag(eps), torch.FloatTensor([0])), "eps complex?"

    exp_grad_v = -lbd_s_mat.conj() @ system.U.T.conj() @ (x + 0j).T @ eps / n

    assert torch.allclose(system.V.grad, exp_grad_v), "V.grad wrong"
    print("ok")


# %% colab={"base_uri": "https://localhost:8080/"} id="BdsdwW2xtjgg" outputId="3098ed27-6a76-42f6-8433-c9fc37c4643e"
test_v_derivative()


# %% id="nTFEUlxdtkr6"
def test_lbd_derivative():
    torch.manual_seed(2)

    n = 10
    m = 6
    samples = 1

    x = torch.normal(torch.zeros(samples, n))
    y = x + 0.1 * torch.normal(torch.zeros(samples, n))
    s = torch.normal(torch.zeros(samples))

    system = PlaceGridSystemNonBioCplx(n, m)

    system.zero_grad()
    l = system.loss(x, y, s)

    l.backward()

    # lbd_s_mat = torch.diag(system.lbd) ** s[:, None, None]
    lbd_s_mat = torch.stack([torch.diag(system.lbd ** _) for _ in s])
    y_pred = (x + 0j) @ system.U @ lbd_s_mat @ system.V
    eps = (y - y_pred)[0]

    assert torch.allclose(torch.imag(eps), torch.FloatTensor([0])), "eps complex?"

    exp_grad_lbd = -s[0] * (system.lbd ** (s[0] - 1)).conj() * torch.diag(
        system.V @ eps.T @ (x + 0j) @ system.U).conj() / n

    assert torch.allclose(system.lbd.grad, exp_grad_lbd), "lbd.grad wrong"
    print("ok")


# %% colab={"base_uri": "https://localhost:8080/"} id="1Vu7ZsMUwCil" outputId="696ab2cb-48cd-46fe-ab63-d11247ad398e"
test_lbd_derivative()


# %% id="WAsjoImNL4d_"
def test_lbd_derivative_again():
    torch.manual_seed(2)

    n = 10
    m = 6
    samples = 1

    x = torch.normal(torch.zeros(samples, n))
    y = x + 0.1 * torch.normal(torch.zeros(samples, n))
    s = torch.normal(torch.zeros(samples))

    system = PlaceGridSystemNonBioCplx(n, m)

    system.zero_grad()
    l = system.loss(x, y, s)

    l.backward()

    g = (x + 0j) @ system.U
    lbd_s_mat = torch.stack([torch.diag(system.lbd ** _) for _ in s])
    y_pred = g @ lbd_s_mat @ system.V
    eps = (y - y_pred)[0]

    assert torch.allclose(torch.imag(eps), torch.FloatTensor([0])), "eps complex?"

    vt_eps = eps @ system.V.T.conj()

    exp_grad_lbd = -s[0] * (system.lbd ** (s[0] - 1) * g).conj() * vt_eps / n

    assert torch.allclose(system.lbd.grad, exp_grad_lbd), "lbd.grad wrong"
    print("ok")


# %% colab={"base_uri": "https://localhost:8080/"} id="8geS780wOd12" outputId="266c0781-aa15-4b3b-8cf9-48822e0e28a9"
test_lbd_derivative_again()


# %% [markdown] id="iebpUAstX8h6"
# ### Real-valued version

# %% id="BaZbZ8w-ZgkR"
def test_loss_averages_over_samples():
    torch.manual_seed(0)

    n = 10
    m = 6
    samples = 3

    x = torch.normal(torch.zeros(samples, n))
    y = x + 0.1 * torch.normal(torch.zeros(samples, n))
    s = torch.normal(torch.zeros(samples))

    system = PlaceGridSystemNonBio(n, m)

    all_l = [system.loss(x[[i]], y[[i]], s[[i]]) for i in range(samples)]

    l = system.loss(x, y, s)

    assert torch.allclose(l, sum(all_l) / samples), "wrong multi-sample loss"
    print("ok")


# %% colab={"base_uri": "https://localhost:8080/"} id="O7KQoymtZii9" outputId="962466cb-2352-469a-a8a0-71589a3e1ea9"
test_loss_averages_over_samples()


# %% id="9YKxFn_7c1X5"
def test_lambda_s_matrix_is_size_m():
    torch.manual_seed(0)
    n = 9
    m = 7
    samples = 1

    x = torch.normal(torch.zeros(samples, n))
    y = x + 0.1 * torch.normal(torch.zeros(samples, n))
    s = torch.normal(torch.zeros(samples))

    system = PlaceGridSystemNonBio(n, m)

    lbd_s = system.get_lambda_s_matrix(s.item())

    assert lbd_s.shape == (m, m), "wrong Lambda ** s shape"
    print("ok")


# %% colab={"base_uri": "https://localhost:8080/"} id="eyFk4YCPdmjx" outputId="1a8e0680-b1ba-45bb-a2cf-63bef53055dc"
test_lambda_s_matrix_is_size_m()


# %% id="4FfXiuwodtO8"
def test_lambda_s_matrix_is_block_diagonal():
    torch.manual_seed(1)
    n = 9
    m = 7
    samples = 1

    x = torch.normal(torch.zeros(samples, n))
    y = x + 0.1 * torch.normal(torch.zeros(samples, n))
    s = torch.normal(torch.zeros(samples))

    system = PlaceGridSystemNonBio(n, m)

    lbd_s = system.get_lambda_s_matrix(s.item())

    tzero = torch.FloatTensor([0])
    for i in range(0, 2 * (m // 2), 2):
        crt_rows = lbd_s[[i, i + 1], :]
        crt_rows[:, [i, i + 1]] = 0

        assert torch.allclose(crt_rows, tzero), "spurious non-zeros in some rows"

        crt_cols = lbd_s[:, [i, i + 1]]
        crt_cols[[i, i + 1], :] = 0
        assert torch.allclose(crt_cols, tzero), "spurious non-zeros in some columns"
    
    assert torch.allclose(lbd_s[-1, :-1], tzero)
    assert torch.allclose(lbd_s[:-1, -1], tzero)

    print("ok")


# %% colab={"base_uri": "https://localhost:8080/"} id="imW0lNELdvwp" outputId="a7a26011-0203-4979-ee9d-4c70d5b0e323"
test_lambda_s_matrix_is_block_diagonal()


# %% id="siG1JeLQfiwS"
def test_lambda_s_2x2_blocks_match_xi_and_theta():
    torch.manual_seed(2)
    n = 9
    m = 7
    samples = 1

    x = torch.normal(torch.zeros(samples, n))
    y = x + 0.1 * torch.normal(torch.zeros(samples, n))
    s = torch.normal(torch.zeros(samples))

    system = PlaceGridSystemNonBio(n, m)

    lbd_s = system.get_lambda_s_matrix(s.item())

    rho = 1 / torch.cosh(system.xi) ** s.item()
    theta = s.item() * system.theta

    for i in range(0, 2 * (m // 2), 2):
        crt_block = lbd_s[[i, i + 1], :][:, [i, i + 1]]
        crt_a = rho[i // 2] * torch.cos(theta[i // 2])
        crt_b = rho[i // 2] * torch.sin(theta[i // 2])

        crt_exp = torch.FloatTensor([[crt_a, -crt_b], [crt_b, crt_a]])
        assert torch.allclose(crt_block, crt_exp), f"block {i} wrong"

    print("ok")


# %% colab={"base_uri": "https://localhost:8080/"} id="Ulk2LC_9f9fX" outputId="999e6417-e4c0-4e1b-d8dc-422dca6ce091"
test_lambda_s_2x2_blocks_match_xi_and_theta()


# %% id="xSh5IKNlgPqc"
def test_lambda_s_diagonal_block_matches_xi():
    torch.manual_seed(3)
    n = 9
    m = 7
    samples = 1

    x = torch.normal(torch.zeros(samples, n))
    y = x + 0.1 * torch.normal(torch.zeros(samples, n))
    s = torch.normal(torch.zeros(samples))

    system = PlaceGridSystemNonBio(n, m)

    lbd_s = system.get_lambda_s_matrix(s.item())

    rho = 1 / torch.cosh(system.xi) ** s.item()

    assert torch.allclose(lbd_s[-1, -1], rho[-1]), "diag value mismatch"

    print("ok")


# %% colab={"base_uri": "https://localhost:8080/"} id="RmGV7cvzgYPH" outputId="f1fbf2c2-6e27-4051-8a9e-22d4e27d47ac"
test_lambda_s_diagonal_block_matches_xi()


# %% id="EjM-I2dEges5"
def test_propagate_place_matches_expectation_multi_sample():
    torch.manual_seed(0)

    n = 10
    m = 6
    samples = 3

    x = torch.normal(torch.zeros(samples, n))
    y = x + 0.1 * torch.normal(torch.zeros(samples, n))
    s = torch.normal(torch.zeros(samples))

    system = PlaceGridSystemNonBio(n, m)
    y_hat = system.propagate_place(x, s)

    y_hat_exp = torch.zeros_like(y_hat)
    for i in range(samples):
        crt_lbd_s = system.get_lambda_s_matrix(s[i].item())
        crt_x = x[i]
        crt_y_hat_exp = crt_x @ system.U @ crt_lbd_s @ system.V
        y_hat_exp[i] = crt_y_hat_exp

    assert torch.allclose(y_hat, y_hat_exp), "wrong output from propagate_place"
    
    print("ok")


# %% colab={"base_uri": "https://localhost:8080/"} id="fV-fnlsmhE10" outputId="0dfaed28-9a2d-4e6b-e547-677772a844f5"
test_propagate_place_matches_expectation_multi_sample()


# %% id="tUc4UM1Fhc7x"
def test_loss_is_correct():
    torch.manual_seed(1)

    n = 10
    m = 6
    samples = 3

    x = torch.normal(torch.zeros(samples, n))
    y = x + 0.1 * torch.normal(torch.zeros(samples, n))
    s = torch.normal(torch.zeros(samples))

    system = PlaceGridSystemNonBio(n, m)
    y_hat = system.propagate_place(x, s)

    loss = system.loss(x, y, s)
    loss_exp = 0.5 * torch.mean((y_hat - y) ** 2)

    assert torch.allclose(loss, loss_exp), "wrong loss output"

    print("ok")


# %% colab={"base_uri": "https://localhost:8080/"} id="r1Rb_wUthf6U" outputId="96c3846e-45eb-4f27-f351-2769f5aee6af"
test_loss_is_correct()

# %% [markdown] id="n1dz8V341gMS"
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
# ## Try some learning

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
dataset_test = dataset_full[test_size:]

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

# %% [markdown] id="An9kiTssb1ZN"
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
def fourier_translate(x: Sequence, s: float) -> np.ndarray:
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

# %% [markdown] id="pQU-ZrMTb45r"
# ### Check that a model initialized at Fourier solution yields (almost) zero loss -- complex-valued version

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
# ### Sanity check: projection to valid parameters does not spoil global optimum

# %% id="i50nRq2ZZcZ8"
crt_old_U = torch.clone(test_system.U)
crt_old_V = torch.clone(test_system.V)
crt_old_lbd = torch.clone(test_system.lbd)

test_system.project_to_real()

assert torch.allclose(crt_old_U, test_system.U)
assert torch.allclose(crt_old_V, test_system.V)
assert torch.allclose(crt_old_lbd, test_system.lbd)

# %% [markdown] id="ULf2KRNTYjtr"
# ### Sanity check: gradient (almost) zero at global optimum

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
# ### Sanity check: Adam optimization does not lead us away from global optimum

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


# %% [markdown] id="K9I6NTxOb-dK"
# ### Run the learning, check outcome

# %% id="EpvZawUXtHJg"
class StepwiseScheduler:
    def __init__(self, optimizer, sequence: Sequence):
        assert len(optimizer.param_groups) == 1

        self.optimizer = optimizer
        self.sequence = np.copy(sequence)
        self._borders = np.cumsum([_[0] for _ in self.sequence])

        self._step_count = 0
        self._last_lr = None

        self.step()
    
    def get_last_lr(self) -> list:
        return self._last_lr

    def get_lr(self) -> list:
        mask = self._step_count <= self._borders
        if not np.any(mask):
            lr = [self.sequence[-1][1]]
        else:
            idx = mask.nonzero()[0][0]
            lr = [self.sequence[idx][1]]
        
        return lr
    
    def step(self):
        self._step_count += 1
        lr = self.get_lr()
        for group, crt_lr in zip(self.optimizer.param_groups, lr):
            group["lr"] = crt_lr
        
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]


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
# ### Try learned system on examples

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
dataset_test = dataset_full[test_size:]

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
def fourier_translate(x: Sequence, s: float) -> np.ndarray:
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

# %% id="Y554o0Ps7LJx"
