""" Define a simple place-cell simulator in 1d. """

import torch
import numpy as np

from typing import Optional


class PlaceGridMotionSimulator:
    """A simple implementation of motion on a place grid with periodic or
    non-periodic boundary conditions.
    """

    def __init__(
        self,
        n: int,
        x0: Optional[float] = None,
        sigma: float = 2.0,
        periodic: bool = True,
        fourier: bool = True,
    ):
        """Initialize the motion simulator.

        :param n: number of place cells
        :param x0: initial position; default: `n / 2`
        :param sigma: standard deviation of Gaussian place field
        :param periodic: whether to use periodic boundary conditions. If this is true,
            the shape of the bump is given by the von Mises distribution. If this is
            false, the shape is Gaussian.
        :param fourier: whether to use Fourier translation. If this is true, the
            simulation uses Fourier-based translation to move a von Mises distribution
            to the correct position. If `fourier` is false, the simulation instead uses
            a von Mises distribution with a displaced center. These are similar but not
            identical constructions because the von Mises has higher-frequency content
            than the Nyquist limit.
        """
        self.n = n
        self.x = x0 if x0 is not None else self.n / 2
        self.sigma = sigma
        self.periodic = periodic
        self.fourier = fourier

        if self.fourier and not self.periodic:
            raise NotImplementedError(
                "Currently non-periodic only works with non-Fourier."
            )

        # generate a zero-centered bump, using von Mises distribution
        self._kappa = (self.sigma * 2 * np.pi / self.n) ** (-2)
        self._centers = torch.linspace(0, 2 * np.pi, self.n)
        if self.periodic:
            self._prefactor = 1 / (
                2 * np.pi * torch.i0(torch.FloatTensor([self._kappa]))
            )
        else:
            self._prefactor = 1 / np.sqrt(2 * np.pi * self.sigma ** 2)

        if self.fourier:
            exponents = self._kappa * torch.cos(self._centers)
            self._bump = self._prefactor * torch.exp(exponents)

            # generate matrices for Fourier transform
            self._fourier_U = np.array(
                [
                    [
                        1 / np.sqrt(n) * np.exp(2 * np.pi * k * ell / n * 1j)
                        for k in range(n)
                    ]
                    for ell in range(n)
                ]
            )
            self._fourier_V = self._fourier_U.conj().T
        else:
            self._bump = None
            self._fourier_U = None
            self._fourier_V = None

    def __call__(self) -> torch.Tensor:
        """Generate current place-cell activation vector."""
        return self._batch_call(torch.tensor([self.x]))[0]

    def _batch_call(self, x_history: torch.Tensor) -> torch.Tensor:
        N = len(x_history)
        if self.fourier:
            bumps = torch.zeros((N, self.n))
            for i, x in enumerate(x_history):
                # first move bump coarsely, according the integer part of x
                bump = torch.roll(self._bump, int(x))

                # then move finely, using Fourier transform
                bump = self._fourier_shift(bump, x - int(x))

                bumps[i, :] = bump
        else:
            theta = x_history * 2 * np.pi / self.n
            diff = theta[:, None] - self._centers[None, :]
            if self.periodic:
                exponents = self._kappa * torch.cos(diff)
            else:
                exponents = -0.5 * self._kappa * diff ** 2
            bumps = self._prefactor * torch.exp(exponents)

        return bumps

    def move(self, s: float):
        """Shift current position by `s`."""
        if self.periodic:
            self.x = (self.x + s) % self.n
        else:
            self.x = np.clip(self.x + s, 0, self.n)

    def batch(self, s: torch.Tensor):
        """Generate place-cell activation vectors while moving according to a given
        sequence of shifts.

        :param s: sequence of shifts
        :return: tensor of activation vectors, shape `(len(s), self.n)`
        """
        if self.periodic:
            cumul_s = torch.hstack((torch.tensor([0.0]), torch.cumsum(s, dim=0)))
            x_history = (self.x + cumul_s) % self.n

            # last element is future x position
            self.x = x_history[-1]
            x_history = x_history[:-1]
        else:
            x_history = torch.empty(len(s))
            for i, crt_s in enumerate(s):
                x_history[i] = self.x
                self.move(crt_s.item())

        res = self._batch_call(x_history)

        return res

    def _fourier_shift(self, x: torch.Tensor, s: float) -> torch.Tensor:
        """Shift a vector using Fourier transform."""
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
