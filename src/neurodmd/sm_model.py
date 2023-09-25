""" Define a model that uses similarity matching and grid-space equivariance. """

import torch

from typing import Optional, Union


class EquivariantSM:
    """A model that enforces exponential dynamics in grid space while mapping to and
    from place space using a version of similarity matching.

    Attributes
    ==========
    n, m: int
        number of place and grid cells, respectively
    gamma: float
        ratio between equivariance and similarity-matching loss terms
    fast_lr, slow_lr: float
        learning rates for the fast and slow phases, respectively
    x: torch.Tensor
        current (place-cell) input
    z: torch.Tensor
        current grid-cell state
    s: torch.Tensor
        amount of transformation between previous and current state
    W, M: torch.Tensor
        feedforward (place- to grid-cell) and lateral (grid- to grid-cell) connectivity
        matrices. shapes `(m, n)` and `(m, m)`, respectively.
    mu, theta: torch.Tensor
        arrays defining the dynamics in grid space; see below. shapes `((m + 1) // 2,)`
        and `(m // 2,)`, respectively
    fast_callback: list[Callable]
        functions called before every step of the fast iteration, with this model as
        argument

    More specifically, a shift of magnitude `s` is implemented in grid space by the
    transformation `L ** s`, where `L` is a block-diagonal matrix whose entries are set
    by `mu` and `theta`.

    In detail, most of the diagonal blocks defining `L` are 2x2 blocks, and:
        mu: list of `log` of magnitudes for block-diagonal matrices
        theta: list of angles for block-diagonal matrices

    Thus, a 2x2 block is given by:
        exp(mu) * [[cos[theta], -sin[theta]],
                   [sin[theta],  cos[theta]]] .
    This ensures that the magnitude is always positive, for any real `mu`.

    If `m % 2 == 1`, there is one additional element of `self.mu` compared to
    `self.theta`. This corresponds to a single diagonal entry in `L`.

    The conversion from place- to grid-cells is performed by the "fast" dynamics; see
    below.
    """

    def __init__(
        self,
        n: int,
        m: int,
        gamma: float = 1.0,
        fast_lr: float = 0.01,
        slow_lr: float = 0.01,
    ):
        """Initialize the system.

        :param n: number of place cells
        :param m: number of grid cells
        """
        self.n = n
        self.m = m
        self.gamma = gamma
        self.fast_lr = fast_lr
        self.slow_lr = slow_lr

        self.fast_callback = []

        rnd_scale = 0.1

        self.x = torch.zeros(self.n)
        self.z = torch.zeros(self.m)
        self.s = torch.tensor(0.0)

        self.W = torch.eye(m, n) + rnd_scale * torch.randn(m, n)

        M0 = torch.eye(m, m) + rnd_scale * torch.randn(m, m)
        self.M = 0.5 * (M0 + M0.T)

        self.mu = rnd_scale * torch.randn((m + 1) // 2)
        self.theta = rnd_scale * torch.randn(m // 2)

        self.W.requires_grad = True
        self.M.requires_grad = True
        self.mu.requires_grad = True
        self.theta.requires_grad = True

    def set_state(self, z: Optional[torch.Tensor]):
        """Set the (fast-dynamics) state of the circuit.

        :param z: values for the current grid-cell activities
        """
        self.z = z.detach().clone().to(self.z.device)

    def feed(self, x: torch.Tensor, s: Union[torch.tensor, float]):
        """Feed a new sample into the circuit.

        This fixes the input (place-cell activation) and the magnitude of the
        transformation that connects it to the previous time step.

        It also calculates the grid-cell activity that would be predicted based on the
        activity left over from the previous time step,

            h = L ** s @ z ,

        where `L` is the block-diagonal matrix described in the class docstring.

        :param x: values for the current place-cell activities
        :param s: magnitude of transformation that connects current values to previous
            time step
        """
        device = self.x.device
        self.x = x.detach().clone().to(device)

        if torch.is_tensor(s):
            self.s = s.detach().clone().to(device)
        else:
            self.s = torch.tensor(s).to(device)

        lbd_s = self.get_lambda_s_matrix(self.s)
        self.h = lbd_s @ self.z

        lbdp_s = self._get_lambda_prime_s_matrix(self.s)
        self.hp = lbdp_s @ self.z

    def get_lambda_s_matrix(self, s: torch.Tensor) -> torch.Tensor:
        """Generate the matrix that propagates grid cells forward.

        If `self.m % 2 == 0`, this is a block-diagonal matrix made up of 2x2 blocks
            exp(mu * s) * [[cos(s * theta), -sin(s * theta)],
                           [sin(s * theta),  cos(s * theta)]] .

        If `self.m % 2 == 1`, a single diagonal element is added at the bottom-right of
        the matrix, equal to `exp(self.mu[-1])`.
        """
        return self._get_lambda_from_parts(torch.exp(self.mu * s), s * self.theta)

    def online_loss(
        self, x: torch.Tensor, z: torch.Tensor, h: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the online loss function."""
        w_term = torch.sum(self.W**2) - 2 * torch.dot(z, self.W @ x)
        m_term = torch.dot(z, self.M @ z) - 0.5 * torch.sum(self.M**2)

        delta = z - h
        equivar_term = (self.gamma / 2) * torch.sum(delta**2)

        return w_term + m_term + equivar_term

    def fast_step(self, iterations: int = 100):
        """Run the fast dynamics.

        :param iterations: number of iterations to run
        """
        with torch.no_grad():
            for i in range(iterations):
                for f in self.fast_callback:
                    f(self)

                delta = self.z - self.h
                dz = 2 * (self.W @ self.x - self.M @ self.z) - self.gamma * delta

                self.z += self.fast_lr * dz

    def slow_step(self):
        """Run the slow, synaptic-plasticity step."""
        with torch.no_grad():
            self.W += (2 * self.slow_lr) * (torch.outer(self.z, self.x) - self.W)
            self.M += self.slow_lr * (torch.outer(self.z, self.z) - self.M)

            factor = self.slow_lr * self.gamma * self.s.item()
            delta = self.z - self.h
            half_m = len(self.theta)
            for i in range(half_m):
                crt_delta = delta[2 * i : 2 * (i + 1)]
                crt_h = self.h[2 * i : 2 * (i + 1)]
                crt_dot = torch.dot(crt_delta, crt_h)
                self.mu[i] += factor * crt_dot

                crt_hp = self.hp[2 * i : 2 * (i + 1)]
                crt_dot = torch.dot(crt_delta, crt_hp)
                self.theta[i] += factor * crt_dot

            if len(self.mu) > half_m:
                self.mu[-1] += factor * (delta[-1] * self.h[-1])

    def training_step(
        self, x: torch.Tensor, s: Union[torch.tensor, float], **kwargs
    ) -> float:
        """Run one training step, including running the fast dynamics to convergence,
        followed by the slow, synaptic-plasticity step.

        :param x: values for the current place-cell activities
        :param s: magnitude of transformation that connects current values to previous
            time step
        :param **kwargs: additional keyword arguments are passed to `fast_step`
        :return: loss
        """
        self.feed(x, s)
        self.fast_step(**kwargs)

        with torch.no_grad():
            loss = self.online_loss(x, self.z, self.h).item()

        self.slow_step()

        return loss

    @staticmethod
    def _get_lambda_from_parts(rho: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        xs = rho[: len(theta)] * torch.cos(theta)
        ys = rho[: len(theta)] * torch.sin(theta)

        blocks = []
        for i in range(len(theta)):
            crt_block = torch.vstack(
                (torch.hstack((xs[i], -ys[i])), torch.hstack((ys[i], xs[i])))
            )
            blocks.append(crt_block)

        if len(rho) > len(theta):
            blocks.append(rho[-1])

        return torch.block_diag(*blocks)

    @staticmethod
    def _get_lambda_prime_from_parts(
        rho: torch.Tensor, theta: torch.Tensor
    ) -> torch.Tensor:
        xs = rho[: len(theta)] * torch.cos(theta)
        ys = rho[: len(theta)] * torch.sin(theta)

        blocks = []
        for i in range(len(theta)):
            crt_block = torch.vstack(
                (torch.hstack((-ys[i], -xs[i])), torch.hstack((xs[i], -ys[i])))
            )
            blocks.append(crt_block)

        if len(rho) > len(theta):
            blocks.append(rho[-1])

        return torch.block_diag(*blocks)

    def _get_lambda_prime_s_matrix(self, s: torch.Tensor) -> torch.Tensor:
        """Generate the derivative of \\Lambda_s."""
        return self._get_lambda_prime_from_parts(torch.exp(self.mu * s), s * self.theta)
