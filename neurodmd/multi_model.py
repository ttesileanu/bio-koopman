""" Define a multi-controller version of the place-grid-cell network. This requires
backprop for training. """

import torch


class PlaceGridMultiNonBio:
    """A real-valued, non-biological implementation of our place&grid cell system with
    multiple controllers.

    This allows multiple controllers to act at once.

    Attributes
    ==========
    n_ctrl: int
        number of different controllers
    n, m: int
        number of place cells and number of grid cells per controller, respectively
    U, V: torch.Tensor
        arrays defining the conversion from place to grid cells and from grid to place
        cells, respectively; shapes (n_ctrl, m, n) and (n_ctrl, n, m), respectively
    xi, theta: torch.Tensor
        arrays defining the dynamics in grid space; see below. shapes
        (n_ctrl, (m + 1) // 2) and (n_ctrl, m // 2), respectively

    More specifically, a shift of magnitude s is implemented by the transformation
            mean_k(V[k] @ L[k] ** s[k] @ U[k]) @ x ,
    where x is the input sample, and L[k] is a block-diagonal matrix whose entries are
    set by xi[k] and theta[k].

    In detail, most of the diagonal blocks defining L are 2x2 blocks, and:
        xi: list of `arcsech` of magnitudes for block-diagonal matrices
        theta: list of angles for block-diagonal matrices

    Thus, a 2x2 block is given by:
        sech(xi) * [[cos[theta], -sin[theta]],
                    [sin[theta],  cos[theta]]] .
    This ensures that the magnitude is always between 0 and 1, for any real `xi`.

    If `m % 2 == 1`, there is one additional element of `self.xi` compared to
    `self.theta`. This corresponds to a single diagonal entry in L.
    """

    def __init__(self, n_ctrl: int, n: int, m: int):
        """Initialize the system.

        :param n_ctrl: number of controllers
        :param n: number of place cells
        :param m: number of grid cells
        """
        self.n_ctrl = n_ctrl
        self.n = n
        self.m = m

        self.U = torch.zeros((n_ctrl, m, n))
        self.V = torch.zeros((n_ctrl, n, m))
        self.xi = torch.zeros((n_ctrl, (m + 1) // 2))
        self.theta = torch.zeros((n_ctrl, m // 2))

        rnd_scale = 0.1

        for k in range(n_ctrl):
            self.U[k, ...] = torch.eye(m, n) + rnd_scale * torch.randn(m, n)
            self.V[k, ...] = torch.eye(n, m) + rnd_scale * torch.randn(n, m)

            self.xi[k, :] = rnd_scale * torch.randn((m + 1) // 2)
            self.theta[k, :] = rnd_scale * torch.randn(m // 2)

        self.U.requires_grad = True
        self.V.requires_grad = True
        self.xi.requires_grad = True
        self.theta.requires_grad = True

    def to_grid(self, x: torch.Tensor) -> torch.Tensor:
        """Convert batch of inputs to grid basis.

        This multiplies by `U`.

        :param x: tensor to transform; shape `(n_batch, n)`
        :return: transformed tensor; shape `(n_batch, n_ctrl, m)`
        """
        # need to add some indices to get the proper broadcasting
        Uaug = self.U[None, ...]
        # add a dummy index to treat x as batch of column vectors
        xaug = x[:, None, ..., None]
        yaug = Uaug @ xaug

        # get rid of the dummy index we added
        return yaug[..., 0]

    def from_grid(self, z: torch.Tensor) -> torch.Tensor:
        """Convert batch of grid responses to place basis.

        This multiplies by `V` and sums over the controllers.

        :param z: tensor to transform; shape `(n_batch, n_ctrl, m)`
        :return: transformed tensor; shape `(n_batch, n)`
        """
        # need to add some indices to get the proper broadcasting
        Vaug = self.V
        # add a dummy index to treat z as batch of column vectors
        zaug = z[..., None]
        ypredaug = torch.mean(Vaug @ zaug, dim=1)

        # get rid of the dummy index we added to turn x into a batch of column vectors
        return ypredaug[..., 0]

    def propagate_grid(self, z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Propagate forward in time in grid basis.

        This multiplies by `Lambda ** s`, where `Lambda ** s` is calculated by
        `self.get_lambda_s_matrix`.

        :param z: tensor to propagate forward in time; shape `(n_batch, n_ctrl, m)`
        :param s: amount by which to propagate
        :return: propagated tensor
        """
        grid_prop = self.get_lambda_s_matrix(s)
        z_prop = (grid_prop @ z[..., None])[..., 0]

        return z_prop

    def propagate_place(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Propagate forward in time in place basis.

        :param x: tensor to propagate forward in time; first dimension: batch index
        :param s: amount by which to propagate
        :return: propagated tensor
        """
        z = self.to_grid(x)
        z_tilde = self.propagate_grid(z, s)
        y = self.from_grid(z_tilde)
        return y

    def loss(self, x: torch.Tensor, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Calculate quadratic loss on a batch of input, outputs, and controls.

        First dimension: batch index.

        :param x: initial values in place basis
        :param y: final values in place basis
        :param s: amount of shift (control) for each sample.
        :return: mean losses over the batch
        """
        y_pred = self.propagate_place(x, s)
        return 0.5 * torch.mean((y - y_pred) ** 2)

    def parameters(self) -> list:
        """Get list of parameters that are being optimized."""
        return [self.U, self.V, self.xi, self.theta]

    def zero_grad(self):
        """Set gradients of all parameters to zero."""
        for param in self.parameters():
            if param.grad is not None:
                param.grad.zero_()

    def to(self, device: torch.device) -> "PlaceGridMultiNonBio":
        """Send all parameters to the given device."""
        for param in self.parameters():
            param.to(device)

        return self

    def train(self):
        """Set model to training mode. (Does nothing for now)."""
        pass

    def eval(self):
        """Set model to evaluation (test) mode. (Does nothing for now)."""
        pass

    @staticmethod
    def _get_lambda_from_parts(rho: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        m = rho.shape[1]
        mt = theta.shape[1]

        xs = rho[:, :mt] * torch.cos(theta)
        ys = rho[:, :mt] * torch.sin(theta)

        n_batch = len(rho)
        res = torch.zeros(n_batch, 2 * mt + (m > mt), 2 * mt + (m > mt))

        for j in range(mt):
            idx = 2 * j
            res[:, idx, idx] = xs[:, j]
            res[:, idx + 1, idx + 1] = xs[:, j]
            res[:, idx, idx + 1] = -ys[:, j]
            res[:, idx + 1, idx] = ys[:, j]

        if m > mt:
            res[:, -1, -1] = rho[:, -1]

        return res

    def get_lambda_s_matrix(self, s: torch.Tensor) -> torch.Tensor:
        """Generate the matrix that propagates grid cells forward.

        If `self.m % 2 == 0`, this is a block-diagonal matrix made up of 2x2 blocks
            rho ** s * [[cos(s * theta), -sin(s * theta)],
                        [sin(s * theta),  cos(s * theta)]] .
        `rho` is `sech(self.xi)`.

        If `self.m % 2 == 1`, a single diagonal element is added at the bottom-right of
        the matrix, equal to `sech(self.xi[-1])`.
        """
        if not torch.is_tensor(s):
            s = torch.tensor(s)
        if s.ndim < 2:
            remove_dim = True
            s = s[None, :]
        else:
            remove_dim = False

        grid_prop = torch.empty((len(s), self.n_ctrl, self.m, self.m))
        rho = 1 / torch.cosh(self.xi)

        all_rho = rho[None, :] ** s[..., None]
        all_theta = self.theta[None, :] * s[..., None]
        for k in range(self.n_ctrl):
            grid_prop[:, k, ...] = self._get_lambda_from_parts(
                all_rho[:, k], all_theta[:, k]
            )

        if not remove_dim:
            return grid_prop
        else:
            return grid_prop[0]
