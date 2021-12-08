""" Define a naive version of the place-grid-cell network that requires backprop for
training. """

import torch


class PlaceGridSystemNonBio:
    """ A real-valued, non-biological implementation of our place&grid cell system.

    Attributes
    ==========
    n, m: int
        number of place and grid cells, respectively
    U, V: torch.Tensor
        arrays defining the conversion from place to grid cells and from grid to place
        cells, respectively; shapes (n, m) and (m, n), respectively
    xi, theta: torch.Tensor
        arrays defining the dynamics in grid space; see below. shapes ((m + 1) // 2,)
        and (m // 2,), respectively

    More specifically, a shift of magnitude s is implemented by the transformation
            x @ U @ L ** s @ V ,
    where x is the input sample, and L is a block-diagonal matrix whose entries are set
    by xi and theta.

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

    def to(self, device: torch.device) -> "PlaceGridSystemNonBio":
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

    @staticmethod
    def _get_lambda_from_parts(rho: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        xs = rho[:len(theta)] * torch.cos(theta)
        ys = rho[:len(theta)] * torch.sin(theta)

        blocks = []
        for i in range(len(theta)):
            crt_block = torch.vstack((
                torch.hstack((xs[i], -ys[i])),
                torch.hstack((ys[i], xs[i])),
            ))
            blocks.append(crt_block)

        if len(rho) > len(theta):
            blocks.append(rho[-1])

        return torch.block_diag(*blocks)

    def get_lambda_s_matrix(self, s: torch.Tensor) -> torch.Tensor:
        """ Generate the matrix that propagates grid cells forward.

        If `self.m % 2 == 0`, this is a block-diagonal matrix made up of 2x2 blocks
            rho ** s * [[cos(s * theta), -sin(s * theta)],
                        [sin(s * theta),  cos(s * theta)]] .
        `rho` is `sech(self.xi)`.

        If `self.m % 2 == 1`, a single diagonal element is added at the bottom-right of
        the matrix, equal to `sech(self.xi[-1])`.
        """
        return self._get_lambda_from_parts(1 / torch.cosh(self.xi) ** s, s * self.theta)
