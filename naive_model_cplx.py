""" Define a naive version of the place-grid-cell network that both requires backprop
for training and uses complex-valued matrices. """

import torch


class PlaceGridSystemNonBioCplx:
    """A complex-number based, non-biological implementation of our place&grid cell
    system.
    """

    def __init__(self, n: int, m: int):
        """Initialize the system.

        :param n: number of place cells
        :param m: number of grid cells; if it is even, all eigenvalues and eigenvectors
                  come in complex-conjugate pairs; otherwise, the first eigenvalue and
                  eigenvector are real and non-negative, and real, respectively.
        """
        self.n = n
        self.m = m

        rnd_scale = 0.1

        m_half = m // 2
        U_half = torch.eye(n, m_half) + rnd_scale * torch.view_as_complex(
            torch.normal(torch.zeros(n, m_half, 2))
        )
        V_half = torch.eye(m_half, n) + rnd_scale * torch.view_as_complex(
            torch.normal(torch.zeros(m_half, n, 2))
        )
        lbd_half = 1 + rnd_scale * torch.view_as_complex(
            torch.normal(torch.zeros(m_half, 2))
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
        """Convert batch of inputs to grid basis.

        This multiplies by `U`.

        :param x: tensor to transform; first dimension: batch index
        :return: transformed tensor
        """
        return (x + 0j) @ self.U

    def from_grid(
        self, z: torch.Tensor, atol=1e-4, rtol=1e-2, return_real: bool = True
    ) -> torch.Tensor:
        """Convert batch of grid responses to place basis.

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

    def propagate_grid(self, z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Propagate forward in time in grid basis.

        This multiplies by `diag(lbd ** s)`.

        :param z: tensor to propagate forward in time; first dimension: batch index
        :param s: amount by which to propagate
        :return: propagated tensor
        """
        return z * torch.tile(self.lbd, (len(s), 1)) ** s[:, None]

    def propagate_place(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Propagate forward in time in place basis.

        :param x: tensor to propagate forward in time; first dimension: batch index
        :param s: amount by which to propagate
        :return: propagated tensor
        """
        return self.from_grid(self.propagate_grid(self.to_grid(x), s))

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
        return [self.U, self.V, self.lbd]

    def zero_grad(self):
        """Set gradients of all parameters to zero."""
        for param in self.parameters():
            if param.grad is not None:
                param.grad.zero_()

    def to(self, device: torch.device) -> "PlaceGridSystemNonBioCplx":
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

    def project_to_real(self, atol=1.0):
        """Ensure predictions stay real by enforcing conjugation symmetry among pairs
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
            self.U[:, i0::2], self.U[:, i0 + 1 :: 2].conj(), atol=atol
        )
        assert torch.allclose(
            self.V[i0::2, :], self.V[i0 + 1 :: 2, :].conj(), atol=atol
        )
        assert torch.allclose(self.lbd[i0::2], self.lbd[i0 + 1 :: 2].conj(), atol=atol)

        with torch.no_grad():
            U_half = 0.5 * (self.U[:, i0::2] + self.U[:, i0 + 1 :: 2].conj())
            V_half = 0.5 * (self.V[i0::2, :] + self.V[i0 + 1 :: 2, :].conj())
            lbd_half = 0.5 * (self.lbd[i0::2] + self.lbd[i0 + 1 :: 2].conj())

            me = 2 * (self.m // 2)

            n = self.n
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
