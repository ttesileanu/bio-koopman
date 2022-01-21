import torch

from naive_model_cplx import PlaceGridSystemNonBioCplx


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

    ell = system.loss(x, y, s)

    assert torch.allclose(ell, sum(all_l) / samples)


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
    ell = system.loss(x, y, s)

    ell.backward()

    lbd_s_mat = torch.diag(system.lbd ** s[0])
    y_pred = system.V @ lbd_s_mat @ system.U @ (x[0] + 0j)
    eps = y[0] - y_pred

    assert torch.allclose(
        torch.imag(eps), torch.FloatTensor([0]), atol=1e-7
    ), "eps complex?"

    exp_grad_u = (
        -(lbd_s_mat.conj() @ system.V.T.conj() @ eps[:, None] @ (x[0] + 0j)[None, :])
        / n
    )

    assert torch.allclose(system.U.grad, exp_grad_u)


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
    ell = system.loss(x, y, s)

    ell.backward()

    lbd_s_mat = torch.diag(system.lbd ** s[0])
    y_pred = system.V @ lbd_s_mat @ system.U @ (x[0] + 0j)
    eps = y[0] - y_pred

    assert torch.allclose(torch.imag(eps), torch.FloatTensor([0])), "eps complex?"

    exp_grad_v = (
        -eps[:, None] @ (x[0] + 0j)[None, :] @ system.U.T.conj() @ lbd_s_mat.conj() / n
    )

    assert torch.allclose(system.V.grad, exp_grad_v)


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
    ell = system.loss(x, y, s)

    ell.backward()

    lbd_s_mat = torch.diag(system.lbd ** s[0])
    y_pred = system.V @ lbd_s_mat @ system.U @ (x[0] + 0j)
    eps = y[0] - y_pred

    assert torch.allclose(torch.imag(eps), torch.FloatTensor([0])), "eps complex?"

    exp_grad_lbd = (
        -s[0]
        * (system.lbd ** (s[0] - 1)).conj()
        * torch.diag(system.U @ (x[0] + 0j)[:, None] @ eps[None, :] @ system.V).conj()
        / n
    )

    assert torch.allclose(system.lbd.grad, exp_grad_lbd)


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
    ell = system.loss(x, y, s)

    ell.backward()

    lbd_s_mat = torch.diag(system.lbd ** s[0])
    y_pred = system.V @ lbd_s_mat @ system.U @ (x[0] + 0j)
    eps = y[0] - y_pred

    assert torch.allclose(torch.imag(eps), torch.FloatTensor([0])), "eps complex?"

    g = (system.U @ (x[0] + 0j)[:, None]).squeeze()
    vt_eps = (eps[None, :] @ system.V).squeeze()
    exp_grad_lbd = -s[0] * (system.lbd ** (s[0] - 1) * g * vt_eps).conj() / n

    assert torch.allclose(system.lbd.grad, exp_grad_lbd)
