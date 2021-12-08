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

    # lbd_s_mat = torch.diag(system.lbd) ** s[:, None, None]
    lbd_s_mat = torch.stack([torch.diag(system.lbd ** _) for _ in s])
    y_pred = (x + 0j) @ system.U @ lbd_s_mat @ system.V
    eps = (y - y_pred)[0]

    assert torch.allclose(
        torch.imag(eps), torch.FloatTensor([0]), atol=1e-7
    ), "eps complex?"

    exp_grad_u = -(x + 0j).T @ eps @ system.V.T.conj() @ lbd_s_mat[0].conj() / n

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

    # lbd_s_mat = torch.diag(system.lbd) ** s[:, None, None]
    lbd_s_mat = torch.stack([torch.diag(system.lbd ** _) for _ in s])
    y_pred = (x + 0j) @ system.U @ lbd_s_mat @ system.V
    eps = (y - y_pred)[0]

    assert torch.allclose(torch.imag(eps), torch.FloatTensor([0])), "eps complex?"

    exp_grad_v = -lbd_s_mat.conj() @ system.U.T.conj() @ (x + 0j).T @ eps / n

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

    # lbd_s_mat = torch.diag(system.lbd) ** s[:, None, None]
    lbd_s_mat = torch.stack([torch.diag(system.lbd ** _) for _ in s])
    y_pred = (x + 0j) @ system.U @ lbd_s_mat @ system.V
    eps = (y - y_pred)[0]

    assert torch.allclose(torch.imag(eps), torch.FloatTensor([0])), "eps complex?"

    exp_grad_lbd = -s[0] * (system.lbd ** (s[0] - 1)).conj() * torch.diag(
        system.V @ eps.T @ (x + 0j) @ system.U).conj() / n

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

    g = (x + 0j) @ system.U
    lbd_s_mat = torch.stack([torch.diag(system.lbd ** _) for _ in s])
    y_pred = g @ lbd_s_mat @ system.V
    eps = (y - y_pred)[0]

    assert torch.allclose(torch.imag(eps), torch.FloatTensor([0])), "eps complex?"

    vt_eps = eps @ system.V.T.conj()

    exp_grad_lbd = -s[0] * (system.lbd ** (s[0] - 1) * g).conj() * vt_eps / n

    assert torch.allclose(system.lbd.grad, exp_grad_lbd)
