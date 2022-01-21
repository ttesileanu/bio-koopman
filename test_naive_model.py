import torch

from naive_model import PlaceGridSystemNonBio


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

    assert torch.allclose(l, sum(all_l) / samples)


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

    assert lbd_s.shape == (m, m)


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

        assert torch.allclose(crt_rows, tzero)

        crt_cols = lbd_s[:, [i, i + 1]]
        crt_cols[[i, i + 1], :] = 0
        assert torch.allclose(crt_cols, tzero)

    assert torch.allclose(lbd_s[-1, :-1], tzero)
    assert torch.allclose(lbd_s[:-1, -1], tzero)


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

    assert torch.allclose(lbd_s[-1, -1], rho[-1])


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
        crt_y_hat_exp = (system.V @ crt_lbd_s @ system.U @ crt_x[..., None])[..., 0]
        y_hat_exp[i] = crt_y_hat_exp

    assert torch.allclose(y_hat, y_hat_exp)


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

    assert torch.allclose(loss, loss_exp)


def test_u_derivative():
    torch.manual_seed(0)

    n = 10
    m = 6
    samples = 1

    x = torch.normal(torch.zeros(samples, n))
    y = x + 0.1 * torch.normal(torch.zeros(samples, n))
    s = torch.normal(torch.zeros(samples))

    system = PlaceGridSystemNonBio(n, m)

    system.zero_grad()
    l = system.loss(x, y, s)

    l.backward()

    lbd_s = system.get_lambda_s_matrix(s.item())

    y_pred = (system.V @ lbd_s @ system.U @ x[..., None])[..., 0]
    eps = y - y_pred

    exp_grad_u = -lbd_s.T @ system.V.T @ eps.T @ x / n

    assert torch.allclose(system.U.grad, exp_grad_u)


def test_v_derivative():
    torch.manual_seed(0)

    n = 10
    m = 6
    samples = 1

    x = torch.normal(torch.zeros(samples, n))
    y = x + 0.1 * torch.normal(torch.zeros(samples, n))
    s = torch.normal(torch.zeros(samples))

    system = PlaceGridSystemNonBio(n, m)

    system.zero_grad()
    l = system.loss(x, y, s)

    l.backward()

    lbd_s = system.get_lambda_s_matrix(s.item())

    y_pred = (system.V @ lbd_s @ system.U @ x[..., None])[..., 0]
    eps = y - y_pred

    exp_grad_v = -eps.T @ x @ system.U.T @ lbd_s.T / n

    assert torch.allclose(system.V.grad, exp_grad_v)


def test_xi_derivative_for_1x1_block():
    torch.manual_seed(0)

    n = 10
    m = 7
    samples = 1

    x = torch.normal(torch.zeros(samples, n))
    y = x + 0.1 * torch.normal(torch.zeros(samples, n))
    s = torch.normal(torch.zeros(samples))

    system = PlaceGridSystemNonBio(n, m)

    system.zero_grad()
    l = system.loss(x, y, s)

    l.backward()

    lbd_s = system.get_lambda_s_matrix(s.item())

    y_pred = (system.V @ lbd_s @ system.U @ x[..., None])[..., 0]
    eps = y - y_pred

    rho = 1 / torch.cosh(system.xi[-1])
    der_rho = s.item() * rho ** (s.item() - 1)

    # exp_grad_rho_1x1 = (
    #     -der_rho
    #     * torch.dot(eps[0], system.V[-1, :] * torch.dot(system.U[:, -1], x[0]))
    #     / n
    # )
    exp_grad_rho_1x1 = (
        -der_rho
        * torch.dot(eps[0], system.V[:, -1] * torch.dot(system.U[-1, :], x[0]))
        / n
    )
    exp_grad_xi_1x1 = -exp_grad_rho_1x1 * rho ** 2 * torch.sinh(system.xi[-1])

    assert torch.allclose(system.xi.grad[-1], exp_grad_xi_1x1)


def test_xi_derivative_for_2x2_block():
    torch.manual_seed(0)

    n = 10
    m = 7
    samples = 1

    x = torch.normal(torch.zeros(samples, n))
    y = x + 0.1 * torch.normal(torch.zeros(samples, n))
    s = torch.normal(torch.zeros(samples))

    system = PlaceGridSystemNonBio(n, m)

    system.zero_grad()
    l = system.loss(x, y, s)

    l.backward()

    lbd_s = system.get_lambda_s_matrix(s.item())

    y_pred = (system.V @ lbd_s @ system.U @ x[..., None])[..., 0]
    eps = y - y_pred

    rho = 1 / torch.cosh(system.xi[0])
    der_lbd_s = s.item() * lbd_s[:2, :2] / rho

    exp_grad_rho_2x2 = (
        -torch.dot(eps[0], system.V[:, :2] @ der_lbd_s @ system.U[:2, :] @ x[0]) / n
    )
    exp_grad_xi_2x2 = -exp_grad_rho_2x2 * rho ** 2 * torch.sinh(system.xi[0])

    assert torch.allclose(system.xi.grad[0], exp_grad_xi_2x2)


def test_theta_derivative_for_2x2_block():
    torch.manual_seed(0)

    n = 10
    m = 7
    samples = 1

    x = torch.normal(torch.zeros(samples, n))
    y = x + 0.1 * torch.normal(torch.zeros(samples, n))
    s = torch.normal(torch.zeros(samples))

    system = PlaceGridSystemNonBio(n, m)

    system.zero_grad()
    l = system.loss(x, y, s)

    l.backward()

    lbd_s = system.get_lambda_s_matrix(s.item())

    y_pred = (system.V @ lbd_s @ system.U @ x[..., None])[..., 0]
    eps = y - y_pred

    rho = 1 / torch.cosh(system.xi[0])
    theta = s * system.theta[0]
    cc = torch.cos(theta)
    ss = torch.sin(theta)
    der_lbd_s = s.item() * rho ** s * torch.FloatTensor([[-ss, -cc], [cc, -ss]])

    # exp_grad_theta_2x2 = (
    #     -torch.dot(eps[0], x[0] @ system.U[:, :2] @ der_lbd_s @ system.V[:2, :]) / n
    # )
    exp_grad_theta_2x2 = (
        -torch.dot(eps[0], system.V[:, :2] @ der_lbd_s @ system.U[:2, :] @ x[0]) / n
    )

    assert torch.allclose(system.theta.grad[0], exp_grad_theta_2x2)
