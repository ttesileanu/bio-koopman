import pytest

import torch

from types import SimpleNamespace
from multi_model import PlaceGridMultiNonBio


@pytest.fixture
def default_sys() -> PlaceGridMultiNonBio:
    torch.manual_seed(0)
    return PlaceGridMultiNonBio(n_ctrl=3, n=10, m=6)


@pytest.fixture
def default_sys_odd() -> PlaceGridMultiNonBio:
    torch.manual_seed(0)
    return PlaceGridMultiNonBio(n_ctrl=4, n=9, m=7)


def much_data(system: PlaceGridMultiNonBio) -> SimpleNamespace:
    torch.manual_seed(0)

    data = SimpleNamespace(samples=5)
    data.x = torch.normal(torch.zeros(data.samples, system.n))
    data.y = data.x + 0.1 * torch.normal(torch.zeros(data.samples, system.n))
    data.s = torch.rand(data.samples, system.n_ctrl)

    return data


def single_data(system: PlaceGridMultiNonBio) -> SimpleNamespace:
    torch.manual_seed(1)

    data = SimpleNamespace(samples=1)
    data.x = torch.normal(torch.zeros(data.samples, system.n))
    data.y = data.x + 0.1 * torch.normal(torch.zeros(data.samples, system.n))
    data.s = torch.rand(data.samples, system.n_ctrl)

    return data


def test_loss_averages_over_samples(default_sys):
    data = much_data(default_sys)

    all_ell = [
        default_sys.loss(data.x[[i]], data.y[[i]], data.s[[i]])
        for i in range(data.samples)
    ]
    ell = default_sys.loss(data.x, data.y, data.s)

    assert torch.allclose(ell, sum(all_ell) / data.samples)


def test_len_u_equals_n_ctrl(default_sys):
    assert len(default_sys.U) == default_sys.n_ctrl


def test_len_v_equals_n_ctrl(default_sys):
    assert len(default_sys.V) == default_sys.n_ctrl


def test_len_lambda_s_equals_n_ctrl(default_sys):
    lbd_s = default_sys.get_lambda_s_matrix(torch.FloatTensor([0.5, 0.2, 0.1]))
    assert len(lbd_s) == default_sys.n_ctrl


def test_each_lambda_s_matrix_is_size_m_by_m(default_sys_odd):
    lbd_s = default_sys_odd.get_lambda_s_matrix(torch.FloatTensor([0.5, 0.2, 0.1, 0.0]))

    for crt_lbd_s in lbd_s:
        assert crt_lbd_s.shape == (default_sys_odd.m, default_sys_odd.m)


def test_lambda_s_matrix_is_block_diagonal(default_sys_odd):
    lbd_s = default_sys_odd.get_lambda_s_matrix(torch.FloatTensor([0.5, 0.2, 0.1, 0.0]))

    t_zero = torch.FloatTensor([0])
    for crt_lbd_s in lbd_s:
        for i in range(0, 2 * (default_sys_odd.m // 2), 2):
            crt_rows = torch.clone(crt_lbd_s[[i, i + 1], :])
            crt_rows[:, [i, i + 1]] = 0

            assert torch.allclose(crt_rows, t_zero)

            crt_cols = torch.clone(crt_lbd_s[:, [i, i + 1]])
            crt_cols[[i, i + 1], :] = 0
            assert torch.allclose(crt_cols, t_zero)

        assert torch.allclose(crt_lbd_s[-1, :-1], t_zero)
        assert torch.allclose(crt_lbd_s[:-1, -1], t_zero)


def test_lambda_s_2x2_blocks_match_xi_and_theta(default_sys_odd):
    s = [0.5, 0.2, 0.1, 0.0]
    lbd_s = default_sys_odd.get_lambda_s_matrix(torch.FloatTensor(s))

    for k, crt_lbd_s in enumerate(lbd_s):
        xi = default_sys_odd.xi[k]
        theta = default_sys_odd.theta[k]

        rho = 1 / torch.cosh(xi) ** s[k]
        theta = s[k] * theta

        for i in range(0, 2 * (default_sys_odd.m // 2), 2):
            crt_block = crt_lbd_s[[i, i + 1], :][:, [i, i + 1]]
            crt_a = rho[i // 2] * torch.cos(theta[i // 2])
            crt_b = rho[i // 2] * torch.sin(theta[i // 2])

            crt_exp = torch.FloatTensor([[crt_a, -crt_b], [crt_b, crt_a]])
            assert torch.allclose(crt_block, crt_exp), f"block {i} wrong"


def test_lambda_s_diagonal_block_matches_xi(default_sys_odd):
    s = [0.5, 0.2, 0.1, 0.0]
    lbd_s = default_sys_odd.get_lambda_s_matrix(torch.FloatTensor(s))

    for k, crt_lbd_s in enumerate(lbd_s):
        rho = 1 / torch.cosh(default_sys_odd.xi[k]) ** s[k]

        assert torch.allclose(crt_lbd_s[-1, -1], rho[-1])


def test_propagate_place_matches_expectation_multi_sample(default_sys):
    data = much_data(default_sys)
    y_hat = default_sys.propagate_place(data.x, data.s)

    y_hat_exp = torch.zeros_like(y_hat)
    for i in range(data.samples):
        lbd_s = default_sys.get_lambda_s_matrix(data.s[i])
        crt_x = data.x[i]

        U = default_sys.U
        V = default_sys.V

        for k, crt_lbd_s in enumerate(lbd_s):
            crt_y_hat_exp = V[k] @ crt_lbd_s @ U[k] @ crt_x
            y_hat_exp[i] += crt_y_hat_exp

    assert torch.allclose(y_hat, y_hat_exp / default_sys.n_ctrl)


def test_loss_is_correct(default_sys):
    data = much_data(default_sys)
    y_hat = default_sys.propagate_place(data.x, data.s)

    loss = default_sys.loss(data.x, data.y, data.s)
    loss_exp = 0.5 * torch.mean((y_hat - data.y) ** 2)

    assert torch.allclose(loss, loss_exp)


def test_u_derivative(default_sys):
    data = single_data(default_sys)

    default_sys.zero_grad()
    loss = default_sys.loss(data.x, data.y, data.s)
    loss.backward()

    lbd_s = default_sys.get_lambda_s_matrix(data.s[0])

    y_pred = default_sys.propagate_place(data.x, data.s)
    eps = data.y - y_pred

    for k, crt_lbd_s in enumerate(lbd_s):
        exp_grad_u = -crt_lbd_s.T @ default_sys.V[k].T @ eps.T @ data.x / default_sys.n

        assert torch.allclose(default_sys.U.grad[k], exp_grad_u / default_sys.n_ctrl)


def test_v_derivative(default_sys):
    data = single_data(default_sys)

    default_sys.zero_grad()
    loss = default_sys.loss(data.x, data.y, data.s)
    loss.backward()

    lbd_s = default_sys.get_lambda_s_matrix(data.s[0])

    y_pred = default_sys.propagate_place(data.x, data.s)
    eps = data.y - y_pred

    for k, crt_lbd_s in enumerate(lbd_s):
        exp_grad_v = -eps.T @ data.x @ default_sys.U[k].T @ crt_lbd_s.T / default_sys.n

        assert torch.allclose(default_sys.V.grad[k], exp_grad_v / default_sys.n_ctrl)


def test_xi_derivative_for_1x1_block(default_sys_odd):
    data = single_data(default_sys_odd)

    default_sys_odd.zero_grad()
    loss = default_sys_odd.loss(data.x, data.y, data.s)
    loss.backward()

    lbd_s = default_sys_odd.get_lambda_s_matrix(data.s[0])

    y_pred = default_sys_odd.propagate_place(data.x, data.s)
    eps = data.y - y_pred

    rho = 1 / torch.cosh(default_sys_odd.xi[:, -1])
    der_rho = data.s[0] * rho ** (data.s[0] - 1)

    for k, crt_lbd_s in enumerate(lbd_s):
        exp_grad_rho_1x1 = (
            -der_rho[k]
            * torch.dot(
                eps[0],
                default_sys_odd.V[k][:, -1]
                * torch.dot(default_sys_odd.U[k][-1, :], data.x[0]),
            )
            / default_sys_odd.n
        )
        exp_grad_xi_1x1 = (
            -exp_grad_rho_1x1 * rho[k] ** 2 * torch.sinh(default_sys_odd.xi[k, -1])
        )

        assert torch.allclose(
            default_sys_odd.xi.grad[k, -1], exp_grad_xi_1x1 / default_sys_odd.n_ctrl
        )


def test_xi_derivative_for_2x2_block(default_sys_odd):
    data = single_data(default_sys_odd)

    default_sys_odd.zero_grad()
    loss = default_sys_odd.loss(data.x, data.y, data.s)
    loss.backward()

    lbd_s = default_sys_odd.get_lambda_s_matrix(data.s[0])

    y_pred = default_sys_odd.propagate_place(data.x, data.s)
    eps = data.y - y_pred

    rho = 1 / torch.cosh(default_sys_odd.xi[0])

    for k, crt_lbd_s in enumerate(lbd_s):
        der_lbd_s = data.s[0, k] * lbd_s[k, :2, :2] / rho[k]

        exp_grad_rho_2x2 = (
            -torch.dot(
                eps[0],
                default_sys_odd.V[k, :, :2]
                @ der_lbd_s
                @ default_sys_odd.U[k, :2, :]
                @ data.x[0],
            )
            / default_sys_odd.n
        )
        exp_grad_xi_2x2 = (
            -exp_grad_rho_2x2 * rho[k] ** 2 * torch.sinh(default_sys_odd.xi[k, 0])
        )

        assert torch.allclose(
            default_sys_odd.xi.grad[k, 0],
            exp_grad_xi_2x2 / default_sys_odd.n_ctrl,
            atol=1e-6,
        ), f"problem at index {k}"


def test_theta_derivative_for_2x2_block(default_sys_odd):
    data = single_data(default_sys_odd)

    default_sys_odd.zero_grad()
    loss = default_sys_odd.loss(data.x, data.y, data.s)
    loss.backward()

    lbd_s = default_sys_odd.get_lambda_s_matrix(data.s[0])

    y_pred = default_sys_odd.propagate_place(data.x, data.s)
    eps = data.y - y_pred

    rho = 1 / torch.cosh(default_sys_odd.xi[0])
    theta = data.s[0] * default_sys_odd.theta[:, 0]
    cc = torch.cos(theta)
    ss = torch.sin(theta)

    for k, crt_lbd_s in enumerate(lbd_s):
        der_lbd_s = (
            data.s[0, k]
            * rho[k] ** data.s[0, k]
            * torch.FloatTensor([[-ss[k], -cc[k]], [cc[k], -ss[k]]])
        )

        exp_grad_theta_2x2 = (
            -torch.dot(
                eps[0],
                default_sys_odd.V[k, :, :2]
                @ der_lbd_s
                @ default_sys_odd.U[k, :2, :]
                @ data.x[0],
            )
            / default_sys_odd.n
        )

        assert torch.allclose(
            default_sys_odd.theta.grad[k, 0],
            exp_grad_theta_2x2 / default_sys_odd.n_ctrl,
            atol=1e-5,
        ), f"problem at index {k}"
