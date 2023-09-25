import pytest
import torch

from neurodmd.sm_model import EquivariantSM
from unittest.mock import Mock


@pytest.fixture
def model():
    torch.manual_seed(42)
    n = 9
    m = 7

    return EquivariantSM(n, m)


def test_lambda_s_matrix_is_size_m(model):
    s = torch.randn(1)
    lbd_s = model.get_lambda_s_matrix(s.item())

    assert lbd_s.shape == (model.m, model.m)


def test_lambda_s_matrix_is_block_diagonal(model):
    s = torch.randn(1)
    lbd_s = model.get_lambda_s_matrix(s.item())

    t_zero = torch.FloatTensor([0])
    for i in range(0, 2 * (model.m // 2), 2):
        crt_rows = lbd_s[[i, i + 1], :]
        crt_rows[:, [i, i + 1]] = 0

        assert torch.allclose(crt_rows, t_zero)

        crt_cols = lbd_s[:, [i, i + 1]]
        crt_cols[[i, i + 1], :] = 0
        assert torch.allclose(crt_cols, t_zero)

    assert torch.allclose(lbd_s[-1, :-1], t_zero)
    assert torch.allclose(lbd_s[:-1, -1], t_zero)


def test_lambda_s_2x2_blocks_match_mu_and_theta(model):
    s = torch.randn(1)
    lbd_s = model.get_lambda_s_matrix(s.item())

    rho = torch.exp(model.mu * s.item())
    theta = s.item() * model.theta

    for i in range(0, 2 * (model.m // 2), 2):
        crt_block = lbd_s[[i, i + 1], :][:, [i, i + 1]]
        crt_a = rho[i // 2] * torch.cos(theta[i // 2])
        crt_b = rho[i // 2] * torch.sin(theta[i // 2])

        crt_exp = torch.FloatTensor([[crt_a, -crt_b], [crt_b, crt_a]])
        assert torch.allclose(crt_block, crt_exp), f"block {i} wrong"


def test_lambda_s_diagonal_block_matches_mu(model):
    s = torch.randn(1)
    lbd_s = model.get_lambda_s_matrix(s.item())

    rho = torch.exp(model.mu * s.item())

    assert torch.allclose(lbd_s[-1, -1], rho[-1])


def test_online_loss(model):
    x = torch.randn(model.n)
    z = torch.randn(model.m)
    z_hat = torch.randn(model.m)

    loss = model.online_loss(x, z, z_hat)

    with torch.no_grad():
        expected_ww = torch.trace(model.W.T @ model.W)
        expected_mm = -0.5 * torch.trace(model.M.T @ model.M)
        expected_cross = -2 * torch.dot(z, model.W @ x)
        expected_self = torch.dot(z, model.M @ z)
        expected_change = (model.gamma / 2) * torch.linalg.norm(z - z_hat) ** 2

        expected_loss = (
            expected_ww + expected_mm + expected_cross + expected_self + expected_change
        )

    assert torch.allclose(loss, expected_loss)


def test_set_state(model):
    z = torch.randn(model.m)

    model.set_state(z)

    assert torch.allclose(model.z, z)


def test_feed(model):
    x = torch.randn(model.n)
    s = torch.randn(1)
    z = torch.randn(model.m)

    lbd_s = model.get_lambda_s_matrix(s.item())
    h = lbd_s @ z

    model.set_state(z)
    model.feed(x, s)

    assert torch.allclose(model.x, x)
    assert torch.allclose(model.s, s)
    assert torch.allclose(model.h, h)


def test_z_update_follows_online_loss(model):
    x = torch.randn(model.n)
    s = torch.randn(1)
    z = torch.randn(model.m)
    z.requires_grad = True

    lbd_s = model.get_lambda_s_matrix(s.item())
    z_prev = torch.randn(model.m)
    h = lbd_s @ z_prev

    loss = model.online_loss(x, z, h)
    loss.backward()
    grad = z.grad.detach().clone()

    model.set_state(z_prev)

    model.feed(x, s)

    model.set_state(z)
    z0 = model.z.detach().clone()

    model.fast_step(iterations=1)

    delta = model.z - z0
    assert torch.allclose(delta, -model.fast_lr * grad)


@pytest.mark.parametrize("var,sign", [("W", -1), ("M", 1), ("mu", -1), ("theta", -1)])
def test_slow_params_update_follows_online_loss(model, var, sign):
    x = torch.randn(model.n)
    s = torch.randn(1)
    z = torch.randn(model.m)

    lbd_s = model.get_lambda_s_matrix(s.item())
    z_prev = torch.randn(model.m)
    h = lbd_s @ z_prev

    loss = model.online_loss(x, z, h)
    loss.backward()
    grad = getattr(model, var).grad.detach().clone()

    param0 = getattr(model, var).detach().clone()
    model.set_state(z_prev)
    model.feed(x, s)

    model.set_state(z)
    model.slow_step()

    delta = getattr(model, var) - param0
    assert torch.allclose(delta, sign * model.slow_lr * grad, atol=1e-6)


@pytest.mark.parametrize("var", ["z", "W", "M", "mu", "theta"])
def test_training_step(var):
    seed = 42
    n = 9
    m = 7

    torch.manual_seed(0)
    x = torch.randn(n)
    s = torch.randn(1)

    torch.manual_seed(seed)
    model1 = EquivariantSM(n, m)
    model1.feed(x, s)
    model1.fast_step()
    model1.slow_step()

    torch.manual_seed(seed)
    model2 = EquivariantSM(n, m)
    model2.training_step(x, s)

    assert torch.allclose(getattr(model1, var), getattr(model2, var))


def test_training_step_passes_iterations_to_fast_step():
    seed = 42
    n = 9
    m = 7
    it = 32

    torch.manual_seed(0)
    x = torch.randn(n)
    s = torch.randn(1)

    torch.manual_seed(seed)
    model1 = EquivariantSM(n, m)
    model1.feed(x, s)
    model1.fast_step(iterations=it)
    model1.slow_step()

    torch.manual_seed(seed)
    model2 = EquivariantSM(n, m)
    model2.training_step(x, s, iterations=it)

    assert torch.allclose(model1.W, model2.W)


def test_training_step_returns_loss_before_slow_step():
    seed = 42
    n = 9
    m = 7

    torch.manual_seed(0)
    x = torch.randn(n)
    s = torch.randn(1)

    torch.manual_seed(seed)
    model1 = EquivariantSM(n, m)
    model1.feed(x, s)
    model1.fast_step()

    with torch.no_grad():
        expected = model1.online_loss(x, model1.z, model1.h)

    torch.manual_seed(seed)
    model2 = EquivariantSM(n, m)
    loss = model2.training_step(x, s)

    assert pytest.approx(loss) == expected.item()


def test_fast_callback_called(model):
    x = torch.randn(model.n)
    s = torch.randn(1)

    mock = Mock()
    model.fast_callback.append(mock)

    model.feed(x, s)
    model.fast_step()

    assert mock.called
