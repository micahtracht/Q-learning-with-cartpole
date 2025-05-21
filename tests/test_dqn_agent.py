import torch
import pytest
from dqn_agent import DQN


# code reviewer told me I should use variables like this for self documenting code... good idea.
STATE_DIM = 4
ACTION_DIM = 2


def test_forward_batch_shape():
    """
    Given a batch of states, the model should return a tensor of shape [batch_size, action_dim].
    """
    batch_size = 5
    model = DQN(STATE_DIM, ACTION_DIM)
    x = torch.randn(batch_size, STATE_DIM)
    out = model(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (batch_size, ACTION_DIM)


def test_forward_single_sample_outputs_correct_shape():
    """
    Passing a 1D tensor should be treated as a single sample,
    producing a 1D output of length ACTION_DIM.
    """
    model = DQN(STATE_DIM, ACTION_DIM)
    x = torch.randn(STATE_DIM)
    out = model(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (ACTION_DIM,)


def test_gradients_flow():
    """
    After a forward and backward pass, all parameters should have non-zero gradients.
    """
    model = DQN(STATE_DIM, ACTION_DIM)
    x = torch.randn(3, STATE_DIM)
    out = model(x)
    loss = out.mean()
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None
        assert torch.any(p.grad != 0)


def test_device_forward():
    """
    The model should run on CUDA if available (if not available, just skip the test)
    """
    model = DQN(STATE_DIM, ACTION_DIM)
    if torch.cuda.is_available():
        model.to(torch.device('cuda'))
        x = torch.randn(2, STATE_DIM, device='cuda')
        out = model(x)
        assert out.device.type == 'cuda'
    else:
        pytest.skip("CUDA not available")
