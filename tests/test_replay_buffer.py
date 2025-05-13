import pytest
from replay_buffer import ReplayBuffer

def make_transition(i):
    '''
    Creates a dummy transition based on an index for reproducibility.
    '''
    return ([i, i+0.1, i+0.2, i+0.3], i % 2, i * 0.5, [i+1, i+1.1, i+1.2, i+1.3], i % 3 == 0)

def test_add_and_len():
    buffer = ReplayBuffer(capacity=10)
    for i in range(5):
        buffer.add(*make_transition(i))
    assert len(buffer) == 5

def test_capacity_overflow():
    buffer = ReplayBuffer(capacity=3)
    for i in range(5):
        buffer.add(*make_transition(i))
    assert len(buffer) == 3
    sampled = [s[0][0] for s in zip(*buffer.sample(3))]
    for val in sampled:
        assert val in [2, 3, 4]

def test_sample_shape():
    buffer = ReplayBuffer(capacity=10)
    for i in range(6):
        buffer.add(*make_transition(i))
    batch = buffer.sample(4)
    states, actions, rewards, next_states, dones = batch
    assert len(states) == len(actions) == len(rewards) == len(next_states) == len(dones) == 4

def test_sample_requires_enough_data():
    buffer = ReplayBuffer(capacity = 5)
    for i in range(3):
        buffer.add(*make_transition(i))
    with pytest.raises(ValueError):
        _ = buffer.sample(4) # not enough data, should raise an error (has 3 elements, I'm asking for 4)

def test_buffer_retains_order():
    buffer = ReplayBuffer(capacity = 5)
    for i in range(5):
        buffer.add(*make_transition(i))
    assert list(buffer.buffer)[0][0][0] == 0 # i = 0 -> 0, so we check that the order is retained and the first entry corresponds to i = 0
    buffer.add(*make_transition(5))
    assert list(buffer.buffer)[0][0][0] == 1 # oldest, from i = 0, should now be gone.