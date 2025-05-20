import numpy as np
from discretizer import Discretizer
import pytest

@pytest.fixture
def simple_discretizer():
    '''
    Makes a discretizer w/ 2 bins over [-1, 1].
    '''
    return Discretizer(bins_per_feature=[2], lower_bounds=[-1.0], upper_bounds=[1.0])

def test_bins_internal_structure(simple_discretizer):
    '''
    There should be exactly 1 internal boundary for 2 bins.
    '''
    bins = simple_discretizer.bins
    assert isinstance(bins, list)
    assert len(bins) == 1
    # Expect bin edge of [0.0]
    assert np.allclose(bins[0], np.array([0.0]))
    
@pytest.mark.parametrize("obs,expected", [
    ([-0.5], (0,)),    # below the boundary -> bin 0
    ([0.5], (1,)),     # above the boundary -> bin 1
    ([0.0], (1,)),     # exactly on the boundary -> bin 1
    ])
def test_discretize_single_value(simple_discretizer, obs, expected):
    res = simple_discretizer.discretize(obs)
    assert res == expected

def test_discretize_out_of_bounds(simple_discretizer):
    '''
    values below lower_bound should map to 0, above upper bound should map to last bin
    '''
    assert simple_discretizer.discretize([-2.0]) == (0,)
    assert simple_discretizer.discretize([ 2.0]) == (1,)

def test_multiple_features():
    '''
    Test a 2-dimensional discretizer
    '''
    lb = [-10.0, 0.0]
    ub = [ 10.0, 5.0]
    # 4 bins for first feature, 2 bins for second
    disc = Discretizer(bins_per_feature=[4, 2], lower_bounds=lb, upper_bounds=ub)

    # feature0 edges: np.linspace(-10,10,5)[1:-1] = [-5, 0, 5]
    # feature1 edges: np.linspace(0,5,3)[1:-1]    = [2.5]
    obs = [ 6.0, 1.0]
    state = disc.discretize(obs)
    # 6.0 falls into bin 3 (edges [-5,0,5] → bins: <-5=0, -5-0=1,0-5=2,>5=3)
    # 1.0 falls into bin 0 (edge [2.5] → <2.5 = bin 0)
    assert state == (3, 0)


def test_invalid_input_length():
    '''
    If observation length != number of features, zip will truncate.
    We choose to assert that this returns a shorter tuple, not crash.
    '''
    disc = Discretizer(bins_per_feature=[2,2,2], lower_bounds=[0,0,0], upper_bounds=[1,1,1])
    # passing only two values should give a 2‐tuple
    state = disc.discretize([0.5, 0.5])
    assert isinstance(state, tuple)
    assert len(state) == 2

def test_edge_case_exact_bounds():
    '''
    Test that values exactly at lower_bounds map to bin 0, and exactly at upper_bounds map to last bin per feature.
    '''
    lb = [0.0, -1.0]
    ub = [1.0,  1.0]
    disc = Discretizer([3, 3], lb, ub)
    # 3 bins → edges np.linspace(0,1,4)[1:-1] = [0.25, 0.5, 0.75]
    assert disc.discretize([0.0, -1.0]) == (0, 0) # exactly lower
    assert disc.discretize([1.0,  1.0]) == (2, 2) # exactly upper