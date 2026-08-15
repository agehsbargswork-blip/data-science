import numpy as np

from densmix.generator import Generator

# Test for exp-norm
EXP_NORM_PARAMS = {
    "weight": 0.5,
    "lambda": 0.6,
    "mu": 10.0,
    "sigma": 1.0,
}

def test_gen_exp_norm_returns_correct_size():
    np.random.seed(123)

    result = Generator(
        size=1000,
        parameters=EXP_NORM_PARAMS
    ).gen_exp_norm()

    assert isinstance(result, np.ndarray)
    assert result.shape == (1000,)
    assert np.all(np.isfinite(result))


def test_gen_exp_norm_is_reproducible():
    np.random.seed(123)
    first = Generator(100, EXP_NORM_PARAMS).gen_exp_norm()

    np.random.seed(123)
    second = Generator(100, EXP_NORM_PARAMS).gen_exp_norm()

    np.testing.assert_array_equal(first, second)

# Test for multinom
MULTINOM_PARAMS = {
    "n_components": 3,
    "n_buckets": 12,
    "dirichlet": np.ones(12),
    "component_weights": np.array([0.1, 0.2, 0.7]),
    "n_actions_per_bucket": {
        "max_actions": 100,
        "prob_action": 0.5,
    },
}


def test_gen_multinom_output_structure():
    np.random.seed(123)

    result = Generator(
        size=500,
        parameters=MULTINOM_PARAMS
    ).gen_multinom()

    data = result["data"]
    labels = result["component_labels"]
    mixture_profiles = result["mixture_profiles"]

    assert data.shape == (500, 12)
    assert mixture_profiles.shape == (12, 3)
    assert labels.shape == (500,)
    assert np.all(data >= 0)
    assert set(np.unique(labels)) <= {0, 1, 2}