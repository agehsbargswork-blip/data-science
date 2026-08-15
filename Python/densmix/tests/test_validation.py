from copy import deepcopy

import numpy as np
import pytest

from densmix.validation import Checker


VALID_EXP_NORM_PARAMS = {
    "weight": 0.5,
    "lambda": 0.6,
    "mu": 10.0,
    "sigma": 1.0,
}


VALID_MULTINOM_PARAMS_GEN = {
    "n_components": 3,
    "n_buckets": 12,
    "dirichlet": np.repeat(1, 12),
    "component_weights": [0.1, 0.2, 0.7],
    "n_actions_per_bucket": {
        "max_actions": 300,
        "prob_action": 0.2,
    },
}

VALID_MULTINOM_PARAMS_FIT = {
    "mixture_profiles": np.ones(shape=(12,3))/12,
    "weights": np.ones(3)/3
}


def test_valid_exp_norm_parameters():
    result_gen = Checker(
        model_name="exp-norm",
        check_type="gen",
        start=VALID_EXP_NORM_PARAMS,
    ).validate_parameters()

    assert result_gen == VALID_EXP_NORM_PARAMS

    result_fit = Checker(
        model_name="exp-norm",
        check_type="fit",
        start=VALID_EXP_NORM_PARAMS,
    ).validate_parameters()

    assert result_fit == VALID_EXP_NORM_PARAMS



@pytest.mark.parametrize(
    ("parameter", "invalid_value"),
    [
        ("weight", -0.1),
        ("weight", 1.1),
        ("lambda", 0),
        ("sigma", 0),
    ],
)
def test_invalid_exp_norm_parameters(
    parameter,
    invalid_value,
):
    params = VALID_EXP_NORM_PARAMS.copy()
    params[parameter] = invalid_value

    with pytest.raises(ValueError):
        Checker(
            model_name="exp-norm",
            check_type="gen",
            start=params,
        ).validate_parameters()

    with pytest.raises(ValueError):
        Checker(
            model_name="exp-norm",
            check_type="fit",
            start=params,
        ).validate_parameters()


def test_valid_multinom_parameters():
    result_gen = Checker(
        model_name="multinom",
        check_type="gen",
        start=VALID_MULTINOM_PARAMS_GEN,
    ).validate_parameters()

    assert result_gen["n_components"] == 3
    assert result_gen["n_buckets"] == 12

    np.testing.assert_array_equal(
        result_gen["dirichlet"],
        VALID_MULTINOM_PARAMS_GEN["dirichlet"],
    )

    np.testing.assert_array_equal(
        result_gen["component_weights"],
        VALID_MULTINOM_PARAMS_GEN["component_weights"],
    )

    assert (
        result_gen["n_actions_per_bucket"]
        == VALID_MULTINOM_PARAMS_GEN["n_actions_per_bucket"]
    )

    result_fit = Checker(
        model_name="multinom",
        check_type="fit",
        start=VALID_MULTINOM_PARAMS_FIT,
    ).validate_parameters()

    np.testing.assert_array_equal(
        result_fit["mixture_profiles"],
        VALID_MULTINOM_PARAMS_FIT["mixture_profiles"],
    )

    np.testing.assert_array_equal(
        result_fit["weights"],
        VALID_MULTINOM_PARAMS_FIT["weights"],
    )


@pytest.mark.parametrize(
    ("parameter", "invalid_value"),
    [
        ("n_components", 0),
        ("n_buckets", 0),
        ("dirichlet", [1, 1, 0]),
        ("component_weights", [0, 0.3, 0.7]),
        ("component_weights", [0.1, 0.2, 1.0]),
    ],
)
def test_invalid_multinom_parameters_gen(
    parameter,
    invalid_value,
):
    params = deepcopy(VALID_MULTINOM_PARAMS_GEN)
    params[parameter] = invalid_value

    with pytest.raises(ValueError):
        Checker(
            model_name="multinom",
            check_type="gen",
            start=params,
        ).validate_parameters()


@pytest.mark.parametrize(
    ("parameter", "invalid_value"),
    [
        ("max_actions", 1),
        ("prob_action", 0),
    ],
)
def test_invalid_actions_per_bucket_gen(
    parameter,
    invalid_value,
):
    params = deepcopy(VALID_MULTINOM_PARAMS_GEN)
    params["n_actions_per_bucket"][parameter] = invalid_value

    with pytest.raises(ValueError):
        Checker(
            model_name="multinom",
            check_type="gen",
            start=params,
        ).validate_parameters()


@pytest.mark.parametrize(
    ("parameter", "invalid_value"),
    [
        ("mixture_profiles", np.zeros(shape=(12,3))),
        ("mixture_profiles", np.ones(shape=(12,3))),
        ("weights", [0, 0.3, 0.7]),
        ("weights", [0.1, 0.2, 1.0]),
    ],
)
def test_invalid_multinom_parameters_fit(
    parameter,
    invalid_value,
):
    params = deepcopy(VALID_MULTINOM_PARAMS_FIT)
    params[parameter] = invalid_value

    with pytest.raises(ValueError):
        Checker(
            model_name="multinom",
            check_type="fit",
            start=params,
        ).validate_parameters()
