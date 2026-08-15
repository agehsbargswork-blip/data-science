from copy import deepcopy

import numpy as np
import pytest

from densmix.validation import (
    Checker,
    validate_em_controls,
    validate_exp_norm_data,
    validate_multinom_data,
    validate_size,
)


VALID_EXP_NORM_PARAMS = {
    "weight": 0.5,
    "lambda": 0.6,
    "mu": 10.0,
    "sigma": 1.0,
}

VALID_MULTINOM_PARAMS_GEN = {
    "n_components": 3,
    "n_buckets": 12,
    "dirichlet": np.ones(12),
    "component_weights": [0.1, 0.2, 0.7],
    "n_actions_per_bucket": {
        "max_actions": 300,
        "prob_action": 0.2,
    },
}

VALID_MULTINOM_PARAMS_FIT = {
    "mixture_profiles": np.ones(shape=(12, 3)) / 12,
    "weights": np.ones(3) / 3,
}


def check_parameters(model_name, check_type, parameters):
    return Checker(
        model_name=model_name,
        check_type=check_type,
        start=parameters,
    ).validate_parameters()


def test_valid_exp_norm_parameters():
    assert check_parameters(
        "exp-norm", "gen", VALID_EXP_NORM_PARAMS
    ) == VALID_EXP_NORM_PARAMS
    assert check_parameters(
        "exp-norm", "fit", VALID_EXP_NORM_PARAMS
    ) == VALID_EXP_NORM_PARAMS


@pytest.mark.parametrize("missing_parameter", VALID_EXP_NORM_PARAMS)
def test_missing_exp_norm_parameters(missing_parameter):
    parameters = VALID_EXP_NORM_PARAMS.copy()
    parameters.pop(missing_parameter)

    with pytest.raises(ValueError, match="Missing parameters"):
        check_parameters("exp-norm", "fit", parameters)


@pytest.mark.parametrize(
    ("parameter", "invalid_value"),
    [
        ("weight", -0.1),
        ("weight", 1.1),
        ("lambda", 0),
        ("sigma", 0),
        ("mu", np.nan),
        ("lambda", np.inf),
        ("sigma", "one"),
        ("weight", True),
        ("mu", [10.0]),
    ],
)
def test_invalid_exp_norm_parameters(parameter, invalid_value):
    parameters = VALID_EXP_NORM_PARAMS.copy()
    parameters[parameter] = invalid_value

    with pytest.raises((TypeError, ValueError)):
        check_parameters("exp-norm", "gen", parameters)

    with pytest.raises((TypeError, ValueError)):
        check_parameters("exp-norm", "fit", parameters)


@pytest.mark.parametrize("valid_size", [1, 100, np.int64(10)])
def test_valid_size(valid_size):
    assert validate_size(valid_size) == valid_size


@pytest.mark.parametrize(
    "invalid_size", [0, -1, 1.5, True, "10", np.nan, np.inf]
)
def test_invalid_size(invalid_size):
    with pytest.raises(ValueError):
        validate_size(invalid_size)


def test_valid_multinom_generation_parameters():
    result = check_parameters(
        "multinom", "gen", VALID_MULTINOM_PARAMS_GEN
    )

    assert result["n_components"] == 3
    assert result["n_buckets"] == 12
    np.testing.assert_array_equal(
        result["dirichlet"], VALID_MULTINOM_PARAMS_GEN["dirichlet"]
    )
    np.testing.assert_array_equal(
        result["component_weights"],
        VALID_MULTINOM_PARAMS_GEN["component_weights"],
    )


@pytest.mark.parametrize("parameter", VALID_MULTINOM_PARAMS_GEN)
def test_missing_multinom_generation_parameters(parameter):
    parameters = deepcopy(VALID_MULTINOM_PARAMS_GEN)
    parameters.pop(parameter)

    with pytest.raises(ValueError, match="Missing parameters"):
        check_parameters("multinom", "gen", parameters)


@pytest.mark.parametrize(
    ("parameter", "invalid_value"),
    [
        ("n_components", 0),
        ("n_components", 3.0),
        ("n_components", True),
        ("n_buckets", 0),
        ("n_buckets", 12.5),
        ("dirichlet", np.ones((3, 4))),
        ("dirichlet", np.ones(11)),
        ("dirichlet", np.array([1] * 11 + [0])),
        ("dirichlet", np.array([1] * 11 + [np.nan])),
        ("component_weights", [[0.1, 0.2, 0.7]]),
        ("component_weights", [0.1, 0.9]),
        ("component_weights", [0.0, 0.3, 0.7]),
        ("component_weights", [0.1, 0.2, 0.6]),
        ("component_weights", [0.1, 0.2, np.inf]),
    ],
)
def test_invalid_multinom_generation_parameters(parameter, invalid_value):
    parameters = deepcopy(VALID_MULTINOM_PARAMS_GEN)
    parameters[parameter] = invalid_value

    with pytest.raises((TypeError, ValueError)):
        check_parameters("multinom", "gen", parameters)


@pytest.mark.parametrize(
    ("parameter", "invalid_value"),
    [
        ("max_actions", 1),
        ("max_actions", 100.5),
        ("max_actions", True),
        ("prob_action", 0),
        ("prob_action", 1.1),
        ("prob_action", np.nan),
        ("prob_action", "0.5"),
    ],
)
def test_invalid_actions_per_bucket(parameter, invalid_value):
    parameters = deepcopy(VALID_MULTINOM_PARAMS_GEN)
    parameters["n_actions_per_bucket"][parameter] = invalid_value

    with pytest.raises((TypeError, ValueError)):
        check_parameters("multinom", "gen", parameters)


def test_valid_multinom_fit_parameters():
    result = check_parameters(
        "multinom", "fit", VALID_MULTINOM_PARAMS_FIT
    )

    np.testing.assert_array_equal(
        result["mixture_profiles"],
        VALID_MULTINOM_PARAMS_FIT["mixture_profiles"],
    )
    np.testing.assert_array_equal(
        result["weights"], VALID_MULTINOM_PARAMS_FIT["weights"]
    )


@pytest.mark.parametrize(
    ("parameter", "invalid_value"),
    [
        ("mixture_profiles", np.ones(12) / 12),
        ("mixture_profiles", np.ones((12, 2)) / 12),
        ("mixture_profiles", np.ones((12, 3)) / 10),
        ("mixture_profiles", np.zeros((12, 3))),
        ("mixture_profiles", np.full((12, 3), np.nan)),
        ("weights", [[0.1, 0.2, 0.7]]),
        ("weights", [0.1, 0.9]),
        ("weights", [0.0, 0.3, 0.7]),
        ("weights", [0.1, 0.2, 0.6]),
    ],
)
def test_invalid_multinom_fit_parameters(parameter, invalid_value):
    parameters = deepcopy(VALID_MULTINOM_PARAMS_FIT)
    parameters[parameter] = invalid_value

    with pytest.raises((TypeError, ValueError)):
        check_parameters("multinom", "fit", parameters)


def test_valid_data():
    exp_data = validate_exp_norm_data([0.1, 1.2, 3.4])
    multinom_data = validate_multinom_data([[1, 2, 0], [0, 3, 4]])

    assert exp_data.shape == (3,)
    assert multinom_data.shape == (2, 3)


@pytest.mark.parametrize(
    "invalid_data",
    [[], [[1, 2]], [1, np.nan], [1, np.inf], ["one", "two"]],
)
def test_invalid_exp_norm_data(invalid_data):
    with pytest.raises(ValueError):
        validate_exp_norm_data(invalid_data)


@pytest.mark.parametrize(
    "invalid_data",
    [
        [],
        [1, 2, 3],
        [[1, -1], [2, 3]],
        [[1, 0.5], [2, 3]],
        [[1, np.nan], [2, 3]],
        [[1, np.inf], [2, 3]],
        [["one", "two"]],
    ],
)
def test_invalid_multinom_data(invalid_data):
    with pytest.raises(ValueError):
        validate_multinom_data(invalid_data)


def test_valid_controls_can_be_partial():
    controls = {
        "max_iter": 500,
        "tolerance": 1e-6,
        "verbose": True,
        "min_weight": 1e-4,
        "min_lambda": 1e-4,
        "min_mu": 1e-4,
        "min_sigma": 1e-4,
    }

    assert validate_em_controls(controls, "exp-norm") == controls
    assert validate_em_controls({"max_iter": 10}, "multinom") == {
        "max_iter": 10
    }


@pytest.mark.parametrize(
    ("control", "model_name"),
    [
        ({"max_iter": 0}, "multinom"),
        ({"max_iter": 10.5}, "multinom"),
        ({"max_iter": True}, "multinom"),
        ({"tolerance": 0}, "multinom"),
        ({"tolerance": np.nan}, "multinom"),
        ({"verbose": "yes"}, "multinom"),
        ({"min_weight": 1}, "exp-norm"),
        ({"min_lambda": 0}, "exp-norm"),
        ({"min_mu": np.inf}, "exp-norm"),
        ({"min_sigma": "small"}, "exp-norm"),
        ({"min_weight": 1e-4}, "multinom"),
        ({"unknown": 1}, "exp-norm"),
    ],
)
def test_invalid_controls(control, model_name):
    with pytest.raises((TypeError, ValueError)):
        validate_em_controls(control, model_name)


def test_checker_exposes_data_and_control_validation():
    checker = Checker(
        model_name="multinom",
        check_type="fit",
        start=VALID_MULTINOM_PARAMS_FIT,
    )

    counts = checker.validate_data([[1, 2], [3, 4]])
    controls = checker.validate_controls({"max_iter": 20})

    assert counts.shape == (2, 2)
    assert controls == {"max_iter": 20}


def test_unknown_model_or_check_type_is_rejected():
    with pytest.raises(ValueError, match="Unknown model/check type"):
        check_parameters("exp-norm", "unknown", VALID_EXP_NORM_PARAMS)

    with pytest.raises(ValueError, match="Unknown model"):
        validate_em_controls({}, "unknown")
