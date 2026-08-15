from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np


def _require_mapping(value, name):
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")


def _require_numeric_scalar(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
    ):
        raise TypeError(f"{name} must be a finite numeric scalar")


def _require_positive_integer(value, name, minimum=1):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or value < minimum
    ):
        raise ValueError(
            f"{name} must be an integer greater than or equal to {minimum}"
        )


def _require_numeric_array(value, name, ndim):
    array = np.asarray(value)

    if (
        array.ndim != ndim
        or not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise ValueError(f"{name} must be a {ndim}-dimensional numeric array")

    if array.size == 0:
        raise ValueError(f"{name} must not be empty")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")

    return array


def _require_probability_vector(value, name, length=None):
    probabilities = _require_numeric_array(value, name, ndim=1)

    if length is not None and probabilities.shape[0] != length:
        raise ValueError(f"{name} must contain exactly {length} values")

    if np.any(probabilities <= 0) or np.any(probabilities > 1):
        raise ValueError(f"{name} values must be greater than 0 and at most 1")

    if not np.isclose(np.sum(probabilities), 1.0):
        raise ValueError(f"{name} must sum to 1")

    return probabilities


def validate_size(size):
    """Validate a requested simulation sample size."""
    _require_positive_integer(size, "size")
    return size


def validate_exp_norm_data(data):
    """Validate one-dimensional data used by the exp-normal model."""
    return _require_numeric_array(data, "data", ndim=1)


def validate_multinom_data(data):
    """Validate a two-dimensional matrix of multinomial counts."""
    counts = _require_numeric_array(data, "data", ndim=2)

    if np.any(counts < 0):
        raise ValueError("multinomial data must contain non-negative counts")

    if not np.all(counts == np.floor(counts)):
        raise ValueError("multinomial data must contain integer counts")

    return counts


def validate_em_controls(control, model_name=None):
    """Validate a full or partial dictionary of user-supplied EM controls."""
    _require_mapping(control, "control")

    common_controls = {"max_iter", "tolerance", "verbose"}
    exp_norm_controls = {
        "min_weight",
        "min_lambda",
        "min_mu",
        "min_sigma",
    }

    if model_name is None:
        allowed_controls = common_controls | exp_norm_controls
    elif model_name == "exp-norm":
        allowed_controls = common_controls | exp_norm_controls
    elif model_name == "multinom":
        allowed_controls = common_controls
    else:
        raise ValueError(f"Unknown model: {model_name}")

    unknown = set(control) - allowed_controls
    if unknown:
        raise ValueError(f"Unknown control parameters: {sorted(unknown)}")

    if "max_iter" in control:
        _require_positive_integer(control["max_iter"], "max_iter")

    if "tolerance" in control:
        _require_numeric_scalar(control["tolerance"], "tolerance")
        if control["tolerance"] <= 0:
            raise ValueError("tolerance must be greater than 0")

    if "verbose" in control and not isinstance(
        control["verbose"], (bool, np.bool_)
    ):
        raise TypeError("verbose must be a boolean")

    if "min_weight" in control:
        _require_numeric_scalar(control["min_weight"], "min_weight")
        if not 0 < control["min_weight"] < 1:
            raise ValueError("min_weight must be between 0 and 1")

    for name in ("min_lambda", "min_mu", "min_sigma"):
        if name in control:
            _require_numeric_scalar(control[name], name)
            if control[name] <= 0:
                raise ValueError(f"{name} must be greater than 0")

    return dict(control)


class Checker:
    def __init__(self, model_name, check_type, start):
        self.model_name = model_name
        self.check_type = check_type
        self.start = start

    def _validate_required_parameters(self, required_parameters):
        _require_mapping(self.start, "parameters")
        missing = required_parameters - self.start.keys()

        if missing:
            raise ValueError(f"Missing parameters: {sorted(missing)}")

    def validate_exp_norm_parameters(self):
        self._validate_required_parameters(
            {"weight", "lambda", "mu", "sigma"}
        )

        for name in ("weight", "lambda", "mu", "sigma"):
            _require_numeric_scalar(self.start[name], name)

        if not 0 < self.start["weight"] < 1:
            raise ValueError("weight must be between 0 and 1")

        if self.start["lambda"] <= 0:
            raise ValueError("lambda must be greater than 0")

        if self.start["sigma"] <= 0:
            raise ValueError("sigma must be greater than 0")

        return self.start.copy()

    def validate_mn_parameters_gen(self):
        self._validate_required_parameters(
            {
                "n_components",
                "n_buckets",
                "dirichlet",
                "component_weights",
                "n_actions_per_bucket",
            }
        )

        n_components = self.start["n_components"]
        n_buckets = self.start["n_buckets"]
        _require_positive_integer(n_components, "n_components")
        _require_positive_integer(n_buckets, "n_buckets")

        dirichlet = _require_numeric_array(
            self.start["dirichlet"], "dirichlet", ndim=1
        )
        if dirichlet.shape[0] != n_buckets:
            raise ValueError("dirichlet length must match n_buckets")
        if np.any(dirichlet <= 0):
            raise ValueError("All dirichlet entries must be above zero")

        _require_probability_vector(
            self.start["component_weights"],
            "component_weights",
            length=n_components,
        )

        actions = self.start["n_actions_per_bucket"]
        _require_mapping(actions, "n_actions_per_bucket")
        required_actions = {"max_actions", "prob_action"}
        missing_actions = required_actions - actions.keys()

        if missing_actions:
            raise ValueError(
                "Missing n_actions_per_bucket parameters: "
                f"{sorted(missing_actions)}"
            )

        _require_positive_integer(actions["max_actions"], "max_actions", minimum=2)
        _require_numeric_scalar(actions["prob_action"], "prob_action")
        if not 0 < actions["prob_action"] <= 1:
            raise ValueError("prob_action must be greater than 0 and at most 1")

        return self.start.copy()

    def validate_mn_parameters_fit(self):
        self._validate_required_parameters({"mixture_profiles", "weights"})

        weights = _require_probability_vector(self.start["weights"], "weights")
        mixture_profiles = _require_numeric_array(
            self.start["mixture_profiles"], "mixture_profiles", ndim=2
        )

        if mixture_profiles.shape[1] != weights.shape[0]:
            raise ValueError(
                "mixture_profiles columns must match the number of weights"
            )

        if np.any(mixture_profiles <= 0) or np.any(mixture_profiles > 1):
            raise ValueError(
                "mixture profile values must be greater than 0 and at most 1"
            )

        if not np.allclose(np.sum(mixture_profiles, axis=0), 1.0):
            raise ValueError("each mixture profile column must sum to 1")

        return self.start.copy()

    def validate_parameters(self):
        if self.model_name == "exp-norm" and self.check_type in {"gen", "fit"}:
            return self.validate_exp_norm_parameters()

        if self.model_name == "multinom" and self.check_type == "gen":
            return self.validate_mn_parameters_gen()

        if self.model_name == "multinom" and self.check_type == "fit":
            return self.validate_mn_parameters_fit()

        raise ValueError(
            f"Unknown model/check type: {self.model_name}/{self.check_type}"
        )

    def validate_data(self, data):
        if self.model_name == "exp-norm":
            return validate_exp_norm_data(data)
        if self.model_name == "multinom":
            return validate_multinom_data(data)
        raise ValueError(f"Unknown model: {self.model_name}")

    def validate_controls(self, control):
        return validate_em_controls(control, model_name=self.model_name)
