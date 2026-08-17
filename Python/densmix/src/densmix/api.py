# SPDX-FileCopyrightText: 2026-present Aleksandrs Gehsbargs <agehsbargs.work@gmail.com>
#
# SPDX-License-Identifier: MIT
from .generator import Generator
from .models import Models
from .results import (
    ExpNormalFit,
    ExpNormalSimulation,
    MultinomialFit,
    MultinomialSimulation,
)
from .validation import (
    validate_em_controls,
    validate_exp_norm_data,
    validate_multinom_data,
    validate_size,
)


def _validate_control(control, model_name):
    if control is None:
        return None
    return validate_em_controls(control, model_name=model_name)


def fit_exp_normal(data, control=None, start=None):
    """Fit an exponential-normal mixture model."""
    data = validate_exp_norm_data(data)
    control = _validate_control(control, model_name="exp-norm")

    result = Models(
        data=data,
        control=control,
        start=start,
    ).fit_exp_norm()

    return ExpNormalFit(
        parameters=result["parameters"],
        weights=result["weights"],
        converged=result["converged"],
        n_iter=int(result["iterations"]),
        responsibilities=result["bayes_probs"],
        log_likelihood_history=result["loglik"],
        deltas=result["deltas"],
        total_delta=result["total_delta"],
    )


def fit_multinomial(data, n_components, control=None, start=None):
    """Fit a multinomial mixture model."""
    data = validate_multinom_data(data)
    control = _validate_control(control, model_name="multinom")

    result = Models(
        data=data,
        control=control,
        start=start,
    ).fit_multinom(n_components=n_components)

    parameters = {
        "mixture_profiles": result["mixture_profiles"],
        "weights": result["weights"],
    }

    return MultinomialFit(
        parameters=parameters,
        weights=result["weights"],
        mixture_profiles=result["mixture_profiles"],
        converged=result["converged"],
        n_iter=int(result["iterations"]),
        responsibilities=result["bayes_probs"],
        profiles_delta=result["profiles_delta"],
        log_likelihood_history=result["loglik"],
    )


def simulate_exp_normal(size, parameters=None):
    """Simulate observations from an exponential-normal mixture."""
    size = validate_size(size)

    generator = Generator(
        size=size,
        parameters=parameters,
    )
    data = generator.gen_exp_norm()

    return ExpNormalSimulation(
        data=data,
        parameters=generator.parameters.copy(),
    )


def simulate_multinomial(size, parameters=None):
    """Simulate observations from a multinomial mixture."""
    size = validate_size(size)

    generator = Generator(
        size=size,
        parameters=parameters,
    )
    result = generator.gen_multinom()

    return MultinomialSimulation(
        data=result["data"],
        labels=result["component_labels"],
        mixture_profiles=result["mixture_profiles"],
        parameters=generator.parameters.copy(),
    )
