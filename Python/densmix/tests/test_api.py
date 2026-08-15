import numpy as np

import densmix.api as api
from densmix import (
    ExpNormalFit,
    ExpNormalSimulation,
    MultinomialFit,
    MultinomialSimulation,
    fit_exp_normal,
    fit_multinomial,
    simulate_exp_normal,
    simulate_multinomial,
)


def test_fit_exp_normal_returns_structured_result(monkeypatch):
    calls = {}
    raw_result = {
        "parameters": {
            "weight": 0.4,
            "lambda": 0.5,
            "mu": 10.0,
            "sigma": 2.0,
        },
        "weights": {"exp": 0.4, "normal": 0.6},
        "converged": True,
        "iterations": 7,
        "bayes_probs": np.array([0.8, 0.2]),
        "loglik": [-12.0, -10.0],
        "deltas": [0.01, 0.02, 0.01, 0.01],
        "total_delta": 0.05,
    }

    class FakeModels:
        def __init__(self, data, control, start):
            calls["init"] = {
                "data": data,
                "control": control,
                "start": start,
            }

        def fit_exp_norm(self):
            return raw_result

    monkeypatch.setattr(api, "Models", FakeModels)

    data = object()
    control = {"max_iter": 50}
    start = {"weight": 0.4}
    result = fit_exp_normal(data, control=control, start=start)

    assert isinstance(result, ExpNormalFit)
    assert result.parameters is raw_result["parameters"]
    assert result.weights is raw_result["weights"]
    assert result.converged is True
    assert result.n_iter == 7
    assert result.responsibilities is raw_result["bayes_probs"]
    assert result.log_likelihood_history is raw_result["loglik"]
    assert result.log_likelihood == -10.0
    assert calls["init"] == {
        "data": data,
        "control": control,
        "start": start,
    }


def test_fit_multinomial_returns_structured_result(monkeypatch):
    calls = {}
    raw_result = {
        "mixture_profiles": np.ones((4, 3)) / 4,
        "weights": np.array([0.2, 0.3, 0.5]),
        "converged": True,
        "bayes_probs": np.ones((2, 3)) / 3,
        "iterations": np.int64(5),
        "profiles_delta": 1e-6,
    }

    class FakeModels:
        def __init__(self, data, control, start):
            calls["init"] = {
                "data": data,
                "control": control,
                "start": start,
            }

        def fit_multinom(self, n_components):
            calls["n_components"] = n_components
            return raw_result

    monkeypatch.setattr(api, "Models", FakeModels)

    data = object()
    result = fit_multinomial(data, n_components=3)

    assert isinstance(result, MultinomialFit)
    assert result.parameters["mixture_profiles"] is raw_result["mixture_profiles"]
    assert result.parameters["weights"] is raw_result["weights"]
    assert result.weights is raw_result["weights"]
    assert result.mixture_profiles is raw_result["mixture_profiles"]
    assert result.converged is True
    assert result.n_iter == 5
    assert result.responsibilities is raw_result["bayes_probs"]
    assert result.log_likelihood_history is None
    assert result.log_likelihood is None
    assert calls["n_components"] == 3


def test_simulate_exp_normal_returns_structured_result(monkeypatch):
    data = np.array([0.5, 9.5])
    used_parameters = {
        "weight": 0.4,
        "lambda": 0.5,
        "mu": 10.0,
        "sigma": 2.0,
    }

    class FakeGenerator:
        def __init__(self, size, parameters):
            self.parameters = used_parameters

        def gen_exp_norm(self):
            return data

    monkeypatch.setattr(api, "Generator", FakeGenerator)

    result = simulate_exp_normal(size=2)

    assert isinstance(result, ExpNormalSimulation)
    assert result.data is data
    assert result.parameters == used_parameters
    assert result.parameters is not used_parameters


def test_simulate_multinomial_returns_structured_result(monkeypatch):
    data = np.array([[2, 1], [0, 3]])
    labels = np.array([0, 1])
    mixture_profiles = np.array([[0.7, 0.2], [0.3, 0.8]])
    used_parameters = {
        "n_components": 2,
        "n_buckets": 2,
    }

    class FakeGenerator:
        def __init__(self, size, parameters):
            self.parameters = used_parameters

        def gen_multinom(self):
            return {
                "data": data,
                "component_labels": labels,
                "mixture_profiles": mixture_profiles,
            }

    monkeypatch.setattr(api, "Generator", FakeGenerator)

    result = simulate_multinomial(size=2)

    assert isinstance(result, MultinomialSimulation)
    assert result.data is data
    assert result.labels is labels
    assert result.mixture_profiles is mixture_profiles
    assert result.parameters == used_parameters
    assert result.parameters is not used_parameters


def test_public_api_end_to_end_returns_structured_results():
    np.random.seed(123)

    exp_simulation = simulate_exp_normal(size=500)
    exp_fit = fit_exp_normal(exp_simulation.data)

    assert isinstance(exp_simulation, ExpNormalSimulation)
    assert isinstance(exp_fit, ExpNormalFit)
    assert exp_simulation.data.shape == (500,)
    assert exp_fit.responsibilities.shape == (500,)
    assert exp_fit.log_likelihood == exp_fit.log_likelihood_history[-1]

    multinomial_simulation = simulate_multinomial(size=500)
    multinomial_fit = fit_multinomial(
        multinomial_simulation.data,
        n_components=3,
    )

    assert isinstance(multinomial_simulation, MultinomialSimulation)
    assert isinstance(multinomial_fit, MultinomialFit)
    assert multinomial_simulation.data.shape == (500, 12)
    assert multinomial_simulation.labels.shape == (500,)
    assert multinomial_fit.responsibilities.shape == (500, 3)
