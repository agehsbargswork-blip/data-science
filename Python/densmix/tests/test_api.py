import densmix.api as api
from densmix import (
    fit_exp_normal,
    fit_multinomial,
    simulate_exp_normal,
    simulate_multinomial,
)


def test_fit_exp_normal_delegates_to_models(monkeypatch):
    calls = {}
    expected = object()

    class FakeModels:
        def __init__(self, data, control, start):
            calls["init"] = {
                "data": data,
                "control": control,
                "start": start,
            }

        def fit_exp_norm(self):
            calls["method"] = "fit_exp_norm"
            return expected

    monkeypatch.setattr(api, "Models", FakeModels)

    data = object()
    control = {"max_iter": 50}
    start = {"weight": 0.4}
    result = fit_exp_normal(data, control=control, start=start)

    assert result is expected
    assert calls == {
        "init": {
            "data": data,
            "control": control,
            "start": start,
        },
        "method": "fit_exp_norm",
    }


def test_fit_multinomial_delegates_to_models(monkeypatch):
    calls = {}
    expected = object()

    class FakeModels:
        def __init__(self, data, control, start):
            calls["init"] = {
                "data": data,
                "control": control,
                "start": start,
            }

        def fit_multinom(self, n_components):
            calls["n_components"] = n_components
            return expected

    monkeypatch.setattr(api, "Models", FakeModels)

    data = object()
    control = {"tolerance": 1e-6}
    start = {"weights": [0.2, 0.3, 0.5]}
    result = fit_multinomial(
        data,
        n_components=3,
        control=control,
        start=start,
    )

    assert result is expected
    assert calls == {
        "init": {
            "data": data,
            "control": control,
            "start": start,
        },
        "n_components": 3,
    }


def test_simulate_exp_normal_delegates_to_generator(monkeypatch):
    calls = {}
    expected = object()

    class FakeGenerator:
        def __init__(self, size, parameters):
            calls["init"] = {
                "size": size,
                "parameters": parameters,
            }

        def gen_exp_norm(self):
            calls["method"] = "gen_exp_norm"
            return expected

    monkeypatch.setattr(api, "Generator", FakeGenerator)

    parameters = {"weight": 0.4}
    result = simulate_exp_normal(size=100, parameters=parameters)

    assert result is expected
    assert calls == {
        "init": {
            "size": 100,
            "parameters": parameters,
        },
        "method": "gen_exp_norm",
    }


def test_simulate_multinomial_delegates_to_generator(monkeypatch):
    calls = {}
    expected = object()

    class FakeGenerator:
        def __init__(self, size, parameters):
            calls["init"] = {
                "size": size,
                "parameters": parameters,
            }

        def gen_multinom(self):
            calls["method"] = "gen_multinom"
            return expected

    monkeypatch.setattr(api, "Generator", FakeGenerator)

    parameters = {"n_components": 3}
    result = simulate_multinomial(size=100, parameters=parameters)

    assert result is expected
    assert calls == {
        "init": {
            "size": 100,
            "parameters": parameters,
        },
        "method": "gen_multinom",
    }
