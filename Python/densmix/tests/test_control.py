from densmix.control import Controller


def test_exp_norm_em_control_values():
    control = Controller(
        model_name="exp-norm"
    ).get_em_controls()

    assert isinstance(control, dict)

    assert control["max_iter"] == 1000
    assert control["tolerance"] == 1e-5
    assert control["verbose"] is False

    assert control["min_weight"] > 0
    assert control["min_lambda"] > 0
    assert control["min_mu"] > 0
    assert control["min_sigma"] > 0


def test_multinom_em_control_values():
    control = Controller(
        model_name="multinom"
    ).get_em_controls()

    assert isinstance(control, dict)

    assert control["max_iter"] == 1000
    assert control["tolerance"] == 1e-5
    assert control["verbose"] is False

    assert control == {
        "max_iter": 1000,
        "tolerance": 1e-5,
        "verbose": False,
    }