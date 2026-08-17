import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from scipy.stats import multinomial

from densmix.models import Models
from densmix.generator import Generator

EXP_NORM_PARAMS = {
    "weight": 0.5,
    "lambda": 0.6,
    "mu": 10.0,
    "sigma": 1.0,
}


def test_exp_norm_fit():
    np.random.seed(123)

    data = Generator(
        size=3000,
        parameters=EXP_NORM_PARAMS,
    ).gen_exp_norm()

    result = Models(data=data).fit_exp_norm()

    assert result["converged"] is True
    assert result["iterations"] > 0

    fitted = result["parameters"]

    assert 0 < fitted["weight"] < 1
    assert fitted["lambda"] > 0
    assert fitted["sigma"] > 0

    assert result["bayes_probs"].shape == (3000,)
    assert np.all(np.isfinite(result["bayes_probs"]))
    assert np.all(result["bayes_probs"] >= 0)
    assert np.all(result["bayes_probs"] <= 1)

    assert np.all(np.isfinite(result["loglik"]))

    # EM log-likelihood should not decrease.
    assert np.all(
        np.diff(result["loglik"]) >= -1e-8
    )

    assert np.isclose(
        result["weights"]["exp"]
        + result["weights"]["normal"],
        1.0,
    )

    # Check recovery of the generating parameters.
    assert np.isclose(
        fitted["weight"],
        EXP_NORM_PARAMS["weight"],
        atol=0.05,
    )

    assert np.isclose(
        fitted["lambda"],
        EXP_NORM_PARAMS["lambda"],
        atol=0.08,
    )

    assert np.isclose(
        fitted["mu"],
        EXP_NORM_PARAMS["mu"],
        atol=0.12,
    )

    assert np.isclose(
        fitted["sigma"],
        EXP_NORM_PARAMS["sigma"],
        atol=0.12,
    )

MULTINOM_PARAMS = {
    "n_components": 3,
    "n_buckets": 12,
    "dirichlet": np.repeat(1, 12),
    "component_weights": [0.1, 0.2, 0.7],
    "n_actions_per_bucket": {
        "max_actions": 300,
        "prob_action": 0.2,
    },
}


def matched_accuracy(true_labels, predicted_labels):
    _, true_indices = np.unique(
        true_labels,
        return_inverse=True,
    )

    _, predicted_indices = np.unique(
        predicted_labels,
        return_inverse=True,
    )

    confusion = np.zeros(
        (
            true_indices.max() + 1,
            predicted_indices.max() + 1,
        ),
        dtype=int,
    )

    np.add.at(
        confusion,
        (true_indices, predicted_indices),
        1,
    )

    rows, columns = linear_sum_assignment(-confusion)

    return (
        confusion[rows, columns].sum()
        / len(true_labels)
    )


def test_multinom_fit():
    np.random.seed(123)

    generated = Generator(
        size=1000,
        parameters=MULTINOM_PARAMS,
    ).gen_multinom()

    result = Models(
        data=generated["data"],
    ).fit_multinom(
        n_components=MULTINOM_PARAMS["n_components"],
    )

    assert result["converged"] is True
    assert result["iterations"] >= 0

    assert result["bayes_probs"].shape == (1000, 3)
    assert result["mixture_profiles"].shape == (12, 3)
    assert result["weights"].shape == (3,)

    np.testing.assert_allclose(
        result["bayes_probs"].sum(axis=1),
        1.0,
        atol=1e-8,
    )

    np.testing.assert_allclose(
        result["mixture_profiles"].sum(axis=0),
        1.0,
        atol=1e-8,
    )

    np.testing.assert_allclose(
        result["weights"].sum(),
        1.0,
        atol=1e-8,
    )

    assert len(result["loglik"]) == result["iterations"] + 2
    assert np.all(np.isfinite(result["loglik"]))
    assert np.all(np.diff(result["loglik"]) >= -1e-8)
    assert "profiles_delta" not in result
    assert np.isclose(
        result["loglik_delta"],
        np.abs(result["loglik"][-1] - result["loglik"][-2])
        / np.abs(result["loglik"][-2]),
    )

    final_component_log_probabilities = np.column_stack([
        np.log(result["weights"][j])
        + multinomial.logpmf(
            generated["data"],
            n=np.sum(generated["data"], axis=1),
            p=result["mixture_profiles"][:, j],
        )
        for j in range(MULTINOM_PARAMS["n_components"])
    ])
    expected_final_log_likelihood = np.sum(logsumexp(
        final_component_log_probabilities,
        axis=1,
    ))

    assert np.isclose(
        result["loglik"][-1],
        expected_final_log_likelihood,
    )

    predicted_labels = np.argmax(
        result["bayes_probs"],
        axis=1,
    )

    accuracy = matched_accuracy(
        generated["component_labels"],
        predicted_labels,
    )

    assert accuracy > 0.9
