# densmix

[![PyPI - Version](https://img.shields.io/pypi/v/densmix.svg)](https://pypi.org/project/densmix)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/densmix.svg)](https://pypi.org/project/densmix)

`densmix` simulates and fits exponential-normal and multinomial mixture models
with expectation-maximization (EM).

## Features

- Simulate shuffled exponential-normal and multinomial datasets.
- Fit both mixture types through a small public function API.
- Inspect fitted parameters, responsibilities, convergence status, and
  log-likelihood history through structured result objects.
- Validate simulation sizes, fitting data, parameters, and EM controls.

## Installation

Install the published package:

```console
pip install densmix
```

For local development, run the following from `Python/densmix`:

```console
python -m pip install -e .
```

The package requires Python 3.9 or newer.

## Exponential-normal example

Generate an exponential-normal dataset, fit it, and compare the generating and
fitted parameters:

```python
import numpy as np

from densmix import fit_exp_normal, simulate_exp_normal

np.random.seed(123)

simulation = simulate_exp_normal(
    size=1_000,
    parameters={
        "weight": 0.4,
        "lambda": 0.5,
        "mu": 10.0,
        "sigma": 2.0,
    },
)

fit = fit_exp_normal(simulation.data)

print("Generating parameters:", simulation.parameters)
print("Fitted parameters:", fit.parameters)
print("Converged:", fit.converged)
print("Iterations:", fit.n_iter)
print("Final log-likelihood:", fit.log_likelihood)
```

The posterior probability that each observation belongs to the exponential
component is available through `fit.responsibilities`.

## Multinomial example

Generate a three-component multinomial dataset and fit the same number of
components:

```python
import numpy as np

from densmix import fit_multinomial, simulate_multinomial

np.random.seed(123)

simulation = simulate_multinomial(
    size=1_000,
    parameters={
        "n_components": 3,
        "n_buckets": 12,
        "component_weights": [0.2, 0.3, 0.5],
    },
)

fit = fit_multinomial(
    simulation.data,
    n_components=3,
)

print("Generating weights:", simulation.parameters["component_weights"])
print("Fitted weights:", fit.weights)
print("Converged:", fit.converged)
print("Iterations:", fit.n_iter)
print("Final log-likelihood:", fit.log_likelihood)
print("Final relative likelihood change:", fit.log_likelihood_delta)
```

`fit.responsibilities` has shape `(n_samples, n_components)`, while
`fit.mixture_profiles` has shape `(n_buckets, n_components)`.

## Public API

| Function | Returns | Purpose |
| --- | --- | --- |
| `simulate_exp_normal()` | `ExpNormalSimulation` | Generate exponential-normal observations. |
| `fit_exp_normal()` | `ExpNormalFit` | Fit an exponential-normal mixture. |
| `simulate_multinomial()` | `MultinomialSimulation` | Generate multinomial count data and component labels. |
| `fit_multinomial()` | `MultinomialFit` | Fit a multinomial mixture. |

Fit results expose the following common attributes:

- `parameters`
- `weights`
- `converged`
- `n_iter`
- `responsibilities`
- `log_likelihood`
- `log_likelihood_history`

Multinomial fits additionally expose `mixture_profiles` and
`log_likelihood_delta`.

## EM controls

Supply a partial control dictionary to either fitting function:

```python
fit = fit_multinomial(
    simulation.data,
    n_components=3,
    control={
        "max_iter": 500,
        "tolerance": 1e-6,
    },
)
```

Supported common controls are `max_iter`, `tolerance`, and `verbose`.
Exponential-normal fitting also supports `min_weight`, `min_lambda`, `min_mu`,
and `min_sigma`.

## Input requirements

- Exponential-normal fitting expects a non-empty one-dimensional array of
  finite numeric values.
- Multinomial fitting expects a non-empty two-dimensional array of finite,
  non-negative integer counts.
- Simulation `size` values and multinomial component counts must be positive
  integers.
- Probability vectors must contain valid probabilities that sum to one.

Invalid inputs raise `TypeError` or `ValueError` before fitting or simulation
starts.

## Examples

- [Exponential-normal simulation and fitting](docs/examples/exp-norm-sim-fit-1.md)
- [Exponential-normal large-scale example](docs/examples/exp-norm-large-scale-1.md)
- [Multinomial simulation and fitting](docs/examples/multinom-sim-fit-1.md)
- [Multinomial large-scale example](docs/examples/multinom-large-scale-1.md)

The corresponding executable Jupyter notebooks are in
[`docs/examples`](docs/examples).

## Running tests

With Hatch installed:

```console
hatch run pytest
```

## License

`densmix` is distributed under the terms of the
[MIT License](https://spdx.org/licenses/MIT.html).
