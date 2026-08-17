# densmix

Here we show example of how to generate and fit exp-norm density.

## Generate data

Start by generating data


```python
import numpy as np

from densmix import simulate_exp_normal, fit_exp_normal

# Make the example reproducible
np.random.seed(123)

# 1. Generate observations from an exponential-normal mixture

true_parameters = {
    "weight": 0.5,   # 50% exponential, 50% normal
    "lambda": 0.6,   # exponential rate
    "mu": 10.0,      # normal mean
    "sigma": 1.0,    # normal standard deviation
}

simulation = simulate_exp_normal(
    size=3000,
    parameters=true_parameters,
)

print(type(simulation))
print(simulation.data.shape)
print(simulation.parameters)
```

    <class 'densmix.results.ExpNormalSimulation'>
    (3000,)
    {'weight': 0.5, 'lambda': 0.6, 'mu': 10.0, 'sigma': 1.0}
    

## Fit


```python
# 2. Fit an exponential-normal mixture to the generated data

fit = fit_exp_normal(simulation.data)

print(type(fit))
print(fit.converged)
print(fit.n_iter)
print(fit.parameters)
```

    <class 'densmix.results.ExpNormalFit'>
    True
    13
    {'weight': np.float64(0.512846037980407), 'lambda': np.float64(0.5969316360467579), 'mu': np.float64(10.031474915013208), 'sigma': np.float64(0.9504012356830546)}
    

## Compare


```python
# 3. Compare generating and fitted parameters

print()
print(f"{'Parameter':<12} {'Generated':>12} {'Fitted':>12} {'Difference':>12}")

for parameter, generated_value in simulation.parameters.items():
    fitted_value = fit.parameters[parameter]
    difference = fitted_value - generated_value

    print(
        f"{parameter:<12}"
        f"{generated_value:>12.4f}"
        f"{fitted_value:>12.4f}"
        f"{difference:>12.4f}"
    )


# Additional fit information

print()
print("Component weights:", fit.weights)
print("Final log-likelihood:", fit.log_likelihood)
print("First five responsibilities:", fit.responsibilities[:5])
```

    
    Parameter       Generated       Fitted   Difference
    weight            0.5000      0.5128      0.0128
    lambda            0.6000      0.5969     -0.0031
    mu               10.0000     10.0315      0.0315
    sigma             1.0000      0.9504     -0.0496
    
    Component weights: {'exp': np.float64(0.512846037980407), 'normal': np.float64(0.48715396201959305)}
    Final log-likelihood: -6322.375883233362
    First five responsibilities: [1.         1.         1.         1.         0.95469638]
    


```python

```
