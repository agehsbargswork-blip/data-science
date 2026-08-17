## One example sim and fit


```python
import numpy as np
from scipy.optimize import linear_sum_assignment

from densmix import simulate_multinomial, fit_multinomial


# Make the example reproducible
np.random.seed(123)


# 1. Generate multinomial data

parameters = {
    "n_components": 3,
    "n_buckets": 12,
    "dirichlet": np.ones(12),
    "component_weights": [0.1, 0.2, 0.7],
    "n_actions_per_bucket": {
        "max_actions": 300,
        "prob_action": 0.2,
    },
}

simulation = simulate_multinomial(
    size=1000,
    parameters=parameters,
)

print("Data shape:", simulation.data.shape)
print("Labels shape:", simulation.labels.shape)
print("Profile shape:", simulation.mixture_profiles.shape)
print("Generating weights:", simulation.parameters["component_weights"])## One example

```

    Data shape: (1000, 12)
    Labels shape: (1000,)
    Profile shape: (12, 3)
    Generating weights: [0.1, 0.2, 0.7]
    

## fit


```python
# 2. Fit a multinomial mixture

fit = fit_multinomial(
    simulation.data,
    n_components=3,
)

print("Converged:", fit.converged)
print("Iterations:", fit.n_iter)
print("Fitted weights:", fit.weights)
print("Fitted profile shape:", fit.mixture_profiles.shape)
print("Responsibilities shape:", fit.responsibilities.shape)
```

    Converged: True
    Iterations: 11
    Fitted weights: [0.68544881 0.09873076 0.21582042]
    Fitted profile shape: (12, 3)
    Responsibilities shape: (1000, 3)
    

The fitted component order will not necessarily match the generated order. For example, generated component 0 might be fitted component 2.

Therefore, align components using the distance between their profiles:


```python
# 3. Match fitted components to generated components

true_profiles = simulation.mixture_profiles
fitted_profiles = fit.mixture_profiles

n_components = true_profiles.shape[1]

cost = np.empty((n_components, n_components))

for true_component in range(n_components):
    for fitted_component in range(n_components):
        cost[true_component, fitted_component] = np.linalg.norm(
            true_profiles[:, true_component]
            - fitted_profiles[:, fitted_component]
        )

true_indices, fitted_indices = linear_sum_assignment(cost)
```

Create aligned weights and profiles:


```python
aligned_weights = np.empty_like(fit.weights)
aligned_profiles = np.empty_like(fit.mixture_profiles)

for true_component, fitted_component in zip(
    true_indices,
    fitted_indices,
):
    aligned_weights[true_component] = fit.weights[fitted_component]

    aligned_profiles[:, true_component] = (
        fit.mixture_profiles[:, fitted_component]
    )
```

Compare the weights:


```python
true_weights = np.asarray(
    simulation.parameters["component_weights"]
)

print()
print(
    f"{'Component':<12}"
    f"{'Generated':>12}"
    f"{'Fitted':>12}"
    f"{'Difference':>12}"
)

for component in range(n_components):
    difference = (
        aligned_weights[component]
        - true_weights[component]
    )

    print(
        f"{component:<12}"
        f"{true_weights[component]:>12.4f}"
        f"{aligned_weights[component]:>12.4f}"
        f"{difference:>12.4f}"
    )
```

    
    Component      Generated      Fitted  Difference
    0                 0.1000      0.0987     -0.0013
    1                 0.2000      0.2158      0.0158
    2                 0.7000      0.6854     -0.0146
    

Compare the profiles using RMSE:


```python
predicted_labels = np.argmax(
    fit.responsibilities,
    axis=1,
)

fitted_to_true = {
    fitted_component: true_component
    for true_component, fitted_component in zip(
        true_indices,
        fitted_indices,
    )
}

aligned_predicted_labels = np.array([
    fitted_to_true[label]
    for label in predicted_labels
])

accuracy = np.mean(
    aligned_predicted_labels == simulation.labels
)

print("Aligned label accuracy:", accuracy)
```

    Aligned label accuracy: 0.996
    


```python

```
