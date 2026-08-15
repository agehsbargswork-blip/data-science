import numpy as np

class Checker:
    def __init__(self, model_name, check_type, start):
        self.model_name = model_name
        self.check_type = check_type
        self.start = start

    def validate_exp_norm_parameters(self):
        required_parameters = {
            "weight",
            "lambda",
            "mu",
            "sigma",
        }

        missing = required_parameters - self.start.keys()

        if missing:
            raise ValueError(
                f"Missing parameters: {sorted(missing)}"
            )

        if not 0 < self.start["weight"] < 1:
            raise ValueError("weight must be between 0 and 1")

        if self.start["lambda"] <= 0:
            raise ValueError("lambda must be greater than 0")

        if self.start["sigma"] <= 0:
            raise ValueError("sigma must be greater than 0")

        return self.start.copy()

    def validate_mn_parameters_gen(self):
        required_parameters = {
            "n_components",
            "n_buckets",
            "dirichlet",
            "component_weights",
            "n_actions_per_bucket",
        }

        missing = required_parameters - self.start.keys()

        if missing:
            raise ValueError(
                f"Missing parameters: {sorted(missing)}"
            )

        if self.start["n_components"] <= 0:
            raise ValueError("n_components must be above zero")

        if self.start["n_buckets"] <= 0:
            raise ValueError("n_buckets must be above zero")

        dirichlet = np.asarray(self.start["dirichlet"])

        if not np.all(dirichlet > 0):
            raise ValueError(
                "All dirichlet entries must be above zero"
            )

        component_weights = np.asarray(
            self.start["component_weights"]
        )

        if not np.all(
                (component_weights > 0)
                & (component_weights < 1)
        ):
            raise ValueError(
                "All component weights must be between 0 and 1"
            )

        actions = self.start["n_actions_per_bucket"]

        required_action_parameters = {
            "max_actions",
            "prob_action",
        }

        missing_actions = (
                required_action_parameters - actions.keys()
        )

        if missing_actions:
            raise ValueError(
                "Missing n_actions_per_bucket parameters: "
                f"{sorted(missing_actions)}"
            )

        if actions["max_actions"] <= 1:
            raise ValueError("max_actions must be above 1")

        if actions["prob_action"] <= 0:
            raise ValueError("prob_action must be positive")

        return self.start.copy()


    def validate_mn_parameters_fit(self):
        required_parameters = {
            "mixture_profiles",
            "weights",
        }

        missing = required_parameters - self.start.keys()

        if missing:
            raise ValueError(
                f"Missing parameters: {sorted(missing)}"
            )

        weights = np.asarray(self.start["weights"])
        mixture_profiles = np.asarray(
            self.start["mixture_profiles"]
        )

        if not np.all((weights > 0) & (weights < 1)):
            raise ValueError(
                "All weights must be between 0 and 1"
            )

        if not np.all(mixture_profiles > 0):
            raise ValueError(
                "All mixture profile values must positive"
            )

        if not np.all(mixture_profiles < 1):
            raise ValueError(
                "All mixture profile values must be smaller than 1"
            )

        return self.start.copy()

    def validate_parameters(self):
        if self.model_name == "exp-norm" and self.check_type == "gen":
            return self.validate_exp_norm_parameters()

        if self.model_name == "exp-norm" and self.check_type == "fit":
            return self.validate_exp_norm_parameters()

        if self.model_name == "multinom" and self.check_type == "gen":
            return self.validate_mn_parameters_gen()

        if self.model_name == "multinom" and self.check_type == "fit":
            return self.validate_mn_parameters_fit()

        raise ValueError(f"Unknown model: {self.model_name}")