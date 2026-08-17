#
#
#
import numpy as np
from .validation import Checker

def get_dirichlet(n, dparam):
    gamma_samples = np.asarray(
        [np.random.gamma(shape=x, scale=1, size=n) for x in dparam]
    )
    gamma_samples = gamma_samples / np.sum(gamma_samples, axis=0)
    return gamma_samples

def get_default_mn_params():
    dp = {
        "n_components": 3,
        "n_buckets": 12,
        "dirichlet": np.repeat(1, 12),
        "component_weights": [0.1, 0.2, 0.7],
        "n_actions_per_bucket": {
            "max_actions": 300,
            "prob_action": 0.2
        }
    }
    return dp

def get_default_exp_norm_params():
    dp = {
        "weight": 0.5,
        "lambda": 1.0,
        "mu": 10,
        "sigma": 1.0
    }
    return dp


class Generator:
    def __init__(self, size, parameters=None):
        self.size = size
        if parameters is None:
            parameters = {}
        self.parameters = parameters

    def gen_exp_norm(self):

        default_params = get_default_exp_norm_params()
        self.parameters = default_params | self.parameters

        checker = Checker(
            model_name="exp-norm",
            check_type="gen",
            start=self.parameters
        )
        parameters = checker.validate_parameters()

        # in numpy binomial "size" is output size which we need as 1
        n1 = np.random.binomial(n=self.size,
                                p=parameters["weight"],
                                size=1
                                )
        n2 = self.size - n1
        s1 = np.random.exponential(size=n1,
                                   scale=1/parameters["lambda"]
                                   )
        s2 = np.random.normal(size=n2,
                              loc=parameters["mu"],
                              scale=parameters["sigma"]
                              )
        data = np.concatenate((s1,s2),axis=0)
        return data[np.random.permutation(data.shape[0])]

    def gen_multinom(self):

        default_params = get_default_mn_params()
        self.parameters = default_params | self.parameters

        checker = Checker(
            model_name="multinom",
            check_type="gen",
            start=self.parameters
        )
        parameters = checker.validate_parameters()

        mixture_profiles = get_dirichlet(
            n=parameters["n_components"],
            dparam=parameters["dirichlet"]
        )

        # numpy random multinomial "size" is numnber of trials
        # we could run it without size=1, then flatten can be dropped
        mixture_n_obs = np.random.multinomial(n=self.size,
                                              pvals=parameters["component_weights"],
                                              size=1)
        mixture_n_obs = mixture_n_obs.flatten()

        actions_per_component = np.random.binomial(
            n=parameters["n_actions_per_bucket"]["max_actions"],
            p=parameters["n_actions_per_bucket"]["prob_action"],
            size=parameters["n_components"]
        )

        component_labels = np.repeat(np.arange(0, parameters["n_components"])
                                     , mixture_n_obs)

        datasets = [
            np.random.multinomial(n=actions_per_component[j],
                                  pvals=mixture_profiles[:,j],
                                  size=mixture_n_obs[j])
            for j in np.arange(0, parameters["n_components"])
        ]
        datasets = np.concatenate(datasets, axis=0)

        permutation = np.random.permutation(datasets.shape[0])
        datasets = datasets[permutation]
        component_labels = component_labels[permutation]

        res = {
            "data": datasets,
            "component_labels": component_labels,
            "mixture_profiles": mixture_profiles
        }

        return res
