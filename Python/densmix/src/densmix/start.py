#
#
#
import numpy as np
from .generator import get_dirichlet

class Starter:

    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

    def get_start(self):
        start = None
        if self.model_name == "exp-norm":
            data = self.kwargs["data"]
            qqs = np.quantile(data, [0.25, 0.5, 0.75])
            start = {
                "weight": 0.5,
                "lambda": qqs[0],
                "mu": qqs[2],
                "sigma": np.std(data[data > qqs[1]])
            }
        if self.model_name == "multinom":
            n_components = self.kwargs["n_components"]
            n_buckets = self.kwargs["n_buckets"]
            dirichlet = np.repeat(1, n_buckets)
            mixture_profiles = get_dirichlet(
                n_components,
                dparam = dirichlet
            )
            start = {
                "mixture_profiles" : mixture_profiles,
                "weights" : np.repeat(1 / n_components, n_components)
            }

        return start