import numpy as np
# from scipy.stats import expon, norm
# from densmix.generator import Generator
# from densmix.models import Models
#
# params = {
#     "weight": 0.25,
#     "lambda": 2.0,
#     "mu": 10.0,
#     "sigma": 2.0
# }
#
#
# gen = Generator(size=1000, parameters=params)
# data = gen.gen_exp_norm()
# print(" ------------------- DATA ------------------- ")
#
#
# model = Models(data)
# fit_em = model.fit_exp_norm()
#
# print(fit_em["parameters"])
# print(fit_em["iterations"])
# print(fit_em["loglik"])

# from densmix.generator import Generator
#
# g = Generator(size=100)
# res = g.gen_exp_norm()
# print(res)
#
# from densmix.generator import Generator
#
# g = Generator(size=100)
# res = g.gen_multinom()
# print(len(res))
# print(res['data'].shape)

# from densmix.generator import Generator
#
# g = Generator(size=100)
# endata = g.gen_exp_norm()
#
# from densmix.models import Models
#
# m = Models(data=endata)
# res = m.fit_exp_norm()
# print(res['parameters'])
# print(res['weights'])

# from densmix.generator import get_dirichlet

# from densmix.start import Starter
# n_buckets = 10
# n_components = 3
# starter = Starter(model_name="multinom",
#                   n_components=n_components,
#                   n_buckets=n_buckets)
# default_start = starter.get_start()
# print(default_start)

# from densmix.generator import Generator
# from scipy.stats import multinomial
#
# dp = {
#     "n_components": 4,
#     "n_buckets": 8,
#     "dirichlet": np.repeat(1, 8),
#     "component_weights": [0.1, 0.2, 0.3, 0.4],
#     "n_actions_per_bucket": {
#         "max_actions": 300,
#         "prob_action": 0.2
#     }
# }
# g = Generator(size=2000, parameters=dp)
# mndata = g.gen_multinom()
# print(mndata.keys())
#
# data=mndata["data"]
#
# from densmix.models import Models
#
# m = Models(data=mndata["data"])
# res = m.fit_multinom(n_components=4)
# print(res["iterations"])
# print(res["weights"])

import numpy as np
#
# from densmix.generator import Generator
#
# np.random.seed(123)
#
# MULTINOM_PARAMS = {
#     "n_components": 3,
#     "n_buckets": 12,
#     "dirichlet": np.ones(12),
#     "component_weights": np.array([0.1, 0.2, 0.7]),
#     "n_actions_per_bucket": {
#         "max_actions": 100,
#         "prob_action": 0.5,
#     },
# }
#
# result = Generator(
#     size=500,
#     parameters=MULTINOM_PARAMS
# ).gen_multinom()
#
# data = result["data"]
# labels = result["component_labels"]
# mixture_profiles = result["mixture_profiles"]
# print(mixture_profiles.shape)

import numpy as np
from scipy.optimize import linear_sum_assignment

from densmix.models import Models
from densmix.generator import Generator


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

generated = Generator(
        size=1000,
        parameters=MULTINOM_PARAMS,
).gen_multinom()

result = Models(
    data=generated["data"],
).fit_multinom(
    n_components=MULTINOM_PARAMS["n_components"],
)

print(result["converged"])