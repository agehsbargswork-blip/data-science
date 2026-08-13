import numpy as np
from scipy.stats import expon, norm
from densmix.generator import Generator
from densmix.models import Models

params = {
    "weight": 0.25,
    "lambda": 2.0,
    "mu": 10.0,
    "sigma": 2.0
}


gen = Generator(size=1000, parameters=params)
data = gen.gen_exp_norm()
print(" ------------------- DATA ------------------- ")


model = Models(data)
fit_em = model.fit_exp_norm()

print(fit_em["parameters"])
print(fit_em["iterations"])
print(fit_em["loglik"])