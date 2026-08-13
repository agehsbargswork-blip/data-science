#
#
#
import numpy as np
from .validation import Checker
class Generator:
    def __init__(self, size, parameters):
        self.size = size
        self.parameters = parameters

    def gen_exp_norm(self):

        checker = Checker(model_name="exp-norm", start=self.parameters)
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
        return np.concatenate((s1,s2),axis=0)

    def gen_multinom(self):
        pass
