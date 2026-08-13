#
#
#
import numpy as np
from .validation import Checker
from .control import Controller
from scipy.stats import expon, norm

class Models:
    def __init__(self, data, control = {}, start = None):
        self.data = data
        self.start = start
        self.control = control

    def fit_exp_norm(self):

        if self.start is None:
            qqs = np.quantile(self.data, [0.25, 0.5, 0.75])
            self.start = {
                "weight": 0.5,
                "lambda": qqs[0],
                "mu": qqs[2],
                "sigma": np.std(self.data[self.data>qqs[1]])
            }

        checker = Checker(model_name="exp-norm", start=self.start)
        parameters = checker.validate_parameters()

        controller = Controller()
        default_control = controller.get_em_controls()
        control = default_control | self.control
        control = control | {
            "min_lambda" : 1e-3,
            "min_mu": 1e-3,
            "min_sigma" : 1e-3
        }

        data_sum = np.sum(self.data)
        n = len(self.data)
        converged = False
        loglik = []
        i = 1

        while i <= control["max_iter"]:

            expd = parameters["weight"] * expon.pdf(self.data,
                                                    scale=1/parameters["lambda"]
                                                    )
            normd = (1-parameters["weight"]) * norm.pdf(self.data,
                                                        loc=parameters["mu"],
                                                        scale=parameters["sigma"]
                                                        )
            loglik.append(sum(np.log(expd+normd)))

            bayes_probs = expd / (expd + normd)
            total_prob = sum(bayes_probs)
            weighted_data = sum(bayes_probs * self.data)

            new_parameters = {
                "weight": total_prob / n,
                "lambda" : total_prob / weighted_data,
                "mu" : (data_sum - weighted_data) / (n - total_prob)
            }

            new_parameters["sigma"] = np.sqrt(
                sum((1 - bayes_probs) * (self.data - new_parameters["mu"]) ** 2) / (n - total_prob)
            )

            if (
                    new_parameters["weight"] < control["min_weight"] or
                    new_parameters["lambda"] < control["min_lambda"] or
                    new_parameters["mu"] < control["min_mu"] or
                    new_parameters["sigma"] < control["min_sigma"]
            ):
                # min_params = ['min_weight', 'min_lambda', 'min_mu', 'min_sigma']
                raise ValueError("Min tolerance for parameters reached")

            deltas = [ abs(parameters[k] - new_parameters[k])/abs(parameters[k]) for k in parameters]
            total_delta = sum(deltas)
            parameters = new_parameters.copy()
            i = i + 1

            if total_delta < control["tolerance"]:
                converged = True
                expd = parameters["weight"] * expon.pdf(self.data,
                                                        scale=1 / parameters["lambda"]
                                                        )
                normd = (1 - parameters["weight"]) * norm.pdf(self.data,
                                                              loc=parameters["mu"],
                                                              scale=parameters["sigma"]
                                                              )
                loglik.append(sum(np.log(expd + normd)))

                break

        res = {
            "converged": converged,
            "parameters": parameters,
            "iterations": i,
            "loglik": loglik,
            "bayes_probs": bayes_probs,
            "deltas": deltas,
            "total_delta": total_delta,
            "weights": {
                "exp": parameters["weight"],
                "normal": 1-parameters["weight"]
            }
        }

        return res

