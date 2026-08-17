#
#
#
import numpy as np
from .validation import Checker
from .control import Controller
from .start import Starter
from scipy.special import logsumexp
from scipy.stats import expon, norm, multinomial

class Models:

    def _calc_exp_norm_log_densities(self, parameters):
        expd = np.log(parameters["weight"]) + expon.logpdf(
            self.data,
            scale=1 / parameters["lambda"]
        )
        normd = np.log(1 - parameters["weight"]) + norm.logpdf(
            self.data,
            loc=parameters["mu"],
            scale=parameters["sigma"]
        )

        return expd, normd

    def _calc_multinom_log_probabilities(self, parameters, n_components):
        return np.column_stack([
            np.log(parameters["weights"][j])
            + multinomial.logpmf(
                self.data,
                n=np.sum(self.data, axis=1),
                p=parameters["mixture_profiles"][:, j]
            )
            for j in range(n_components)
        ])

    def __init__(self, data, control = None, start = None):
        self.data = data

        if start is None:
            start = {}
        self.start = start

        if control is None:
            control = {}
        self.control = control

    def fit_exp_norm(self):

        starter = Starter(model_name="exp-norm", data = self.data)
        default_start = starter.get_start()
        start = default_start | self.start

        checker = Checker(
            model_name="exp-norm",
            check_type="fit",
            start=start
        )
        parameters = checker.validate_parameters()

        controller = Controller(model_name="exp-norm")
        default_control = controller.get_em_controls()
        control = default_control | self.control

        data_sum = np.sum(self.data)
        n = len(self.data)
        converged = False
        loglik = []
        i = 1

        while i <= control["max_iter"]:

            expd, normd = self._calc_exp_norm_log_densities(parameters)
            total_log_density = logsumexp(
                np.column_stack((expd, normd)),
                axis=1
            )
            loglik.append(np.sum(total_log_density))

            bayes_probs = np.exp(expd - total_log_density)
            total_prob = np.sum(bayes_probs)
            weighted_data = np.sum(bayes_probs * self.data)

            new_parameters = {
                "weight": total_prob / n,
                "lambda" : total_prob / weighted_data,
                "mu" : (data_sum - weighted_data) / (n - total_prob)
            }

            new_parameters["sigma"] = np.sqrt(
                np.sum((1 - bayes_probs) * (self.data - new_parameters["mu"]) ** 2) / (n - total_prob)
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
            total_delta = np.sum(deltas)
            parameters = new_parameters.copy()
            i = i + 1

            if total_delta < control["tolerance"]:
                converged = True
                expd, normd = self._calc_exp_norm_log_densities(parameters)
                total_log_density = logsumexp(
                    np.column_stack((expd, normd)),
                    axis=1
                )
                bayes_probs = np.exp(expd - total_log_density)
                loglik.append(np.sum(total_log_density))

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

    def fit_multinom(self, n_components):

        controller = Controller(model_name="multinom")
        default_control = controller.get_em_controls()
        control = default_control | self.control

        n_buckets = self.data.shape[1]
        n_obs = self.data.shape[0]
        starter = Starter(model_name="multinom",
                          n_components = n_components,
                          n_buckets = n_buckets)
        default_start = starter.get_start()
        start = default_start | self.start

        checker = Checker(
            model_name="multinom",
            check_type="fit",
            start=start
        )
        parameters = checker.validate_parameters()
        converged = False
        loglik = []

        for i in np.arange(control["max_iter"]):
            # E-step
            bayes_probs_tmp = self._calc_multinom_log_probabilities(
                parameters,
                n_components
            )
            total_log_probability = logsumexp(
                bayes_probs_tmp,
                axis=1,
                keepdims=True
            )
            loglik.append(float(np.sum(total_log_probability)))

            bayes_probs = np.exp(
                    bayes_probs_tmp
                    - total_log_probability
            )

            bayes_col_sums = np.sum(bayes_probs, axis=0)

            mixture_profiles_tmp = []

            for j in range(n_components):
                z = np.sum(
                    self.data * bayes_probs[:, j, np.newaxis],
                    axis=0
                )
                mixture_profiles_tmp.append(z / np.sum(z))

            new_parameters = {
                "mixture_profiles": np.column_stack(mixture_profiles_tmp),
                "weights": bayes_col_sums / n_obs
            }

            profiles_delta = np.abs(
                -1
                + np.sqrt(np.sum(parameters["mixture_profiles"] ** 2))
                / np.sqrt(np.sum(new_parameters["mixture_profiles"] ** 2))
            )

            weights_delta = np.mean(
                np.abs(parameters["weights"] - new_parameters["weights"])
                / np.abs(parameters["weights"])
            )

            total_delta = profiles_delta + weights_delta
            parameters = new_parameters

            if total_delta < control['tolerance']:
                converged = True
                break

        final_log_probabilities = self._calc_multinom_log_probabilities(
            parameters,
            n_components
        )
        loglik.append(float(np.sum(logsumexp(
            final_log_probabilities,
            axis=1
        ))))

        res = {
            "mixture_profiles" : parameters["mixture_profiles"],
            "weights": parameters["weights"],
            "converged": converged,
            "bayes_probs": bayes_probs,
            "iterations": i,
            "profiles_delta": profiles_delta,
            "loglik": loglik
        }

        return res


  #
  #
  # datacolsums <- colSums(data)
  # totalobs <- sum(data)
  #
  #
  # parameters <- validate_multinom_parameters(start)
