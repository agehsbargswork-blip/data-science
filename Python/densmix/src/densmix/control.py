#
#
#

def get_default_controls():
    default_controls = {
        "max_iter": 1000,
        "tolerance": 1e-5,
        "verbose": False
    }
    return default_controls

class Controller:
    def __init__(self, model_name):
        self.model_name = model_name
        self.default_controls = get_default_controls()

    def get_em_controls(self):
        controls = None
        if self.model_name == "exp-norm":
            controls = {
                "min_weight": 1e-3,
                "min_lambda" : 1e-3,
                "min_mu": 1e-3,
                "min_sigma" : 1e-3
            }
        if self.model_name == "multinom":
            controls = {
            }
        controls = self.default_controls | controls

        return controls