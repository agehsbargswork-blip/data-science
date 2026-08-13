class Controller:
    def __init__(self,
                 max_iter = 1000,
                 tolerance = 1e-5,
                 min_weight = 1e-5,
                 verbose = False):
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.min_weight = min_weight
        self.verbose = verbose

    def get_em_controls(self):
        controls = {
            "max_iter": self.max_iter,
            "tolerance": self.tolerance,
            "min_weight": self.min_weight,
            "verbose": self.verbose
        }
        return controls