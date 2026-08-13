from asyncio.windows_events import NULL


class Checker:
    def __init__(self, model_name, start):
        self.model_name = model_name
        self.start = start

    def validate_exp_norm_parameters(self):
        verified_params = self.start
        return verified_params

    def validate_multinom_parameters(self):
        verified_params = self.start
        return verified_params

    def validate_parameters(self):
        verified = NULL
        if self.model_name == "exp-norm":
            verified = self.validate_exp_norm_parameters()
        elif self.model_name == "multinomial":
            verified = self.validate_multinom_parameters()
        return verified
