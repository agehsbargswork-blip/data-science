


params <- list(
  weight=0.5,
  lambda=0.6,
  mu=10,
  sigma=2
)

data <- gen_exp_norm(size=1000,params)
emfit <- fit_exp_normal(data)
plot_exp_norm(emfit)
