


params <- list(
  weight=0.5,
  lambda=0.6,
  mu=10,
  sigma=2
)

data <- gen_exp_normal(size=1000,params)
emfit <- fit_exp_normal(data)
plot(emfit)


get_dirichlet <- function(n, dparam){
  gamma_samples <-
    sapply(dparam
           , function(y){
             rgamma(n,shape = y, scale = 1)
             }
           )

  gamma_samples / rowSums(gamma_samples)

}

gen_em_mn <- function(size = 1000,
                      params = list("n_components" = 3,
                                    "dirichlet" = rep(1,12),
                                    "component_weights" = c(0.1,0.2,0.7),
                                    "n_actions" = list(
                                      "max_actions" = 300,
                                      "prob_action" = 0.2
                                      )
                                    )
                      )
{

  mixture_profiles <- get_dirichlet(
    params[['n_components']],
    dparam = params[['dirichlet']]
  )

  mixture_n_obs <- rmultinom(1,
                             size,
                             prob=params[["component_weights"]]
                             )

  actions_per_component <- rbinom(params[["n_components"]]
                                  , params[["n_actions"]][["max_actions"]]
                                  , params[["n_actions"]][["prob_action"]]
                                  )

  datasets <- lapply(
    seq(1:params[['n_components']]),
    function(j) {
      t(rmultinom(mixture_n_obs[j]
                  ,actions_per_component[j]
                  ,mixture_profiles[j,]
      )
      )
    }
  )


  data <- do.call(rbind, datasets)

  list(
    data = data,
    mixture_profiles = mixture_profiles,
    mixture_n_obs = mixture_n_obs
  )

}


gendata <- gen_em_mn(size = 1000,
                     params = list("n_components" = 3,
                                   "n_buckets" = 12,
                                   "dirichlet" = rep(1,12),
                                   "component_weights" = c(0.1,0.2,0.7),
                                   "n_actions_per_bucket" = list(
                                     "max_actions" = 30,
                                     "prob_action" = 0.5
                                   )
                     )
      )


fitdata <- fit_multinomial(gendata$data)
par(mfrow=c(3,3))
for(i in 1:3){
  for(j in 1:3){
    plot(gendata$mixture_profiles[i,]
         ,fitdata$mixture_profiles[j,]
         ,pch=19
         )
  }
}

