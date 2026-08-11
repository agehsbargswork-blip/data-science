# params <- list(
#   weight=0.5,
#   lambda=0.6,
#   mu=10,
#   sigma=2
# )
#
# data <- gen_exp_normal(size=1000,params)
# emfit <- fit_exp_normal(data)
# plot(emfit)

set.seed(123)
gendata <- gen_em_mn(size = 1000,
                     params = list("n_components" = 3,
                                   "n_buckets" = 12,
                                   "dirichlet" = rep(1,12),
                                   "component_weights" = c(0.1,0.2,0.7),
                                   "n_actions_per_bucket" = list(
                                     "max_actions" = 100,
                                     "prob_action" = 0.5
                                   )
                     )
      )


fitdata <- fit_multinomial(gendata$data, control = list(verbose = TRUE))

par(mfrow=c(3,3))
for(i in 1:3){
  for(j in 1:3){
    plot(gendata$mixture_profiles[i,]
         ,fitdata$mixture_profiles[j,]
         ,pch=19
         ,xlab=paste("Generated profile ",i,sep="")
         ,ylab=paste("Fitted profile ",j,sep="")
         )
  }
}

plot(fitdata)



