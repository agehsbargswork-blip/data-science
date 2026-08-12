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
n_sim <- 5
em_nm_gen_list <-
  lapply(
    seq_len(n_sim),
    getnm <- function(j){
      gen_em_mn(size = 1000,
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
    }
  )


em_nm_fit_list <-
  lapply(
    em_nm_gen_list,
    fitem <- function(l){
      fit_multinomial(l$data, control = list(verbose=TRUE))
    })

x <- apply(fitdata$bayes_probs,1,which.max)
y <- gendata$component_labels
confusion_table <- table(x,y)

permut <- clue::solve_LSAP(confusion_table
                           , maximum = TRUE)

confusion_table[1:3,permut]


x[1:10]

library(densmix)

set.seed(123)
n_sim <- 200
exp_norm_param_list <-
  lapply(
    seq_len(n_sim),
    gen_exp_norm_params <- function(j){
      list(
        weight=runif(1,0.5,0.9),
        lambda=runif(1,0.2,0.3),
        mu=runif(1,10,20),
        sigma=runif(1,1,2)
      )
    }
  )

exp_norm_gen_datasets_list <-
  lapply(
    exp_norm_param_list,
    gexpnorm_list <- function(l){
      gen_exp_normal(size=1000,parameters=l)
    }
  )

exp_norm_fits_list <-
  lapply(
    exp_norm_gen_datasets_list,
    gexpnorm_list <- function(d){
      fit_exp_normal(d)
    }
  )

exp_norm_true_param_df <- as.data.frame(
  do.call(rbind, exp_norm_param_list)
)

exp_norm_fit_param_df <-
  as.data.frame(
    do.call(
      rbind,
      lapply(
        exp_norm_fits_list,
        get_param <- function(l){
          l$parameters
        }
      )
    )
  )

par(mfrow=c(2,2))
cnames <- c("weight", "lambda", "mu", "sigma")
for(cname in cnames){
  plot(
    exp_norm_true_param_df[[cname]],
    exp_norm_fit_param_df[[cname]],
    xlab = "True value",
    ylab = "Fitted value",
    main = cname,
    pch = 19,
    col = "purple"
  )
}

par(mfrow=c(2,3))
hist(exp_norm_gen_datasets_list[[1]],main=1,breaks=30)
hist(exp_norm_gen_datasets_list[[5]],main=5,breaks=30)
hist(exp_norm_gen_datasets_list[[10]],main=10,breaks=30)
hist(exp_norm_gen_datasets_list[[100]],main=100,breaks=30)
hist(exp_norm_gen_datasets_list[[150]],main=150,breaks=30)
hist(exp_norm_gen_datasets_list[[200]],main=200,breaks=30)

iter_fit <-
  do.call(
    rbind,
    lapply(
      exp_norm_fits_list,
      get_param <- function(l){
        l$iterations
      }
    )
  )
hist(iter_fit)
quantile(iter_fit)
