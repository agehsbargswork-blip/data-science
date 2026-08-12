#' Generate one vector from Diriclhet distribution
#'
#' @param n dimensionality of Dirichlet
#' @param dparam Parameters of Dirichlet.
#'
#' @return One vector from specified Dirichlet distribution.
#' @export
get_dirichlet <- function(n, dparam){
  gamma_samples <-
    sapply(dparam
           , function(y){
             rgamma(n,shape = y, scale = 1)
           }
    )

  if(n==1){
    gamma_samples / sum(gamma_samples)
  } else {
    gamma_samples / rowSums(gamma_samples)
  }

}

#' Generate dataset from mixture of multinomials
#'
#' @param size Dataset size, total number of observations to generate
#' @param params List with parameters for generation (n of components, n of buckets, component weights, parameters of Dirichlet, and desired n actions.
#'
#' @return A list with dataset, mixture_profiles and component_labels
#' @export
gen_em_mn <- function(size = 1000,
                      params = list("n_components" = 3,
                                    "n_buckets" = 12,
                                    "dirichlet" = rep(1,12),
                                    "component_weights" = c(0.1,0.2,0.7),
                                    "n_actions_per_bucket" = list(
                                      "max_actions" = 300,
                                      "prob_action" = 0.2
                                    )
                      )
)
{

  vec_params <- unlist(params)
  if(any(is.na(vec_params)) || !is.numeric(vec_params) || any(is.infinite(vec_params)) || any(vec_params<0)){
    stop("`params` must be a vector of numeric paramaters. NA values, infinite values and negative values are not allowed.")
  }

  if(params[['n_components']] != length(params[['component_weights']])){
    stop("Length of `component_weights` must be equal to `n_components`.")
  }

  if(params[['n_buckets']] != length(params[['dirichlet']])){
    stop("Length of `dirichlet` must be equal to `n_buckets`.")
  }


  mixture_profiles <- get_dirichlet(
    params[['n_components']],
    dparam = params[['dirichlet']]
  )

  mixture_n_obs <- rmultinom(1,
                             size,
                             prob=params[["component_weights"]]
  )

  actions_per_component <- rbinom(params[["n_components"]]
                                  , params[["n_actions_per_bucket"]][["max_actions"]]
                                  , params[["n_actions_per_bucket"]][["prob_action"]]
  )

  component_labels <- rep(seq(1:params[['n_components']])
                          ,times=as.vector(mixture_n_obs))

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

  shuffle <- sample(1:size)
  component_labels <- component_labels[shuffle]
  data <- data[shuffle,]


  list(
    data = data,
    mixture_profiles = mixture_profiles,
    component_labels = component_labels
  )

}
