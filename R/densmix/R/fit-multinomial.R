#' Fit multinomial density mixture using EM algorithm
#'
#' @param data One dimensional dataset to fit mixture into.
#' @param n_components N of components to fit to data.
#' @param start Optional list of initial approximations to parameters.
#' @param control List that controls how EM is fitted (tolerance, max iterations)
#'
#' @return A list of fitted parameters, weights, component bayes_probilities, iterations, convergence, loglikelihood and deltas
#' @export
fit_multinomial <- function(data,
                            n_components = 3,
                            start = NULL,
                            control = list()

){

  if(any(is.na(data)) || !is.numeric(data) || any(is.infinite(data)) || any(data<0)){
    stop("`data` should be numeric and contain (a) no NA values, (b) no infinite values and (c) only positive values.")
  }


  control <- modifyList(
    em_control(),
    control
  )


  n_buckets <- dim(data)[2]
  n_obs <- dim(data)[1]

  datacolsums <- colSums(data)
  totalobs <- sum(data)

  if(is.null(start)){

    dirichlet = rep(1,n_buckets)
    mixture_profiles <- get_dirichlet(
      n_components,
      dparam = dirichlet
    )

    start <- list(
      "mixture_profiles" = mixture_profiles,
      "weights" = rep(1/n_components,n_components)
    )
  }

  parameters <- validate_multinom_parameters(start)


  converged <- FALSE
  i <- 1
  while(i <= control[['max_iter']]){

    # E-step
    bayes_probs_tmp <-
      lapply(
        seq(1:n_components),
        function(j){
          parameters[['weights']][j]*
            apply(data,
                  1,
                  function(z){
                    dmultinom(x=z, prob=parameters[['mixture_profiles']][j,])
                  })
        }
      )

    bayes_probs_tmp <- do.call(cbind,bayes_probs_tmp)
    bayes_probs <- bayes_probs_tmp / rowSums(bayes_probs_tmp)
    bayescolsums <- colSums(bayes_probs)

    # M-step
    mixture_profiles_tmp <- lapply(
      seq(1:n_components),
      function(j){
        z <- colSums(data * as.vector(bayes_probs[,j]))
        z / sum(z)
      }
    )

    new_parameters <- list(
      mixture_profiles = do.call(rbind,mixture_profiles_tmp),
      weights = bayescolsums /n_obs
    )

    profiles_delta <- abs(
      -1+
      sqrt(sum(parameters[['mixture_profiles']]^2))/
        sqrt(sum(new_parameters[['mixture_profiles']]^2))
    )

    weights_delta <- mean(
      abs(parameters[['weights']] - new_parameters[['weights']])
      /abs(parameters[['weights']]))

    total_delta <- profiles_delta+weights_delta

    if (control$verbose) {
      json <- jsonlite::toJSON(
        list(
          "weights" = parameters[['weights']],
          "profiles_delta" = profiles_delta,
          "weights_delta" = weights_delta
          ),
        auto_unbox = TRUE,
        pretty = FALSE
      )

      cat("\r\033[2K", "Iter: ", i, " ", json, sep = "")
      flush.console()
    }


    parameters <- new_parameters

    if(total_delta < control['tolerance']){
      converged <- TRUE
      break
    }

    i <- i+1
  }

  structure(
    list(
      mixture_profiles = parameters[['mixture_profiles']],
      weights = parameters[['weights']],
      bayes_probs = bayes_probs,
      iterations = i,
      converged = converged,
      profiles_delta = profiles_delta,
      data = data
    ),
    class = c("em_multinomial_fit", "em_fit")
  )

}
