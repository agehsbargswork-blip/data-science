#' Fit one dimensional exponential-normal density using EM algorithm
#'
#' @param data One dimensional dataset to fit mixture into.
#' @param start Optional list of initial approximations to parameters.
#' @param control List that controls how EM is fitted (tolerance, max iterations)
#'
#' @return A list of fitted parameters, weights, component bayes_probilities, iterations, convergence, loglikelihood and deltas
#' @export
fit_exp_normal <- function(data,
                           start = NULL,
                           control = list()){

  if(any(is.na(data)) || !is.numeric(data) || any(is.infinite(data)) || any(data<0)){
    stop("`data` should be numeric and contain (a) no NA values, (b) no infinite values and (c) only positive values.")
  }

  qqs <- as.numeric(quantile(data))
  if(is.null(start)){
    start <- list(
      weight = 0.5,
      lambda = qqs[2],
      mu = qqs[4],
      sigma = sd(data[data>qqs[3]])
    )
  }

  control <- modifyList(
    em_control(),
    control
  )

  control <- modifyList(
    list(
      min_lambda = 1e-3,
      min_mu = 1e-3,
      min_sigma = 1e-3
    ),
    control
  )

  parameters <- validate_exp_normal_parameters(start)

  data_sum <- sum(data)
  n <- length(data)


  converged <- FALSE
  loglik <- c()
  i <- 1

  while( i <= control['max_iter'] ) {

    expd <- parameters['weight'] * dexp(data, rate = parameters['lambda'])
    normd <- (1-parameters['weight']) * dnorm(data, mean = parameters['mu'], sd = parameters['sigma'])
    loglik <- c(loglik,sum(log(expd+normd,exp(1))))


    bayes_probs <- expd / (expd + normd)
    total_prob <- sum(bayes_probs)
    weighted_data <- sum(bayes_probs*data)

    new_parameters <- c(
      weight = total_prob / n,
      lambda = total_prob / weighted_data,
      mu = (data_sum - weighted_data)/ (n - total_prob),
      sigma = NA
    )

    new_parameters['sigma'] = sqrt( sum( (1-bayes_probs)*(data-new_parameters['mu'])^2 ) / (n - total_prob) )

    if(
      new_parameters['weight'] < control['min_weight'] ||
      new_parameters['lambda'] < control['min_lambda'] ||
      new_parameters['mu'] < control['min_mu'] ||
      new_parameters['sigma'] < control['min_sigma']
    ){
      min_params <- c('min_weight','min_lambda','min_mu','min_sigma')
      stop(
        cat(
          "Min tolerance for parameters reached: \n",
          jsonlite::toJSON(
            control[min_params],
            dataframe = "rows",
            pretty = TRUE
          )
        )
      )
    }

    deltas <- abs(parameters - new_parameters) / abs(parameters)
    total_delta <- sum(deltas)

    parameters <- new_parameters
    if (control$verbose) {
      json <- jsonlite::toJSON(
        as.list(parameters),
        auto_unbox = TRUE,
        pretty = FALSE
      )

      cat("\r\033[2K", "Iter: ", i, " ", json, sep = "")
      flush.console()
    }

    if(total_delta < control['tolerance']){
      converged <- TRUE
      expd <- parameters['weight'] * dexp(data, rate = parameters['lambda'])
      normd <- (1-parameters['weight']) * dnorm(data, mean = parameters['mu'], sd = parameters['sigma'])
      loglik <- c(loglik,sum(log(expd+normd,exp(1))))
      i <- i+1
      break
    }



    i <- i+1
  }

  structure(
    list(
      parameters = parameters,
      weights = c(
        exponential = parameters["weight"],
        normal = 1 - parameters["weight"]
      ),
      bayes_probs = cbind(
        exponential = bayes_probs,
        normal = 1 - bayes_probs
      ),
      iterations = i,
      converged = converged,
      loglik = loglik,
      deltas = deltas,
      total_delta = total_delta,
      data = data
    ),
    class = "em_exp_normal"
  )
}
