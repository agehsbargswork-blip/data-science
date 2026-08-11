#' Generate one dimensional exponential-normal density
#'
#' @param size Size of the dataset to generate.
#' @param parameters List of parameters.
#'
#' @return Vector generated from the mixture.
#' @export
gen_exp_normal <- function(size, parameters){

  parameters <- validate_exp_normal_parameters(parameters)

  n1 <- rbinom(n=1,size=size,prob=parameters['weight'])
  n2 <- size - n1

  s1 <- rexp(n1,rate = parameters['lambda'])
  s2 <- rnorm(n2,mean=parameters['mu'],sd=parameters['sigma'])

  res <- c(s1,s2)

  return(res)

}
