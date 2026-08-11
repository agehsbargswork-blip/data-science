#' Validate that parameters for exp-normal density are specified correctly
#'
#' @param start List with parameters
#'
#' @return parameters as vector
#' @export
validate_exp_normal_parameters <- function(start) {
  required <- c("weight", "lambda", "mu", "sigma")

  if (!is.list(start)) {
    stop("`start` must be a named list.", call. = FALSE)
  }

  missing <- setdiff(required, names(start))

  if (length(missing) > 0L) {
    stop(
      "Some parameter(s) is/are missing: ",
      paste(missing, collapse = ", "),
      ".",
      call. = FALSE
    )
  }

  parameters <- unlist(start[required])

  if(
    parameters['weight'] <= 0 ||
    parameters['weight'] >= 1 ||
    parameters['lambda'] <= 0 ||
    parameters['mu'] <= 0 ||
    parameters['sigma'] <= 0
  ){
    stop(
      paste("Wrong parameters of mixture specified. Must be: 0<weight<1, lambda>0, mu>0, sd>0.")
    )
  }


  parameters

}


#' Validate that parameters for multinomial algo are specified correctly
#'
#' @param start List with parameters
#'
#' @return parameters as list
#' @export
validate_multinom_parameters <- function(start){

  required <- c("mixture_profiles","weights")

  if (!is.list(start)) {
    stop("`start` must be a named list.", call. = FALSE)
  }

  missing <- setdiff(required, names(start))

  if (length(missing) > 0L) {
    stop(
      "Some parameter(s) is/are missing: ",
      paste(missing, collapse = ", "),
      ".",
      call. = FALSE
    )
  }

  parameters <- start[required]

  if(
    any(is.na(parameters[['weights']])) ||
    any(is.na(parameters[['mixture_profiles']])) ||
    any(is.infinite(parameters[['weights']])) ||
    any(is.infinite(parameters[['mixture_profiles']])) ||
    any(parameters[['weights']] <= 0) ||
    any(parameters[['mixture_profiles']] <= 0) ||
    any(parameters[['weights']] >=1) ||
    any(parameters[['mixture_profiles']] >=1)
  ){
    stop(
      paste("Wrong parameters of mixture specified. Must be: 0<weights<1, 0<mixture_profiles<1.")
    )
  }


  parameters


}

