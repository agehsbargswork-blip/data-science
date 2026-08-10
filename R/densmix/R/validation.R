#' Validate that parameters for exp-normal density are specified correctly
#'
#' @param parameters List with parameters
#'
#' @return parameters as vector
#' @export
validate_exp_normal_parameters <- function(parameters) {
  required <- c("weight", "lambda", "mu", "sigma")

  if (!is.list(parameters)) {
    stop("`parameters` must be a named list.", call. = FALSE)
  }

  missing <- setdiff(required, names(parameters))

  if (length(missing) > 0L) {
    stop(
      "Some parameter(s) is/are missing: ",
      paste(missing, collapse = ", "),
      ".",
      call. = FALSE
    )
  }

  parameters <- unlist(parameters[required])

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
