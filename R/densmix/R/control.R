#' Configure the EM algorithm
#'
#' @param max_iter Maximum number of iterations.
#' @param tolerance Convergence tolerance (convergence assumed when delta in loglikelihood falls below tolerance).
#' @param min_weight Minimum permitted component weight.
#' @param verbose Whether to print iteration progress.
#'
#' @return A list of EM control settings.
#' @export
em_control <- function(
    max_iter = 1000L,
    tolerance = 1e-5,
    min_weight = 1e-5,
    verbose = FALSE
) {
  if(is.na(max_iter) ||
     is.infinite(max_iter) ||
     !is.numeric(max_iter) ||
     max_iter < 1 ||
     length(max_iter) != 1L){
    stop("`max_iter` must be a positive whole number.")
  }

  if(is.na(tolerance) ||
     is.infinite(tolerance) ||
     !is.numeric(tolerance) ||
     length(tolerance) != 1L ||
     tolerance <= 0 ){
    stop("`tolerance` must be a small positive real number.")
  }

  if(is.na(min_weight) ||
     is.infinite(min_weight) ||
     !is.numeric(min_weight) ||
     length(min_weight) != 1L ||
     min_weight <= 0 ||
     min_weight >= 1){
    stop("`min_weight` must be a real number between 0 and 1.")
  }

  list(
    max_iter = max_iter,
    tolerance = tolerance,
    min_weight = min_weight,
    verbose = FALSE
  )

}
