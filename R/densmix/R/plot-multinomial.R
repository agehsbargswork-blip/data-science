#' Plot output of EM fit to 1-dim exp-normal density using EM.
#'
#' @param x A fitted exponential-normal mixture model.
#' @param ... Additional graphical arguments.
#'
#' @return The fitted model, invisibly.
#' @export
plot.em_multinomial_fit <- function(x, ...){
  par(mfrow=c(2,1))
  image(
    x = seq(1:ncol(x$mixture_profiles)),
    y = seq(1:nrow(x$mixture_profiles)),
    z = t(x$mixture_profiles),
    xlab = "Buckets",
    ylab = "Components",
    main = "Mixture profiles",
    axes = FALSE
  )

  axis(1, at = seq_len(ncol(x$mixture_profiles)))
  axis(2, at = seq_len(nrow(x$mixture_profiles)))

  barplot(
    x$weights,
    names.arg = seq_along(x$weights),
    xlab = "Components",
    ylab = "Component weights",
    main = "Mixture weights"
  )

  invisible(x)

}
