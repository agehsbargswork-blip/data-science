#' Plot output of EM fit to 1-dim exp-normal density using EM.
#'
#' @param x A fitted exponential-normal mixture model.
#' @param ... Additional graphical arguments.
#'
#' @return The fitted model, invisibly.
#' @export
plot.em_exp_normal_fit <- function(x, ...){
  par(mfrow=c(2,2))
  hist(x$data,breaks=100,col="darkgrey",xlab="data", main = paste("Histogram of data"))
  plot(1:x$iterations
       ,x$loglik
       ,pch=19
       ,col="purple"
       ,xlab="Iterations"
       ,ylab="log Likelihood")
  plot(x$data
       ,x$bayes_probs[,1]
       ,pch=19
       ,cex=0.5
       ,col=rgb(0.2,0.2,0.2,0.5)
       ,xlab="Data points (values)"
       ,ylab="Prob of Exp. component."
  )
  plot(x$data
       ,x$bayes_probs[,2]
       ,pch=19
       ,cex=0.5
       ,col=rgb(0.2,0.2,0.2,0.5)
       ,xlab="Data points (values)"
       ,ylab="Prob of to Norm. component.")

  invisible(x)

}
