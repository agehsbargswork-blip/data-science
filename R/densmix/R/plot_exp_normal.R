#' Plot output of EM fit to 1-dim exp-normal density using EM.
#'
#' @param em_exp_norm_list Output of fit_exp_normal()
#'
#' @return NULL
#' @export
plot_exp_norm <- function(em_exp_norm_list){
  par(mfrow=c(2,2))
  hist(em_exp_norm_list$data,breaks=100,col="darkgrey")
  plot(1:em_exp_norm_list$iterations
       ,em_exp_norm_list$loglik
       ,pch=19
       ,col="purple"
       ,xlab="Iterations"
       ,ylab="log Likelihood")
  plot(em_exp_norm_list$data
       ,em_exp_norm_list$bayes_probs[,1]
       ,pch=19
       ,cex=0.5
       ,col=rgb(0.2,0.2,0.2,0.5)
       ,xlab="Data points (values)"
       ,ylab="Prob of Exp. component."
  )
  plot(em_exp_norm_list$data
       ,em_exp_norm_list$bayes_probs[,2]
       ,pch=19
       ,cex=0.5
       ,col=rgb(0.2,0.2,0.2,0.5)
       ,xlab="Data points (values)"
       ,ylab="Prob of to Norm. component.")

}
