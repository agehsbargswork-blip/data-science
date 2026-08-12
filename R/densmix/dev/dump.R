# params <- list(
#   weight=0.5,
#   lambda=0.6,
#   mu=10,
#   sigma=2
# )
#
# data <- gen_exp_normal(size=1000,params)
# emfit <- fit_exp_normal(data)
# plot(emfit)


em_mn_fit_labels_list <- lapply(
  em_nm_fit_list,
  getlabels <- function(l){
    apply(l$bayes_probs,1,which.max)
  })

em_mn_true_labels_list <- lapply(
  em_nm_gen_list,
  getlabels <- function(l){
    l$component_labels
  })


library(clue)
em_mn_acc_weights_list <-
  lapply(
    seq_len(n_sim),
    getacc <- function(j){
      confusion_table <- table(
        em_mn_true_labels_list[[j]],
        em_mn_fit_labels_list[[j]]
        )
      permut <- clue::solve_LSAP(confusion_table
                                 , maximum = TRUE)
      list(
        "accuracy" = sum(diag(confusion_table[,permut])) / sum(confusion_table),
        "weights" = em_nm_fit_list[[j]]$weights[permut]
      )
    })

accuracy <-
  unlist(
    lapply(
      em_mn_acc_weights_list,
      z <- function(l){
        l$accuracy
      })
  )

par(mfrow=c(1,1))
hist(accuracy,breaks=20)

weights <-
  do.call(
    rbind,
    lapply(
      em_mn_acc_weights_list,
      z <- function(l){
        l$weights
      })
  )

par(mfrow=c(1,3))
hist(weights[,1])
hist(weights[,2])
hist(weights[,3])


