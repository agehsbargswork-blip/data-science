test_that("fit_multinomial() returns a fitted EM object", {
  set.seed(123)

  gendata <- gen_em_mn(size = 1000,
                       params = list("n_components" = 3,
                                     "n_buckets" = 12,
                                     "dirichlet" = rep(1,12),
                                     "component_weights" = c(0.1,0.2,0.7),
                                     "n_actions_per_bucket" = list(
                                       "max_actions" = 30,
                                       "prob_action" = 0.5
                                     )
                       )
  )

  fitdata <- fit_multinomial(gendata$data)

  expect_s3_class(fitdata, "em_multinomial_fit")
  expect_s3_class(fitdata, "em_fit")

  expect_true(is.numeric(fitdata$mixture_profiles))
  expect_true(is.numeric(fitdata$weights))
  expect_true(is.numeric(fitdata$bayes_probs))
  expect_true(is.numeric(fitdata$iterations))
  expect_true(is.logical(fitdata$converged))

  expect_equal(sum(fitdata$bayes_probs), dim(gendata$data)[1])
  expect_true(all(fitdata$bayes_probs >= 0))
  expect_true(all(fitdata$bayes_probs <= 1))
})


test_that("fit_exp_normal() approximately recovers parameters", {
  set.seed(123)

  trueweights <- c(0.1,0.2,0.7)
  gendata <- gen_em_mn(size = 1000,
                       params = list("n_components" = 3,
                                     "n_buckets" = 12,
                                     "dirichlet" = rep(1,12),
                                     "component_weights" = trueweights,
                                     "n_actions_per_bucket" = list(
                                       "max_actions" = 30,
                                       "prob_action" = 0.5
                                     )
                       )
  )

  fitdata <- fit_multinomial(gendata$data)

  fitweights <- sort(fitdata$weights)
  trueweights <- sort(trueweights)

  expect_equal(as.numeric(fitweights), trueweights, tolerance = 0.1)

})


test_that("fit_multinomial() remains finite when probabilities underflow", {
  data <- rbind(
    c(10000, 0),
    c(0, 10000)
  )

  fit <- fit_multinomial(
    data,
    n_components = 2,
    start = list(
      mixture_profiles = rbind(
        c(0.6, 0.4),
        c(0.4, 0.6)
      ),
      weights = c(0.5, 0.5)
    ),
    control = list(max_iter = 1)
  )

  expect_true(all(is.finite(fit$bayes_probs)))
  expect_equal(
    rowSums(fit$bayes_probs),
    rep(1, nrow(data)),
    tolerance = 1e-12
  )
})
