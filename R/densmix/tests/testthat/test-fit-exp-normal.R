test_that("fit_exp_normal() returns a fitted EM object", {
  set.seed(123)

  data <- gen_exp_normal(
    size = 1000,
    parameters = list(
      weight = 0.4,
      lambda = 0.5,
      mu = 10,
      sigma = 2
    )
  )

  fit <- fit_exp_normal(data)

  expect_s3_class(fit, "em_exp_normal_fit")
  expect_s3_class(fit, "em_fit")

  expect_true(is.numeric(fit$parameters))
  expect_true(is.numeric(fit$bayes_probs))
  expect_true(is.numeric(fit$loglik))
  expect_true(is.numeric(fit$iterations))
  expect_true(is.logical(fit$converged))

  expect_equal(sum(fit$bayes_probs), length(data))
  expect_true(all(fit$bayes_probs >= 0))
  expect_true(all(fit$bayes_probs <= 1))
})


test_that("fit_exp_normal() approximately recovers parameters", {
  set.seed(123)

  data <- gen_exp_normal(
    size = 5000,
    parameters = list(
      weight = 0.4,
      lambda = 0.5,
      mu = 10,
      sigma = 2
    )
  )

  fit <- fit_exp_normal(data)
  estimates <- fit$parameters

  expect_equal(as.numeric(estimates['weight']), 0.4, tolerance = 0.02)
  expect_equal(as.numeric(estimates["lambda"]), 0.5, tolerance = 0.02)
  expect_equal(as.numeric(estimates["mu"]), 10, tolerance = 0.03)
  expect_equal(as.numeric(estimates["sigma"]), 2, tolerance = 0.03)
})


test_that("fit_exp_normal() remains finite when densities underflow", {
  data <- c(
    seq(0.1, 2, length.out = 50),
    seq(9, 11, length.out = 50),
    1e6
  )

  fit <- fit_exp_normal(
    data,
    start = list(
      weight = 0.5,
      lambda = 1,
      mu = 10,
      sigma = 1
    ),
    control = list(
      max_iter = 1,
      min_lambda = 1e-10,
      min_mu = 1e-10,
      min_sigma = 1e-10
    )
  )

  expect_true(all(is.finite(fit$loglik)))
  expect_true(all(is.finite(fit$bayes_probs)))
  expect_equal(
    rowSums(fit$bayes_probs),
    rep(1, length(data)),
    tolerance = 1e-12
  )
})
