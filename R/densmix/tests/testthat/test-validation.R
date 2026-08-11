valid_parameters <- list(
  weight = 0.4,
  lambda = 0.5,
  mu = 10,
  sigma = 2
)

test_that("valid exponential-normal parameters are accepted", {
  result <- validate_exp_normal_parameters(valid_parameters)

  expect_type(result, "double")

  expect_equal(
    result,
    c(
      weight = 0.4,
      lambda = 0.5,
      mu = 10,
      sigma = 2
    )
  )
})


test_that("`start` must be a named list.", {
  parameters <- unlist(valid_parameters)

  expect_error(
    validate_exp_normal_parameters(parameters),
    "`start` must be a named list."
  )
})


test_that("missing parameters are rejected", {
  parameters <- valid_parameters
  parameters$sigma <- NULL

  expect_error(
    validate_exp_normal_parameters(parameters),
    "missing: sigma"
  )
})

test_that("parameters must be finite numeric scalars", {
  parameters <- valid_parameters
  parameters$lambda <- -1

  expect_error(
    validate_exp_normal_parameters(parameters),
    "Wrong parameters of mixture specified. Must be: 0<weight<1, lambda>0, mu>0, sd>0."
  )
})

test_that("parameters outside permitted ranges are rejected", {
  parameters <- valid_parameters
  parameters$weight <- 1.2

  expect_error(
    validate_exp_normal_parameters(parameters),
    "Wrong parameters of mixture specified. Must be: 0<weight<1, lambda>0, mu>0, sd>0."
  )
})
