test_that("em_control() returns its default settings", {
  control <- em_control()

  expect_equal(control$max_iter, 1000L)
  expect_equal(control$tolerance, 1e-5)
  expect_equal(control$min_weight, 1e-5)
  expect_false(control$verbose)
})

test_that("em_control() accepts custom settings", {
  control <- em_control(
    max_iter = 500,
    tolerance = 1e-6,
    min_weight = 0.001,
    verbose = TRUE
  )

  expect_equal(control$max_iter, 500L)
  expect_equal(control$tolerance, 1e-6)
  expect_equal(control$min_weight, 0.001)
  expect_true(control$verbose)
})

test_that("em_control() rejects invalid settings", {
  expect_error(
    em_control(max_iter = 0),
    "`max_iter` must be a positive whole number"
  )

  expect_error(
    em_control(tolerance = -1),
    "`tolerance` must be a small positive real number."
  )

  expect_error(
    em_control(min_weight = 1),
    "`min_weight` must be a real number between 0 and 1."
  )

  expect_error(
    em_control(verbose = "yes"),
    "`verbose` must be TRUE or FALSE"
  )
})
