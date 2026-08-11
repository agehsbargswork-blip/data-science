# That is a setup file for the package


usethis::use_build_ignore("^dev$")


# Create package source files
usethis::use_r("fit_exp_normal")
usethis::use_r("fit_multinomial")


usethis::use_r("simulate_exp_normal")
usethis::use_r("simulate_multinomial")


usethis::use_r("plot_exp_normal")
usethis::use_r("plot_multinomial")

usethis::use_r("control")

# usethis::use_testthat()

usethis::use_test("control")
usethis::use_test("validation")
usethis::use_test("fit-exp-normal")
usethis::use_test("fit-multinomial")

usethis::use_readme_rmd()


# ---------------------------------------

devtools::document()
attachment::att_amend_desc(
  update.config = TRUE
)
devtools::load_all()
devtools::check()

#
#
# devtools::test()


# ---------------------------------------


devtools::build()
devtools::install()

install.packages(
  "C:/Users/agehs/git/data-science/R/densmix_0.1.0.tar.gz",
  repos = NULL,
  type = "source"
)

devtools::build_readme()
