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

devtools::check()
