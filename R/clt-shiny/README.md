# Central Limit Theorem Shiny app

This app simulates the sampling distribution of the mean. It draws `B` independent samples of size `N` from a selected distribution, then displays a histogram of the sample means and a normal Q-Q plot.

## Run locally

```r
install.packages("shiny")
shiny::runApp("R/clt-shiny")
```

The app uses base R random generators; the Laplace generator is implemented directly with inverse transform sampling.
