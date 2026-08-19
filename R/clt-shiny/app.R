library(shiny)

distribution_choices <- c(
  "Bernoulli",
  "Beta",
  "Binomial",
  "Exponential",
  "Gamma",
  "Geometric",
  "Laplace",
  "Log-Normal",
  "Negative Binomial",
  "Poisson",
  "Uniform"
)

distribution_inputs <- function(distribution) {
  switch(
    distribution,
    "Bernoulli" = numericInput("prob", "Probability", value = 0.5, min = 0, max = 1, step = 0.05),
    "Beta" = tagList(
      numericInput("shape1", "Shape 1", value = 2, min = 0.01, step = 0.1),
      numericInput("shape2", "Shape 2", value = 2, min = 0.01, step = 0.1)
    ),
    "Binomial" = tagList(
      numericInput("trials", "Number of trials", value = 10, min = 0, step = 1),
      numericInput("prob", "Probability", value = 0.5, min = 0, max = 1, step = 0.05)
    ),
    "Exponential" = numericInput("rate", "Rate", value = 1, min = 0.01, step = 0.1),
    "Gamma" = tagList(
      numericInput("shape", "Shape", value = 2, min = 0.01, step = 0.1),
      numericInput("rate", "Rate", value = 1, min = 0.01, step = 0.1)
    ),
    "Geometric" = numericInput("prob", "Probability", value = 0.5, min = 0.01, max = 1, step = 0.05),
    "Laplace" = tagList(
      numericInput("location", "Location", value = 0, step = 0.1),
      numericInput("scale", "Scale", value = 1, min = 0.01, step = 0.1)
    ),
    "Log-Normal" = tagList(
      numericInput("meanlog", "Mean log", value = 0, step = 0.1),
      numericInput("sdlog", "SD log", value = 1, min = 0, step = 0.1)
    ),
    "Negative Binomial" = tagList(
      numericInput("size", "Size", value = 10, min = 0.01, step = 0.1),
      numericInput("prob", "Probability", value = 0.5, min = 0.01, max = 1, step = 0.05)
    ),
    "Poisson" = numericInput("lambda", "Lambda", value = 5, min = 0, step = 0.1),
    "Uniform" = tagList(
      numericInput("minimum", "Minimum", value = 0, step = 0.1),
      numericInput("maximum", "Maximum", value = 1, step = 0.1)
    )
  )
}

ui <- fluidPage(
  tags$head(
    tags$style(HTML(
      ".plot-panel { padding: 10px; }\n       .control-label { font-weight: 600; }\n       .btn-primary { width: 100%; margin-top: 8px; }"
    ))
  ),
  titlePanel("Central Limit Theorem Simulator"),
  sidebarLayout(
    sidebarPanel(
      numericInput("sample_size", "Dataset size (N)", value = 30, min = 1, max = 10000, step = 1),
      numericInput("resamples", "Number of resamples (B)", value = 1000, min = 2, max = 10000, step = 100),
      selectInput("distribution", "Distribution", choices = distribution_choices),
      uiOutput("distribution_parameters"),
      actionButton("generate", "Generate", class = "btn-primary")
    ),
    mainPanel(
      fluidRow(
        column(
          width = 6,
          div(class = "plot-panel", plotOutput("mean_histogram", height = "480px"))
        ),
        column(
          width = 6,
          div(class = "plot-panel", plotOutput("normal_qq", height = "480px"))
        )
      )
    )
  )
)

server <- function(input, output, session) {
  output$distribution_parameters <- renderUI({
    distribution_inputs(input$distribution)
  })

  sample_means <- eventReactive(input$generate, {
    n <- input$sample_size
    b <- input$resamples

    validate(
      need(n >= 1 && n == as.integer(n), "N must be a positive integer."),
      need(b >= 2 && b == as.integer(b), "B must be an integer of at least 2.")
    )

    generator <- switch(
      input$distribution,
      "Bernoulli" = {
        validate(need(input$prob >= 0 && input$prob <= 1, "Probability must be between 0 and 1."))
        function(size) rbinom(size, size = 1, prob = input$prob)
      },
      "Beta" = {
        validate(need(input$shape1 > 0 && input$shape2 > 0, "Both shape parameters must be positive."))
        function(size) rbeta(size, shape1 = input$shape1, shape2 = input$shape2)
      },
      "Binomial" = {
        validate(
          need(input$trials >= 0 && input$trials == as.integer(input$trials), "Number of trials must be a non-negative integer."),
          need(input$prob >= 0 && input$prob <= 1, "Probability must be between 0 and 1.")
        )
        function(size) rbinom(size, size = input$trials, prob = input$prob)
      },
      "Exponential" = {
        validate(need(input$rate > 0, "Rate must be positive."))
        function(size) rexp(size, rate = input$rate)
      },
      "Gamma" = {
        validate(need(input$shape > 0 && input$rate > 0, "Shape and rate must be positive."))
        function(size) rgamma(size, shape = input$shape, rate = input$rate)
      },
      "Geometric" = {
        validate(need(input$prob > 0 && input$prob <= 1, "Probability must be greater than 0 and at most 1."))
        function(size) rgeom(size, prob = input$prob)
      },
      "Laplace" = {
        validate(need(input$scale > 0, "Scale must be positive."))
        function(size) {
          u <- runif(size, min = -0.5, max = 0.5)
          input$location - input$scale * sign(u) * log(1 - 2 * abs(u))
        }
      },
      "Log-Normal" = {
        validate(need(input$sdlog >= 0, "SD log must be non-negative."))
        function(size) rlnorm(size, meanlog = input$meanlog, sdlog = input$sdlog)
      },
      "Negative Binomial" = {
        validate(
          need(input$size > 0, "Size must be positive."),
          need(input$prob > 0 && input$prob <= 1, "Probability must be greater than 0 and at most 1.")
        )
        function(size) rnbinom(size, size = input$size, prob = input$prob)
      },
      "Poisson" = {
        validate(need(input$lambda >= 0, "Lambda must be non-negative."))
        function(size) rpois(size, lambda = input$lambda)
      },
      "Uniform" = {
        validate(need(input$maximum > input$minimum, "Maximum must be greater than minimum."))
        function(size) runif(size, min = input$minimum, max = input$maximum)
      }
    )

    withProgress(message = "Generating samples", value = 0, {
      means <- numeric(b)
      for (i in seq_len(b)) {
        means[i] <- mean(generator(n))
        if (i %% max(1, floor(b / 100)) == 0 || i == b) {
          setProgress(i / b)
        }
      }
      list(means = means, n = n, b = b, distribution = input$distribution)
    })
  }, ignoreInit = TRUE)

  output$mean_histogram <- renderPlot({
    simulation <- sample_means()
    means <- simulation$means
    hist(
      means,
      breaks = "FD",
      col = "#4C78A8",
      border = "white",
      main = sprintf(
        "%s sample means\nN = %d, B = %d",
        simulation$distribution,
        simulation$n,
        simulation$b
      ),
      xlab = "Sample mean"
    )
    abline(v = mean(means), col = "#E45756", lwd = 2, lty = 2)
  })

  output$normal_qq <- renderPlot({
    means <- sample_means()$means
    qqnorm(
      means,
      pch = 19,
      cex = 0.65,
      col = grDevices::adjustcolor("#4C78A8", alpha.f = 0.65),
      main = "Normal Q-Q plot"
    )
    qqline(means, col = "#E45756", lwd = 2)
  })
}

shinyApp(ui, server)
