multinomial_ui <- function(id) {
  ns <- NS(id)

  sidebarLayout(
    sidebarPanel(
      numericInput(
        ns("size"),
        "Sample size",
        value = 1000,
        min = 100,
        step = 100
      ),
      numericInput(
        ns("n_components"),
        "Number of components",
        value = 3,
        min = 2,
        max = 6,
        step = 1
      ),
      numericInput(
        ns("n_buckets"),
        "Number of buckets",
        value = 12,
        min = 2,
        max = 30,
        step = 1
      ),
      numericInput(
        ns("dirichlet_alpha"),
        "Dirichlet concentration",
        value = 1,
        min = 0.05,
        step = 0.1
      ),
      numericInput(
        ns("max_actions"),
        "Maximum actions",
        value = 100,
        min = 1,
        step = 10
      ),
      sliderInput(
        ns("prob_action"),
        "Action probability",
        min = 0.01,
        max = 1,
        value = 0.5,
        step = 0.01
      ),
      numericInput(
        ns("seed"),
        "Random seed",
        value = 123,
        min = 0,
        step = 1
      ),
      actionButton(
        ns("fit"),
        "Generate and fit"
      ),
      helpText(
        "Generated components use equal weights. ",
        "Component labels are arbitrary and may be permuted after fitting."
      )
    ),
    mainPanel(
      plotOutput(ns("model_plot")),
      verbatimTextOutput(ns("results"))
    )
  )
}

multinomial_server <- function(id) {
  moduleServer(id, function(input, output, session) {
    result <- eventReactive(input$fit, {
      n_components <- as.integer(input$n_components)
      n_buckets <- as.integer(input$n_buckets)

      validate(
        need(
          input$size >= n_components,
          "Sample size must be at least the number of components."
        ),
        need(
          input$max_actions * input$prob_action >= 1,
          "Increase maximum actions or action probability."
        )
      )

      set.seed(as.integer(input$seed))

      parameters <- list(
        n_components = n_components,
        n_buckets = n_buckets,
        dirichlet = rep(input$dirichlet_alpha, n_buckets),
        component_weights = rep(1 / n_components, n_components),
        n_actions_per_bucket = list(
          max_actions = as.integer(input$max_actions),
          prob_action = input$prob_action
        )
      )

      generated <- gen_em_mn(
        size = as.integer(input$size),
        params = parameters
      )

      validate(
        need(
          all(rowSums(generated$data) > 0),
          paste(
            "The selected settings generated zero-action observations.",
            "Increase maximum actions or action probability."
          )
        )
      )

      fitted <- fit_multinomial(
        data = generated$data,
        n_components = n_components
      )

      list(
        parameters = parameters,
        generated = generated,
        sampled_weights = tabulate(
          generated$component_labels,
          nbins = n_components
        ) / input$size,
        fitted = fitted
      )
    })

    output$model_plot <- renderPlot({
      plot(result()$fitted)
    })

    output$results <- renderPrint({
      value <- result()

      cat("Component labels are arbitrary and may be permuted.\n\n")

      print(
        list(
          requested_weights = round(
            value$parameters$component_weights,
            4
          ),
          sampled_weights = round(
            value$sampled_weights,
            4
          ),
          fitted_weights = round(
            value$fitted$weights,
            4
          ),
          converged = value$fitted$converged,
          iterations = value$fitted$iterations,
          generated_profiles = round(
            value$generated$mixture_profiles,
            4
          ),
          fitted_profiles = round(
            value$fitted$mixture_profiles,
            4
          )
        )
      )
    })
  })
}
