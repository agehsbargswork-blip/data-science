exp_normal_ui <- function(id) {
  ns <- NS(id)

  sidebarLayout(
    sidebarPanel(
      numericInput(
        ns("size"),
        "Sample size",
        value = 1000,
        min = 1,
        step = 100
      ),
      sliderInput(
        ns("weight"),
        "Exponential weight",
        min = 0.01,
        max = 0.99,
        value = 0.5,
        step = 0.01
      ),
      numericInput(
        ns("lambda"),
        "Lambda",
        value = 0.6,
        min = 0.01,
        step = 0.1
      ),
      numericInput(
        ns("mu"),
        "Normal mean",
        value = 10,
        min = 0.01,
        step = 1
      ),
      numericInput(
        ns("sigma"),
        "Normal SD",
        value = 2,
        min = 0.01,
        step = 0.5
      ),
      actionButton(
        ns("fit"),
        "Generate and fit"
      )
    ),
    mainPanel(
      plotOutput(ns("model_plot")),
      verbatimTextOutput(ns("results"))
    )
  )
}

exp_normal_server <- function(id) {
  moduleServer(id, function(input, output, session) {
    model <- eventReactive(input$fit, {
      parameters <- list(
        weight = input$weight,
        lambda = input$lambda,
        mu = input$mu,
        sigma = input$sigma
      )

      data <- gen_exp_normal(
        size = input$size,
        parameters = parameters
      )

      fit_exp_normal(data)
    })

    output$model_plot <- renderPlot({
      plot(model())
    })

    output$results <- renderPrint({
      model()$parameters
    })
  })
}
