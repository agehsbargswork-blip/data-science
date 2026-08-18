library(shiny)
library(densmix)

ui <- fluidPage(
  numericInput("size", "Sample size", 1000),
  numericInput("weight", "Exponential weight", 0.5),
  numericInput("lambda", "Lambda", 0.6),
  numericInput("mu", "Normal mean", 10),
  numericInput("sigma", "Normal SD", 2),
  actionButton("fit", "Generate and fit"),
  plotOutput("model_plot"),
  verbatimTextOutput("results")
)

server <- function(input, output, session) {

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
}

shinyApp(ui, server)
