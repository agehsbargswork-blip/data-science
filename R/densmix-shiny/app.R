library(shiny)
library(densmix)

source("R/mod_exp_normal.R", local = TRUE)
source("R/mod_multinomial.R", local = TRUE)

ui <- navbarPage(
  title = "densmix",
  tabPanel(
    "Exponential-normal",
    exp_normal_ui("exp_normal")
  ),
  tabPanel(
    "Multinomial",
    multinomial_ui("multinomial")
  )
)

server <- function(input, output, session) {
  exp_normal_server("exp_normal")
  multinomial_server("multinomial")
}

shinyApp(ui, server)
