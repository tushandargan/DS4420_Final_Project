# app.R

library(shiny)
library(neuralnet)
library(caret)
library(dplyr)
library(ggplot2)
library(pROC)
library(DT)

data <- read.csv("Lung_Cancer_Dataset.csv", stringsAsFactors = FALSE)

data$LungCancer <- ifelse(data$PULMONARY_DISEASE == "YES", 1, 0)

binary_cols <- setdiff(names(data), c("AGE", "ENERGY_LEVEL", "OXYGEN_SATURATION", "PULMONARY_DISEASE", "LungCancer"))
data[binary_cols] <- lapply(data[binary_cols], function(x) as.numeric(as.character(x)))

normalize <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}
data[, c("AGE", "ENERGY_LEVEL", "OXYGEN_SATURATION")] <- lapply(data[, c("AGE", "ENERGY_LEVEL", "OXYGEN_SATURATION")], normalize)

set.seed(123)
train_index <- createDataPartition(data$LungCancer, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data  <- data[-train_index, ]

predictor_names <- setdiff(names(train_data), c("PULMONARY_DISEASE", "LungCancer"))
formula <- as.formula(paste("LungCancer ~", paste(predictor_names, collapse = " + ")))

nn_model <- neuralnet(formula,
                      data = train_data,
                      hidden = 5,           
                      linear.output = FALSE,
                      stepmax = 1e6)

predictions <- neuralnet::compute(nn_model, test_data[, predictor_names])
predicted_probabilities <- as.vector(predictions$net.result)


ui <- navbarPage("Cancer Prediction App",
                 
                 tabPanel("Home",
                          fluidPage(
                            titlePanel("Cancer Prediction Project"),
                            sidebarLayout(
                              sidebarPanel(
                                h3("Project Description"),
                                p("In this project we created a CNN to detect which type of skin cancer is in an image as well as an MLP to predict lung cancer outcomes based on various health metrics. 
             In the MLP model, the data is preprocessed and normalized; then, the model is trained and evaluated using different performance metrics.
             Use the interactive page to explore the the MLP model for lung cancer prediction and how the impact of choosing different classification thresholds on the confusion matrix.")
                              ),
                              mainPanel(
                                h4("Overview"),
                                p("Navigate to the 'Visualization' tab to interact with the MLP model output. 
              You can adjust the threshold for converting predicted probabilities into class labels and see how it affects the confusion matrix counts.")
                              )
                            )
                          )
                 ),
                 
                 tabPanel("Visualization",
                          fluidPage(
                            titlePanel("Interactive Confusion Matrix"),
                            sidebarLayout(
                              sidebarPanel(
                                sliderInput("threshold", "Classification Threshold:",
                                            min = 0, max = 1, value = 0.5, step = 0.01,
                                            animate = animationOptions(interval = 1500, loop = TRUE)),
                                p("Move the slider to change the threshold value used to classify the predicted probabilities.")
                              ),
                              mainPanel(
                                plotOutput("confusionPlot"),
                                br(),
                                DT::dataTableOutput("confusionTable")
                              )
                            )
                          )
                 )
)

server <- function(input, output, session) {
  
  reactive_confusion <- reactive({
    thr <- input$threshold
    predicted_class <- ifelse(predicted_probabilities > thr, 1, 0)
    cm <- table(Predicted = predicted_class, Actual = test_data$LungCancer)
    
    cm_df <- as.data.frame(cm)
    names(cm_df) <- c("Predicted", "Actual", "Count")
    cm_df$Actual <- factor(cm_df$Actual, levels = c(0, 1),
                           labels = c("No Pulmonary Disease", "High likelihood for lung cancer"))
    cm_df$Predicted <- factor(cm_df$Predicted, levels = c(0, 1),
                              labels = c("No Pulmonary Disease", "Low likelihood for lung cancer"))
    cm_df
  })
  
  output$confusionPlot <- renderPlot({
    cm_df <- reactive_confusion()
    ggplot(cm_df, aes(x = Actual, y = Count, fill = Predicted)) +
      geom_bar(stat = "identity", position = "dodge") +
      labs(title = "Confusion Matrix Counts",
           x = "Actual Condition",
           y = "Count") +
      theme_minimal()
  })
  
  output$confusionTable <- DT::renderDataTable({
    reactive_confusion()
  })
}

shinyApp(ui, server)
