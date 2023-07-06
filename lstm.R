library(ISLR2)
library(tree)
library(randomForest)
library(gbm)
library(caret)
library(janitor)
library(dplyr)
library(keras)
library(reticulate)
library(tensorflow)
library(MASS)
library(e1071)
library(DMwR)
library(ranger)
library(ROSE)
library(reshape2)
install.packages(c("zoo","xts","quantmod")) ## and perhaps mode
install.packages( "/Users/maxmender/Downloads/DMwR_0.4.1.tar.gz", repos=NULL, type="source" )

install_tensorflow()

# Data
trainTrans <- read.csv('')
trainI <- read.csv('')
testI <- read.csv('')
testTrans <- read.csv('')
trainData <- merge(x = trainI, y = trainTrans, by = "TransactionID", all.y = T, all.x = T)
trainData <- trainData[which(colSums(is.na(trainData))<(nrow(trainData)*0.20))] #Put in report which variables were removed
trainData <- trainData[which(colSums(trainData=="")<(nrow(trainData)*0.20))] #Put in report which variables were removed
testData <- merge(x = testI, y = testTrans, by = "TransactionID", all.y = T, all.x = T)
cols <- colnames(trainData[-c(1,2)])
test <- testData[cols]
one_hot <- function(x) {
  n <- length(unique(x))
  m <- matrix(0, nrow = length(x), ncol = n)
  m[cbind(seq_along(x), match(x, unique(x)))] <- 1
  colnames(m) <- unique(x)
  return(m)
}
trainHot <- trainData %>%
  mutate(across(where(~ !is.numeric(.) || is.matrix(.)), one_hot))
trainHot <- trainHot[,-1]
selected = sample(nrow(trainHot),size = round(nrow(trainHot)*0.75))
steps <- 50
samples <- 442905
batches = floor((samples - steps) / steps) + 1
n_samples <- nrow(trainHot)
n_timesteps <- 50
n_features <- 21
x_train <- array(trainHot[, -1], dim = c(n_samples, n_timesteps, n_features))

x_val <- array_reshape(trainHot[-samples, -1], dim = c(dim(trainHot[-samples, -1])[1], 50, 21))

# Define the model
model <- keras_model_sequential() 

model %>%
  layer_lstm(units = 32, input_shape = c(50, 21), return_sequences = TRUE) %>%
  layer_lstm(units = 16, return_sequences = TRUE) %>%
  layer_lstm(units = 8) %>%
  layer_flatten() %>%
  layer_dense(units = 32, activation = 'relu', name = 'dense') %>%
  layer_dense(units = 1, activation = 'sigmoid', name = 'accuracy')

# Compile the model
model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = 'binary_accuracy'
)

# Train the model
history <- model %>% fit(
  x = list(lstm_44_input = trainHot[samples, -1]),
  y = trainHot[samples, 1],
  batch_size = 8858,
  epochs = 10,
  validation_data = list(lstm_44_input = trainHot[samples, -1], trainHot[-samples, 1])
  #validation_split = .25
)

pred <- predict(model, testData)
