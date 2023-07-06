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
library(imbalance)
library(tidyr)
library(usethis) 
library(pROC)
library(ROCR)
#usethis::edit_r_environ()
set.seed(1)
# Data
trainTrans <- read.csv('')
trainI <- read.csv('')
testI <- read.csv('')
testTrans <- read.csv('')

trainData <- merge(x = trainI, y = trainTrans, by = "TransactionID", all.y = T, all.x = T)
trainData <- trainData[which(colSums(is.na(trainData))<(nrow(trainData)*0.20))] #Put in report which variables were removed
trainData <- trainData[which(colSums(trainData=="")<(nrow(trainData)*0.20))] #Put in report which variables were removed
trainPlace <- trainData
trainHotDummies <- dummyVars("~ .", data = trainData[,c(5,7:8)])
trainHotL3dtest <- predict(trainHotDummies, newdata = trainData[,c(5,7:8)])
trainTest <- cbind(trainData[,c(1:4, 6)], trainHotL3dtest, trainData[, 9:22])
trainTest[, 22:34] <- apply(trainTest[, 22:34], 2, function(x) (x - min(x)) / (max(x) - min(x)))
trainTest <- trainTest[, -c(20)] # Takes out column with non existent test data column
trainHot <- cbind(trainData[,c(1:4, 6)], trainHotL3dtest, trainData[, 9:22])
trainHot[, 22:34] <- apply(trainHot[, 22:34], 2, function(x) (x - min(x)) / (max(x) - min(x)))
trainHot <- trainHot[, -c(20)] # Takes out column with non existent test data column
trainHotB <- trainHot[,-1]
trainHotB$isFraud <- as.factor(trainHotB$isFraud)
trainHotB <- SMOTE(isFraud ~ ., trainHotB, perc.over = 100, perc.under = 200)
trainHot <- trainHotB
smoteSelected <- sample(nrow(trainHot),size = round(nrow(trainHot)*0.75))
trainLstm <- trainHot[,c('isFraud', vars)] # Vars from boosting 
trainLstm <- trainLstm[,c(1,2,3)]
trainLstm <- cbind(trainLstm, trainData[,10:23])
trainLstm[, 4:17] <- apply(trainLstm[, 4:17], 2, function(x) (x - min(x)) / (max(x) - min(x)))

testHot <- cbind(testHot[,c(1,2)], testData[,7:20])
tH <- apply(testData[,7:20], 2, function(x) (x - min(x)) / (max(x) - min(x)))
test <- 't'
cols <- colnames(trainData)

testData <- merge(x = testI, y = testTrans, by = "TransactionID", all.y = T, all.x = T)
testData <- testData[,cols[-c(1,2)]]
testData <- testData[,-7]
dummiesTest <- dummyVars("~ .", data = testData[, c(3,5,6)])
test_dummies <- predict(dummiesTest, newdata = testData[, c(3,5,6)])
testHot <- cbind(testData[,c(1:2)], test_dummies, testData[, c(4,7:20)])
testHot[, 18:31] <- apply(testHot[, 18:31], 2, function(x) (x - min(x)) / (max(x) - min(x)))
testHot <- testHot[,c('TransactionDT',vars)]
selected = (((sample(nrow(trainHot),size = round(nrow(trainHot)*0.75)))))

# GBM with raw data
set.seed(1)
gbmFraud <- gbm(isFraud ~ ., data = juanEncoded2[selected,-1], distribution = 'gaussian', n.trees = 100, interaction.depth = 3, shrinkage = .1)
gbmFraud
predictions <- predict(gbmFraud, newdata = juanEncoded2[-selected,-1], n.trees = 50)
# convert probabilities to 0 or 1 predictions and calculate accuracy
predictions <- ifelse(predictions >= 0.03580072, 1, 0)
mean((predictions != juanEncoded2[-selected,-1]$isFraud))

# create confusion matrix
table(predictions, juanEncoded2[-selected,-1]$isFraud)
####################################################################################################################################

set.seed(5)
gbmTrain <- gbm((isFraud) ~ ., data = trainHot[smoteSelected,], distribution = 'gaussian', n.trees = 100, interaction.depth = 1, shrinkage = .1)
gbmTrain
yhatBoost <- predict(gbmTrain, newdata = trainTest[-smoteSelected,], n.trees = 100)
# Error
yhatBoost <- ifelse(yhatBoost >= 1.5, 1, 0)
mean((yhatBoost != trainTest[-selected,]$isFraud))
cmBoost <- table(predicted = yhatBoost, actual = trainTest[-selected,]$isFraud)
recallBoost <- cmBoost[2, 2] / sum(cmBoost[2, ])
precisionBoost <- cmBoost[2, 2] / sum(cmBoost[, 2])
F1Boost <- 2 * precisionBoost * recallBoost / (precisionBoost + recallBoost)
# Next Part
gbmTrain2 <- gbm(isFraud ~ ., data = trainHot[smoteSelected,-1], distribution = 'bernoulli', n.trees = 100, interaction.depth = 4, shrinkage = .1)
gbmTrain2
yhatBoost2 <- predict(gbmTrain2, newdata = trainTest[-smoteSelected,-1], n.trees = 100)
# Error
yhatBoost2 <- ifelse(yhatBoost2 >= 0.03580072, 1, 0)
mean((yhatBoost2 != juanEncoded2[-selected,-1]$isFraud))
table(yhatBoost2, juanEncoded2[-selected,-1]$isFraud)
# Next Next Part
gbmTrain3 <- gbm(isFraud ~ ., data = juanEncoded2[selected,-1], distribution = 'bernoulli', n.trees = 100, interaction.depth = 1, shrinkage = .01)
gbmTrain3
yhatBoost3 <- predict(gbmTrain3, newdata = juanEncoded2[-selected,-1], n.trees = 100)
# Error
yhatBoost3 <- ifelse(yhatBoost3 >= 0.03580072, 1, 0)
mean((yhatBoost3 != juanEncoded2[-selected,-1]$isFraud))
table(yhatBoost3, juanEncoded2[-selected,-1]$isFraud)
# Next Next Next Part
gbmTrain4 <- gbm(isFraud ~ ., data = juanEncoded2[selected,-1], distribution = 'bernoulli', n.trees = 100, interaction.depth = 4, shrinkage = .01)
gbmTrain4
yhatBoost4 <- predict(gbmTrain4, newdata = juanEncoded2[-selected,-1], n.trees = 100)
# Error
yhatBoost4 <- ifelse(yhatBoost4 >= 0.03580072, 1, 0)
mean((yhatBoost4 != juanEncoded2[-selected,-1]$isFraud))
table(yhatBoost4, juanEncoded2[-selected,-1]$isFraud)
# Model 2 has best parameters
nTrees <- seq(50, 1000, by = 50)
ctrl <- trainControl(method = "cv", number = 5)
nMinobsinnode <- 10
paramGrid <- expand.grid(n.trees = nTrees, interaction.depth = c(1,2,3,4,5,6,7), shrinkage = c(0.1, 0.01, 0.001), n.minobsinnode = c(3, 5, 7,10))

gbmCv <- caret::train(as.factor(isFraud) ~ ., data = trainHot[smoteSelected, ], method = 'gbm', distribution = "bernoulli", verbose = F,
               tuneGrid = paramGrid, trControl = ctrl)
plot(gbmCv)
bestTrees <- gbmCv$bestTune$n.trees
scaled <- abs(scale(juanEncoded2[,4:17]))
juanEncoded2[4:17] <- scaled
pca <- prcomp(trainHot[,-1], center = TRUE, scale. = TRUE)
zero <- which(juanEncoded2$isFraud == 0)
ones <- which(juanEncoded2$isFraud == 1)
zeroSpl <- sample(which(juanEncoded2$isFraud == 0),length(juanEncoded2$isFraud) / 2, replace = TRUE)
onesSpl <- sample(which(juanEncoded2$isFraud == 1),length(juanEncoded2$isFraud) / 2, replace = TRUE)

test_encoded <- dummyVars("~ .", data = testData[, -c(1,3)])
testDataDumb <- predict(test_encoded, newdata = testData)
testHot <- cbind(testData[, c(1,3)], testDataDumb)
smote <- trainHot
smote$isFraud <- as.factor(smote$isFraud)
trainDataBalanced <- SMOTE(isFraud ~ ., smote[,-1], perc.over = 150, perc.under = 0)
trainHotPlace <- trainHot
trainHot <- trainDataBalanced
selected = length(as.numeric(sample(nrow(trainHot),size = round(nrow(trainHot)*0.75))))
cols <- colnames(testHot)
isFraud <- trainHot$isFraud
trainHot <- cbind(isFraud, trainHot[, cols])
trainHot$isFraud <- as.numeric(trainHot$isFraud)
trainHot$isFraud <- ifelse(trainHot$isFraud == 1, 0, ifelse(trainHot$isFraud == 2, 1,trainHot$isFraud))

################################################################################################################################################################################################
trainHot$isFraud <- ifelse(trainHot$isFraud == 1, 0, 1)
gbmFinal <- gbm(isFraud ~ ., data = trainLstm[sel, ], n.trees = 500, interaction.depth = 4, shrinkage = .1, distribution = 'bernoulli')
logit <- glm(isFraud ~ ., data = trainLstm[sel, ], family = binomial())

logit
summary(logit)
sum <- summary(gbmFinal)
vars <- sum[1:10,1]
gbmFinal <- gbm(isFraud ~ ., data = trainHot[smoteSelected, c('isFraud', vars)], n.trees = 2000, interaction.depth = 4, shrinkage = .1, distribution = 'bernoulli')

testPredFold <- predict(gbmFinal, newdata = trainLstm[-sel,  ], n.trees = 2000, type = 'response')
lgrPred <- predict(logit, newdata = trainLstm[-sel, ], type = "response")
pred <- prediction(testPredFold, trainLstm[-sel,1])
predLgr <- prediction(lgrPred, trainLstm[-sel,1])

# calculate performance measures at different thresholds
perf <- performance(pred, "tpr", "fpr")
thresholds <- perf@x.values[[1]]
tpr <- perf@y.values[[1]]
fpr <- perf@x.values[[1]]

# calculate the Youden's index for each threshold
youden <- tpr + (1 - fpr) - 1

# find the threshold that maximizes Youden's index
optimal_threshold <- thresholds[which.max(youden)]
perfLgr <- performance(predLgr, "tpr", "fpr")
thresholdsLgr <- perfLgr@x.values[[1]]
tprLgr <- perfLgr@y.values[[1]]
fprLgr <- perfLgr@x.values[[1]]

# calculate the Youden's index for each threshold
youdenLgr <- tprLgr + (1 - fprLgr) - 1

# find the threshold that maximizes Youden's index
optimal_threshold_lgr <- thresholdsLgr[which.max(youdenLgr)]

logitPredsBinary <- ifelse(as.numeric(lgrPred) >= optimal_threshold_lgr, 1, 0)
mean((as.factor(logitPredsBinary) != (trainHot[-selected,1])))
cmLgr <- table(actual = as.factor(trainHot[-selected,1]),  predicted = as.factor(logitPredsBinary))
cmLgr
recallLgr <- cmLgr[2, 2] / sum(cmLgr[2, ])
precisionLgr <- cmLgr[2, 2] / sum(cmLgr[, 2])

testPredFold <- ifelse(testPredFold >= (optimal_threshold), 1, 0)
mean((as.factor(testPredFold) != (trainLstm[-sel,1])))
cmGbm <- table(actual = as.factor(trainLstm[-sel,1]), predicted = as.factor(testPredFold))
cmGbm
recallGbm <- cmGbm[2, 2] / sum(cmGbm[2, ])
precisionGbm <- cmGbm[2, 2] / sum(cmGbm[, 2])
F1Gbm <- 2 * precisionGbm * recallGbm / (precisionGbm + recallGbm)

combPreds <- rep(0, length(testPredFold))
combPreds <- .3 * logitPredsBinary + 7 * testPredFold
#combPreds <- ifelse(combPreds >= .5, 1, sample(c(0, 1), prob = c(.5,.5), replace = T, size = 1))
for(i in 1:length(combPreds)){
  if(combPreds[i] > .5){
    combPreds[i] <- as.numeric(1)
  }
  else{
    combPreds[i] <- as.numeric(sample(c(0, 1), prob = c(.5,.5), replace = T, size = 1))
  }
}
#binary_vector <- sample(c(1), sum(is.na(combPreds)), replace = TRUE)
#combPreds[is.na(combPreds)] <- binary_vector
mean(as.factor(combPreds) != (trainHot[-selected,1]))
cm <- table(actual = as.factor(trainHot[-selected,1]), predicted = as.factor(combPreds))
cm
recall <- cm[2, 2] / sum(cm[2, ])
precision <- cm[2, 2] / sum(cm[, 2])
F1 <- 2 * precision * recall / (precision + recall)


# Final predictions
finalPredictions <- predict(gbmFinal, newdata = testHot, n.trees = 500, type = 'response')
lgrPredFinal <- predict(logit, newdata = testHot, type = "response")
logitPredsFinal<- ifelse(as.numeric(lgrPredFinal) >= optimal_threshold_lgr, 1, 0)
binary_vector <- sample(c(0,1), sum(is.na(lgrPredFinal)), replace = TRUE, prob = c(.1,.9))
lgrPredFinal[is.na(lgrPredFinal)] <- binary_vectorF
finalPredictions <- ifelse(finalPredictions >= (optimal_threshold + 1.2), 1, 0)
combPredsFinal <- .3 * logitPredsFinal + .7 * finalPredictions
binary_vectorF <- sample(c(0,1), sum(is.na(as.factor(combPredsFinal))), replace = TRUE, prob = c(.1,.9), size = 1)
combPredsFinal[is.na(combPredsFinal)] <- binary_vectorF
for(i in 1:length(combPredsFinal)){
  if(combPredsFinal[i] >= .7){
    combPredsFinal[i] <- as.numeric(1)
  }
  else{
    combPredsFinal[i] <- as.numeric(sample(c(0, 1), prob = c(.7,.3), replace = T, size = 1))
  }
}
finDf <- data.frame(TransactionID = testTrans[,1], isFraud = as.factor(finalPredictions))
finalPreds <- write.csv(finDf, file = 'submission66.csv', row.names = F)

###############################################################################################################################################

# build gbm model using one hot encoding and make predictions
train <- cbind(train[1], model.matrix(~ `card4` + `card6` + `ProductCD` - 1, data = train))
train <- janitor::clean_names(train)
test <- cbind(test[1], model.matrix(~ `card4` + `card6` + `ProductCD` - 1, data = test))
test <- janitor::clean_names(test)
x <- intersect(names(train), names(test))
train <- subset(train, select = c(x))
names(test) <- names(train)

set.seed(1)
gbmFraud<- gbm(is_fraud ~ ., data = train, distribution = 'bernoulli', n.trees = 100, interaction.depth = 4, shrinkage = .1)
gbmFraud
predictions <- predict(gbmFraud, newdata = test, n.trees = 50)

# convert probabilities to 0 or 1 predictions and calculate accuracy
predictions <- ifelse(predictions >= 0.8185358, 1, 0)
mean((predictions != test$is_fraud))

# create confusion matrix
table(predictions, test$is_fraud)

