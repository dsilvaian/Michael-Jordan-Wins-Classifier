## Final Project
## Ian D'Silva
## 4/5/2023

library(dplyr)
library(rsample)
library(RWeka)
library(e1071)
library(caret)
library(naivebayes)
library(pROC)


#Import data-set
#mjStats <- read.csv("./Original Data/michael-jordan-nba-career-regular-season-stats-by-game_Original.csv")
mjStats <- read.csv("michael-jordan-nba-career-regular-season-stats-by-game_Original.csv")

## Pre-processing Data
# Remove dependent and duplicate attributes
dupAttributes <- c("Rk", "G", "Years", "Days", "Diff", "FG", "X3P", "FT", "TRB", "Date", "EndYear", "Opp")
mjStats <- select(mjStats, -dupAttributes)


# Change attributes to categorical
mjStats$Win <- factor(mjStats$Win)
mjStats$Tm <- factor(mjStats$Tm)
mjStats$Home <- factor(mjStats$Home)
mjStats$GS <- factor(mjStats$GS)

# Check for missing values
any(missing(mjStats))
any(is.na(mjStats))

# After inspection, update the NA's to 0.0 since NA values are due to divide by
# zero error
mjStats[is.na(mjStats)] <- 0.0

# Setting seed to produce consistent split for data-set
seed <- 42
set.seed(seed)

# Splitting the data-set into train and test
dataset_split <- initial_split(mjStats, prop = 0.66, strata = Win)
train_mjStats <- training(dataset_split)
test_mjStats <- testing(dataset_split)

wins <- train_mjStats[train_mjStats$Win == 1,]

## Attribute Selection Methods
write.csv(train_mjStats,
          "./michael-jordan-nba-career-regular-season-stats-by-game_training.csv")
numAttributes <- 6

# CorrelationAttributeEval using Weka
corrAttr <- c("Home", "FG_PCT", "GmSc", "Tm", "DRB", "FGA", "Win")
corr_train_mjStats <- select(train_mjStats, all_of(corrAttr))
corr_test_mjStats <- select(test_mjStats, all_of(corrAttr))

# GainRatioAttributeEval using Weka
gainRatioAttr <- c("Home", "FG_PCT", "GS", "GmSc", "MP", "FGA", "Win")
gainRatio_train_mjStats <- select(train_mjStats, all_of(gainRatioAttr))
gainRatio_test_mjStats <- select(test_mjStats, all_of(gainRatioAttr))

# InfoGainAttributeEval using Weka
infoGainAttr <- c("Home", "FG_PCT", "GmSc", "MP", "Tm", "FGA", "Win")
infoGain_train_mjStats <- select(train_mjStats, all_of(infoGainAttr))
infoGain_test_mjStats <- select(test_mjStats, all_of(infoGainAttr))

# OneRAttributeEval using Weka
oneRAttr <- c("GS", "PF", "BLK", "ORB", "FGA", "Home", "FG_PCT", "Win")
oneR_train_mjStats <- select(train_mjStats, all_of(oneRAttr))
oneR_test_mjStats <- select(test_mjStats, all_of(oneRAttr))

# SymmetricalUncertAttributeEval using Weka
symmUncertAttr <- c("Home", "FG_PCT", "GmSc", "MP", "Tm", "FGA", "Win")
symmUncert_train_mjStats <- select(train_mjStats, all_of(symmUncertAttr))
symmUncert_test_mjStats <- select(test_mjStats, all_of(symmUncertAttr))


## Classifications
seed <- 42
set.seed(seed)

train_control <- trainControl(method = "cv", number = 10)

nbGrid <- expand.grid(usekernel = c(TRUE, FALSE),
                      laplace = c(0, 0.5, 1), 
                      adjust = c(0.75, 1, 1.25, 1.5))

J48Grid <- expand.grid(C = c(0.01, 0.25, 0.5), M = (1:4))

mtryValues <- seq(2, ncol(corr_train_mjStats)-1, by = 1)

knnGrid <-  expand.grid(k = (1:50))

# MCC function

mcc <- function(tp, tn, fp, fn) {
  num <- (tp * tn) - (fp * fn)
  den <- sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
  if (den == 0) return (0)
  return (num / den)
}

## Correlation Attribute Classifications
# Naive Bayes

seed <- 42
set.seed(seed)

corr_Model_NB <- train(Win ~ ., data = corr_train_mjStats, method = "naive_bayes",
                       trControl = train_control, tuneGrid = nbGrid)

corr_Model_NB
plot(corr_Model_NB)

corr_pred_NB <- predict(corr_Model_NB, newdata = corr_test_mjStats)

corr_results_NB <- confusionMatrix(corr_pred_NB, corr_test_mjStats$Win)
corr_results_NB

corr_accuracy_NB <- corr_results_NB$overall['Accuracy']
corr_accuracy_NB

corr_tp_NB <- corr_results_NB$table[4]
corr_tp_NB
corr_fp_NB <- corr_results_NB$table[3]
corr_fp_NB 
corr_fn_NB <- corr_results_NB$table[2]
corr_fn_NB
corr_tn_NB <- corr_results_NB$table[1]
corr_tn_NB

corr_precision_NB <- corr_tp_NB / (corr_tp_NB + corr_fp_NB)
corr_precision_NB

corr_recall_NB <- corr_tp_NB / (corr_tp_NB + corr_fn_NB)
corr_recall_NB

corr_Fscore_NB <- 2 * (corr_precision_NB * corr_recall_NB) / (corr_precision_NB + corr_recall_NB)
corr_Fscore_NB

mcc(corr_tp_NB, corr_tn_NB, corr_fp_NB, corr_fn_NB)

# J48 Decision Tree
seed <- 42
set.seed(seed)

corr_Model_J48 <- train(Win ~ ., data = corr_train_mjStats, method = "J48",
                       trControl = train_control, tuneGrid = J48Grid)

corr_Model_J48
plot(corr_Model_J48)

corr_pred_J48 <- predict(corr_Model_J48, newdata = corr_test_mjStats)

corr_results_J48 <- confusionMatrix(corr_pred_J48, corr_test_mjStats$Win)
corr_results_J48

corr_accuracy_J48 <- corr_results_J48$overall['Accuracy']
corr_accuracy_J48

corr_tp_J48 <- corr_results_J48$table[4]
corr_tp_J48
corr_fp_J48 <- corr_results_J48$table[3]
corr_fp_J48 
corr_fn_J48 <- corr_results_J48$table[2]
corr_fn_J48
corr_tn_J48 <- corr_results_J48$table[1]
corr_tn_J48

corr_precision_J48 <- corr_tp_J48 / (corr_tp_J48 + corr_fp_J48)
corr_precision_J48

corr_recall_J48 <- corr_tp_J48 / (corr_tp_J48 + corr_fn_J48)
corr_recall_J48

corr_Fscore_J48 <- 2 * (corr_precision_J48 * corr_recall_J48) / (corr_precision_J48 + corr_recall_J48)
corr_Fscore_J48

mcc(corr_tp_J48, corr_tn_J48, corr_fp_J48, corr_fn_J48)

# Random Forest
seed <- 42
set.seed(seed)

corr_Model_rf <- train(Win ~ ., data = corr_train_mjStats, method = "rf",
                 ntree = 500, tuneGrid = data.frame(mtry = mtryValues),
                 trControl = train_control)

corr_Model_rf
plot(corr_Model_rf)

corr_pred_rf <- predict(corr_Model_rf, newdata = corr_test_mjStats)

corr_results_rf <- confusionMatrix(corr_pred_rf, corr_test_mjStats$Win)
corr_results_rf

corr_accuracy_rf <- corr_results_rf$overall['Accuracy']
corr_accuracy_rf

corr_tp_rf <- corr_results_rf$table[4]
corr_tp_rf
corr_fp_rf <- corr_results_rf$table[3]
corr_fp_rf 
corr_fn_rf <- corr_results_rf$table[2]
corr_fn_rf
corr_tn_rf <- corr_results_rf$table[1]
corr_tn_rf

corr_precision_rf <- corr_tp_rf / (corr_tp_rf + corr_fp_rf)
corr_precision_rf

corr_recall_rf <- corr_tp_rf / (corr_tp_rf + corr_fn_rf)
corr_recall_rf

corr_Fscore_rf <- 2 * (corr_precision_rf * corr_recall_rf) / (corr_precision_rf + corr_recall_rf)
corr_Fscore_rf

mcc(corr_tp_rf, corr_tn_rf, corr_fp_rf, corr_fn_rf)

# Rpart
seed <- 42
set.seed(seed)

corr_Model_RPart <- train(Win ~ ., data = corr_train_mjStats, method = "rpart",
                        trControl = train_control, tuneLength = 20)

corr_Model_RPart
plot(corr_Model_RPart)

corr_pred_RPart <- predict(corr_Model_RPart, newdata = corr_test_mjStats)

corr_results_RPart <- confusionMatrix(corr_pred_RPart, corr_test_mjStats$Win)
corr_results_RPart

corr_accuracy_RPart <- corr_results_RPart$overall['Accuracy']
corr_accuracy_RPart

corr_tp_RPart <- corr_results_RPart$table[4]
corr_tp_RPart
corr_fp_RPart <- corr_results_RPart$table[3]
corr_fp_RPart 
corr_fn_RPart <- corr_results_RPart$table[2]
corr_fn_RPart
corr_tn_RPart <- corr_results_RPart$table[1]
corr_tn_RPart

corr_precision_RPart <- corr_tp_RPart / (corr_tp_RPart + corr_fp_RPart)
corr_precision_RPart

corr_recall_RPart <- corr_tp_RPart / (corr_tp_RPart + corr_fn_RPart)
corr_recall_RPart

corr_Fscore_RPart <- 2 * (corr_precision_RPart * corr_recall_RPart) / (corr_precision_RPart + corr_recall_RPart)
corr_Fscore_RPart

mcc(corr_tp_RPart, corr_tn_RPart, corr_fp_RPart, corr_fn_RPart)

# KNN
seed <- 42
set.seed(seed)

corr_Model_knn <- train(Win ~ ., data = corr_train_mjStats, method = "knn",
                          trControl = train_control, tuneGrid = knnGrid)

corr_Model_knn
plot(corr_Model_knn)

corr_pred_knn <- predict(corr_Model_knn, newdata = corr_test_mjStats)

corr_results_knn <- confusionMatrix(corr_pred_knn, corr_test_mjStats$Win)
corr_results_knn

corr_accuracy_knn <- corr_results_knn$overall['Accuracy']
corr_accuracy_knn

corr_tp_knn <- corr_results_knn$table[4]
corr_tp_knn
corr_fp_knn <- corr_results_knn$table[3]
corr_fp_knn 
corr_fn_knn <- corr_results_knn$table[2]
corr_fn_knn
corr_tn_knn <- corr_results_knn$table[1]
corr_tn_knn

corr_precision_knn <- corr_tp_knn / (corr_tp_knn + corr_fp_knn)
corr_precision_knn

corr_recall_knn <- corr_tp_knn / (corr_tp_knn + corr_fn_knn)
corr_recall_knn

corr_Fscore_knn <- 2 * (corr_precision_knn * corr_recall_knn) / (corr_precision_knn + corr_recall_knn)
corr_Fscore_knn

mcc(corr_tp_knn, corr_tn_knn, corr_fp_knn, corr_fn_knn)

## Gain Ratio Attribute Classifications
# Naive Bayes
seed <- 42
set.seed(seed)

gainRatio_Model_NB <- train(Win ~ ., data = gainRatio_train_mjStats,
                            method = "naive_bayes", trControl = train_control,
                            tuneGrid = nbGrid)

gainRatio_Model_NB
plot(gainRatio_Model_NB)

gainRatio_pred_NB <- predict(gainRatio_Model_NB, newdata = gainRatio_test_mjStats)

gainRatio_results_NB <- confusionMatrix(gainRatio_pred_NB, gainRatio_test_mjStats$Win)
gainRatio_results_NB

gainRatio_accuracy_NB <- gainRatio_results_NB$overall['Accuracy']
gainRatio_accuracy_NB

gainRatio_tp_NB <- gainRatio_results_NB$table[4]
gainRatio_tp_NB
gainRatio_fp_NB <- gainRatio_results_NB$table[3]
gainRatio_fp_NB 
gainRatio_fn_NB <- gainRatio_results_NB$table[2]
gainRatio_fn_NB
gainRatio_tn_NB <- gainRatio_results_NB$table[1]
gainRatio_tn_NB

gainRatio_precision_NB <- gainRatio_tp_NB / (gainRatio_tp_NB + gainRatio_fp_NB)
gainRatio_precision_NB

gainRatio_recall_NB <- gainRatio_tp_NB / (gainRatio_tp_NB + gainRatio_fn_NB)
gainRatio_recall_NB

gainRatio_Fscore_NB <- 2 * (gainRatio_precision_NB * gainRatio_recall_NB) / (gainRatio_precision_NB + gainRatio_recall_NB)
gainRatio_Fscore_NB

mcc(gainRatio_tp_NB, gainRatio_tn_NB, gainRatio_fp_NB, gainRatio_fn_NB)

# J48 Decision Tree
seed <- 42
set.seed(seed)

gainRatio_Model_J48 <- train(Win ~ ., data = gainRatio_train_mjStats,
                             method = "J48", trControl = train_control,
                             tuneGrid = J48Grid)

gainRatio_Model_J48
plot(gainRatio_Model_J48)

gainRatio_pred_J48 <- predict(gainRatio_Model_J48, newdata = gainRatio_test_mjStats)

gainRatio_results_J48 <- confusionMatrix(gainRatio_pred_J48, gainRatio_test_mjStats$Win)
gainRatio_results_J48

gainRatio_accuracy_J48 <- gainRatio_results_J48$overall['Accuracy']
gainRatio_accuracy_J48

gainRatio_tp_J48 <- gainRatio_results_J48$table[4]
gainRatio_tp_J48
gainRatio_fp_J48 <- gainRatio_results_J48$table[3]
gainRatio_fp_J48 
gainRatio_fn_J48 <- gainRatio_results_J48$table[2]
gainRatio_fn_J48
gainRatio_tn_J48 <- gainRatio_results_J48$table[1]
gainRatio_tn_J48

gainRatio_precision_J48 <- gainRatio_tp_J48 / (gainRatio_tp_J48 + gainRatio_fp_J48)
gainRatio_precision_J48

gainRatio_recall_J48 <- gainRatio_tp_J48 / (gainRatio_tp_J48 + gainRatio_fn_J48)
gainRatio_recall_J48

gainRatio_Fscore_J48 <- 2 * (gainRatio_precision_J48 * gainRatio_recall_J48) / (gainRatio_precision_J48 + gainRatio_recall_J48)
gainRatio_Fscore_J48

mcc(gainRatio_tp_J48, gainRatio_tn_J48, gainRatio_fp_J48, gainRatio_fn_J48)

# Random Forest
seed <- 42
set.seed(seed)

gainRatio_Model_rf <- train(Win ~ ., data = gainRatio_train_mjStats,
                            method = "rf", ntree = 500,
                            tuneGrid = data.frame(mtry = mtryValues),
                            trControl = train_control)

gainRatio_Model_rf
plot(gainRatio_Model_rf)

gainRatio_pred_rf <- predict(gainRatio_Model_rf, newdata = gainRatio_test_mjStats)

gainRatio_results_rf <- confusionMatrix(gainRatio_pred_rf, gainRatio_test_mjStats$Win)
gainRatio_results_rf

gainRatio_accuracy_rf <- gainRatio_results_rf$overall['Accuracy']
gainRatio_accuracy_rf

gainRatio_tp_rf <- gainRatio_results_rf$table[4]
gainRatio_tp_rf
gainRatio_fp_rf <- gainRatio_results_rf$table[3]
gainRatio_fp_rf 
gainRatio_fn_rf <- gainRatio_results_rf$table[2]
gainRatio_fn_rf
gainRatio_tn_rf <- gainRatio_results_rf$table[1]
gainRatio_tn_rf

gainRatio_precision_rf <- gainRatio_tp_rf / (gainRatio_tp_rf + gainRatio_fp_rf)
gainRatio_precision_rf

gainRatio_recall_rf <- gainRatio_tp_rf / (gainRatio_tp_rf + gainRatio_fn_rf)
gainRatio_recall_rf

gainRatio_Fscore_rf <- 2 * (gainRatio_precision_rf * gainRatio_recall_rf) / (gainRatio_precision_rf + gainRatio_recall_rf)
gainRatio_Fscore_rf

mcc(gainRatio_tp_rf, gainRatio_tn_rf, gainRatio_fp_rf, gainRatio_fn_rf)

# Rpart
seed <- 42
set.seed(seed)

gainRatio_Model_RPart <- train(Win ~ ., data = gainRatio_train_mjStats,
                               method = "rpart", trControl = train_control,
                               tuneLength = 20)

gainRatio_Model_RPart
plot(gainRatio_Model_RPart)

gainRatio_pred_RPart <- predict(gainRatio_Model_RPart, newdata = gainRatio_test_mjStats)

gainRatio_results_RPart <- confusionMatrix(gainRatio_pred_RPart, gainRatio_test_mjStats$Win)
gainRatio_results_RPart

gainRatio_accuracy_RPart <- gainRatio_results_RPart$overall['Accuracy']
gainRatio_accuracy_RPart

gainRatio_tp_RPart <- gainRatio_results_RPart$table[4]
gainRatio_tp_RPart
gainRatio_fp_RPart <- gainRatio_results_RPart$table[3]
gainRatio_fp_RPart 
gainRatio_fn_RPart <- gainRatio_results_RPart$table[2]
gainRatio_fn_RPart
gainRatio_tn_RPart <- gainRatio_results_RPart$table[1]
gainRatio_tn_RPart

gainRatio_precision_RPart <- gainRatio_tp_RPart / (gainRatio_tp_RPart + gainRatio_fp_RPart)
gainRatio_precision_RPart

gainRatio_recall_RPart <- gainRatio_tp_RPart / (gainRatio_tp_RPart + gainRatio_fn_RPart)
gainRatio_recall_RPart

gainRatio_Fscore_RPart <- 2 * (gainRatio_precision_RPart * gainRatio_recall_RPart) / (gainRatio_precision_RPart + gainRatio_recall_RPart)
gainRatio_Fscore_RPart

mcc(gainRatio_tp_RPart, gainRatio_tn_RPart, gainRatio_fp_RPart, gainRatio_fn_RPart)


# KNN
seed <- 42
set.seed(seed)

gainRatio_Model_knn <- train(Win ~ ., data = gainRatio_train_mjStats,
                             method = "knn", trControl = train_control,
                             tuneGrid = knnGrid)

gainRatio_Model_knn
plot(gainRatio_Model_knn)

gainRatio_pred_knn <- predict(gainRatio_Model_knn, newdata = gainRatio_test_mjStats)

gainRatio_results_knn <- confusionMatrix(gainRatio_pred_knn, gainRatio_test_mjStats$Win)
gainRatio_results_knn

gainRatio_accuracy_knn <- gainRatio_results_knn$overall['Accuracy']
gainRatio_accuracy_knn

gainRatio_tp_knn <- gainRatio_results_knn$table[4]
gainRatio_tp_knn
gainRatio_fp_knn <- gainRatio_results_knn$table[3]
gainRatio_fp_knn 
gainRatio_fn_knn <- gainRatio_results_knn$table[2]
gainRatio_fn_knn
gainRatio_tn_knn <- gainRatio_results_knn$table[1]
gainRatio_tn_knn

gainRatio_precision_knn <- gainRatio_tp_knn / (gainRatio_tp_knn + gainRatio_fp_knn)
gainRatio_precision_knn

gainRatio_recall_knn <- gainRatio_tp_knn / (gainRatio_tp_knn + gainRatio_fn_knn)
gainRatio_recall_knn

gainRatio_Fscore_knn <- 2 * (gainRatio_precision_knn * gainRatio_recall_knn) / (gainRatio_precision_knn + gainRatio_recall_knn)
gainRatio_Fscore_knn

mcc(gainRatio_tp_knn, gainRatio_tn_knn, gainRatio_fp_knn, gainRatio_fn_knn)


## Info Gain Attribute Classifications
# Naive Bayes
seed <- 42
set.seed(seed)

infoGain_Model_NB <- train(Win ~ ., data = infoGain_train_mjStats,
                            method = "naive_bayes", trControl = train_control,
                            tuneGrid = nbGrid)

infoGain_Model_NB
plot(infoGain_Model_NB)

infoGain_pred_NB <- predict(infoGain_Model_NB, newdata = infoGain_test_mjStats)

infoGain_results_NB <- confusionMatrix(infoGain_pred_NB, infoGain_test_mjStats$Win)
infoGain_results_NB

infoGain_accuracy_NB <- infoGain_results_NB$overall['Accuracy']
infoGain_accuracy_NB

infoGain_tp_NB <- infoGain_results_NB$table[4]
infoGain_tp_NB
infoGain_fp_NB <- infoGain_results_NB$table[3]
infoGain_fp_NB 
infoGain_fn_NB <- infoGain_results_NB$table[2]
infoGain_fn_NB
infoGain_tn_NB <- infoGain_results_NB$table[1]
infoGain_tn_NB

infoGain_precision_NB <- infoGain_tp_NB / (infoGain_tp_NB + infoGain_fp_NB)
infoGain_precision_NB

infoGain_recall_NB <- infoGain_tp_NB / (infoGain_tp_NB + infoGain_fn_NB)
infoGain_recall_NB

infoGain_Fscore_NB <- 2 * (infoGain_precision_NB * infoGain_recall_NB) / (infoGain_precision_NB + infoGain_recall_NB)
infoGain_Fscore_NB

mcc(infoGain_tp_NB, infoGain_tn_NB, infoGain_fp_NB, infoGain_fn_NB)

# J48 Decision Tree
seed <- 42
set.seed(seed)

infoGain_Model_J48 <- train(Win ~ ., data = infoGain_train_mjStats,
                             method = "J48", trControl = train_control,
                             tuneGrid = J48Grid)

infoGain_Model_J48
plot(infoGain_Model_J48)

infoGain_pred_J48 <- predict(infoGain_Model_J48, newdata = infoGain_test_mjStats)

infoGain_results_J48 <- confusionMatrix(infoGain_pred_J48, infoGain_test_mjStats$Win)
infoGain_results_J48

infoGain_accuracy_J48 <- infoGain_results_J48$overall['Accuracy']
infoGain_accuracy_J48

infoGain_tp_J48 <- infoGain_results_J48$table[4]
infoGain_tp_J48
infoGain_fp_J48 <- infoGain_results_J48$table[3]
infoGain_fp_J48 
infoGain_fn_J48 <- infoGain_results_J48$table[2]
infoGain_fn_J48
infoGain_tn_J48 <- infoGain_results_J48$table[1]
infoGain_tn_J48

infoGain_precision_J48 <- infoGain_tp_J48 / (infoGain_tp_J48 + infoGain_fp_J48)
infoGain_precision_J48

infoGain_recall_J48 <- infoGain_tp_J48 / (infoGain_tp_J48 + infoGain_fn_J48)
infoGain_recall_J48

infoGain_Fscore_J48 <- 2 * (infoGain_precision_J48 * infoGain_recall_J48) / (infoGain_precision_J48 + infoGain_recall_J48)
infoGain_Fscore_J48

mcc(infoGain_tp_J48, infoGain_tn_J48, infoGain_fp_J48, infoGain_fn_J48)

# Random Forest
seed <- 42
set.seed(seed)

infoGain_Model_rf <- train(Win ~ ., data = infoGain_train_mjStats,
                            method = "rf", ntree = 500,
                            tuneGrid = data.frame(mtry = mtryValues),
                            trControl = train_control)

infoGain_Model_rf
plot(infoGain_Model_rf)

infoGain_pred_rf <- predict(infoGain_Model_rf, newdata = infoGain_test_mjStats)

infoGain_results_rf <- confusionMatrix(infoGain_pred_rf, infoGain_test_mjStats$Win)
infoGain_results_rf

infoGain_accuracy_rf <- infoGain_results_rf$overall['Accuracy']
infoGain_accuracy_rf

infoGain_tp_rf <- infoGain_results_rf$table[4]
infoGain_tp_rf
infoGain_fp_rf <- infoGain_results_rf$table[3]
infoGain_fp_rf 
infoGain_fn_rf <- infoGain_results_rf$table[2]
infoGain_fn_rf
infoGain_tn_rf <- infoGain_results_rf$table[1]
infoGain_tn_rf

infoGain_precision_rf <- infoGain_tp_rf / (infoGain_tp_rf + infoGain_fp_rf)
infoGain_precision_rf

infoGain_recall_rf <- infoGain_tp_rf / (infoGain_tp_rf + infoGain_fn_rf)
infoGain_recall_rf

infoGain_Fscore_rf <- 2 * (infoGain_precision_rf * infoGain_recall_rf) / (infoGain_precision_rf + infoGain_recall_rf)
infoGain_Fscore_rf

mcc(infoGain_tp_rf, infoGain_tn_rf, infoGain_fp_rf, infoGain_fn_rf)

# Rpart
seed <- 42
set.seed(seed)

infoGain_Model_RPart <- train(Win ~ ., data = infoGain_train_mjStats,
                               method = "rpart", trControl = train_control,
                               tuneLength = 20)

infoGain_Model_RPart
plot(infoGain_Model_RPart)

infoGain_pred_RPart <- predict(infoGain_Model_RPart, newdata = infoGain_test_mjStats)

infoGain_results_RPart <- confusionMatrix(infoGain_pred_RPart, infoGain_test_mjStats$Win)
infoGain_results_RPart

infoGain_accuracy_RPart <- infoGain_results_RPart$overall['Accuracy']
infoGain_accuracy_RPart

infoGain_tp_RPart <- infoGain_results_RPart$table[4]
infoGain_tp_RPart
infoGain_fp_RPart <- infoGain_results_RPart$table[3]
infoGain_fp_RPart 
infoGain_fn_RPart <- infoGain_results_RPart$table[2]
infoGain_fn_RPart
infoGain_tn_RPart <- infoGain_results_RPart$table[1]
infoGain_tn_RPart

infoGain_precision_RPart <- infoGain_tp_RPart / (infoGain_tp_RPart + infoGain_fp_RPart)
infoGain_precision_RPart

infoGain_recall_RPart <- infoGain_tp_RPart / (infoGain_tp_RPart + infoGain_fn_RPart)
infoGain_recall_RPart

infoGain_Fscore_RPart <- 2 * (infoGain_precision_RPart * infoGain_recall_RPart) / (infoGain_precision_RPart + infoGain_recall_RPart)
infoGain_Fscore_RPart

mcc(infoGain_tp_RPart, infoGain_tn_RPart, infoGain_fp_RPart, infoGain_fn_RPart)


# KNN
seed <- 42
set.seed(seed)

infoGain_Model_knn <- train(Win ~ ., data = infoGain_train_mjStats,
                             method = "knn", trControl = train_control,
                             tuneGrid = knnGrid)

infoGain_Model_knn
plot(infoGain_Model_knn)

infoGain_pred_knn <- predict(infoGain_Model_knn, newdata = infoGain_test_mjStats)

infoGain_results_knn <- confusionMatrix(infoGain_pred_knn, infoGain_test_mjStats$Win)
infoGain_results_knn

infoGain_accuracy_knn <- infoGain_results_knn$overall['Accuracy']
infoGain_accuracy_knn

infoGain_tp_knn <- infoGain_results_knn$table[4]
infoGain_tp_knn
infoGain_fp_knn <- infoGain_results_knn$table[3]
infoGain_fp_knn 
infoGain_fn_knn <- infoGain_results_knn$table[2]
infoGain_fn_knn
infoGain_tn_knn <- infoGain_results_knn$table[1]
infoGain_tn_knn

infoGain_precision_knn <- infoGain_tp_knn / (infoGain_tp_knn + infoGain_fp_knn)
infoGain_precision_knn

infoGain_recall_knn <- infoGain_tp_knn / (infoGain_tp_knn + infoGain_fn_knn)
infoGain_recall_knn

infoGain_Fscore_knn <- 2 * (infoGain_precision_knn * infoGain_recall_knn) / (infoGain_precision_knn + infoGain_recall_knn)
infoGain_Fscore_knn

mcc(infoGain_tp_knn, infoGain_tn_knn, infoGain_fp_knn, infoGain_fn_knn)

## OneR Attribute Classifications
# Naive Bayes
seed <- 42
set.seed(seed)

oneR_Model_NB <- train(Win ~ ., data = oneR_train_mjStats,
                           method = "naive_bayes", trControl = train_control,
                           tuneGrid = nbGrid)

oneR_Model_NB
plot(oneR_Model_NB)

oneR_pred_NB <- predict(oneR_Model_NB, newdata = oneR_test_mjStats)

oneR_results_NB <- confusionMatrix(oneR_pred_NB, oneR_test_mjStats$Win)
oneR_results_NB

oneR_accuracy_NB <- oneR_results_NB$overall['Accuracy']
oneR_accuracy_NB

oneR_tp_NB <- oneR_results_NB$table[4]
oneR_tp_NB
oneR_fp_NB <- oneR_results_NB$table[3]
oneR_fp_NB 
oneR_fn_NB <- oneR_results_NB$table[2]
oneR_fn_NB
oneR_tn_NB <- oneR_results_NB$table[1]
oneR_tn_NB

oneR_precision_NB <- oneR_tp_NB / (oneR_tp_NB + oneR_fp_NB)
oneR_precision_NB

oneR_recall_NB <- oneR_tp_NB / (oneR_tp_NB + oneR_fn_NB)
oneR_recall_NB

oneR_Fscore_NB <- 2 * (oneR_precision_NB * oneR_recall_NB) / (oneR_precision_NB + oneR_recall_NB)
oneR_Fscore_NB

mcc(oneR_tp_NB, oneR_tn_NB, oneR_fp_NB, oneR_fn_NB)


# J48 Decision Tree
seed <- 42
set.seed(seed)

oneR_Model_J48 <- train(Win ~ ., data = oneR_train_mjStats,
                            method = "J48", trControl = train_control,
                            tuneGrid = J48Grid)

oneR_Model_J48
plot(oneR_Model_J48)

oneR_pred_J48 <- predict(oneR_Model_J48, newdata = oneR_test_mjStats)

oneR_results_J48 <- confusionMatrix(oneR_pred_J48, oneR_test_mjStats$Win)
oneR_results_J48

oneR_accuracy_J48 <- oneR_results_J48$overall['Accuracy']
oneR_accuracy_J48

oneR_tp_J48 <- oneR_results_J48$table[4]
oneR_tp_J48
oneR_fp_J48 <- oneR_results_J48$table[3]
oneR_fp_J48 
oneR_fn_J48 <- oneR_results_J48$table[2]
oneR_fn_J48
oneR_tn_J48 <- oneR_results_J48$table[1]
oneR_tn_J48

oneR_precision_J48 <- oneR_tp_J48 / (oneR_tp_J48 + oneR_fp_J48)
oneR_precision_J48

oneR_recall_J48 <- oneR_tp_J48 / (oneR_tp_J48 + oneR_fn_J48)
oneR_recall_J48

oneR_Fscore_J48 <- 2 * (oneR_precision_J48 * oneR_recall_J48) / (oneR_precision_J48 + oneR_recall_J48)
oneR_Fscore_J48

mcc(oneR_tp_J48, oneR_tn_J48, oneR_fp_J48, oneR_fn_J48)

# Random Forest
seed <- 42
set.seed(seed)

oneR_Model_rf <- train(Win ~ ., data = oneR_train_mjStats,
                           method = "rf", ntree = 500,
                           tuneGrid = data.frame(mtry = mtryValues),
                           trControl = train_control)

oneR_Model_rf
plot(oneR_Model_rf)

oneR_pred_rf <- predict(oneR_Model_rf, newdata = oneR_test_mjStats)

oneR_results_rf <- confusionMatrix(oneR_pred_rf, oneR_test_mjStats$Win)
oneR_results_rf

oneR_accuracy_rf <- oneR_results_rf$overall['Accuracy']
oneR_accuracy_rf

oneR_tp_rf <- oneR_results_rf$table[4]
oneR_tp_rf
oneR_fp_rf <- oneR_results_rf$table[3]
oneR_fp_rf 
oneR_fn_rf <- oneR_results_rf$table[2]
oneR_fn_rf
oneR_tn_rf <- oneR_results_rf$table[1]
oneR_tn_rf

oneR_precision_rf <- oneR_tp_rf / (oneR_tp_rf + oneR_fp_rf)
oneR_precision_rf

oneR_recall_rf <- oneR_tp_rf / (oneR_tp_rf + oneR_fn_rf)
oneR_recall_rf

oneR_Fscore_rf <- 2 * (oneR_precision_rf * oneR_recall_rf) / (oneR_precision_rf + oneR_recall_rf)
oneR_Fscore_rf

mcc(oneR_tp_rf, oneR_tn_rf, oneR_fp_rf, oneR_fn_rf)

# Rpart
seed <- 42
set.seed(seed)

oneR_Model_RPart <- train(Win ~ ., data = oneR_train_mjStats,
                              method = "rpart", trControl = train_control,
                              tuneLength = 20)

oneR_Model_RPart
plot(oneR_Model_RPart)

oneR_pred_RPart <- predict(oneR_Model_RPart, newdata = oneR_test_mjStats)

oneR_results_RPart <- confusionMatrix(oneR_pred_RPart, oneR_test_mjStats$Win)
oneR_results_RPart

oneR_accuracy_RPart <- oneR_results_RPart$overall['Accuracy']
oneR_accuracy_RPart

oneR_tp_RPart <- oneR_results_RPart$table[4]
oneR_tp_RPart
oneR_fp_RPart <- oneR_results_RPart$table[3]
oneR_fp_RPart 
oneR_fn_RPart <- oneR_results_RPart$table[2]
oneR_fn_RPart
oneR_tn_RPart <- oneR_results_RPart$table[1]
oneR_tn_RPart

oneR_precision_RPart <- oneR_tp_RPart / (oneR_tp_RPart + oneR_fp_RPart)
oneR_precision_RPart

oneR_recall_RPart <- oneR_tp_RPart / (oneR_tp_RPart + oneR_fn_RPart)
oneR_recall_RPart

oneR_Fscore_RPart <- 2 * (oneR_precision_RPart * oneR_recall_RPart) / (oneR_precision_RPart + oneR_recall_RPart)
oneR_Fscore_RPart

mcc(oneR_tp_RPart, oneR_tn_RPart, oneR_fp_RPart, oneR_fn_RPart)

# KNN
seed <- 42
set.seed(seed)

oneR_Model_knn <- train(Win ~ ., data = oneR_train_mjStats,
                            method = "knn", trControl = train_control,
                            tuneGrid = knnGrid)

oneR_Model_knn
plot(oneR_Model_knn)

oneR_pred_knn <- predict(oneR_Model_knn, newdata = oneR_test_mjStats)

oneR_results_knn <- confusionMatrix(oneR_pred_knn, oneR_test_mjStats$Win)
oneR_results_knn

oneR_accuracy_knn <- oneR_results_knn$overall['Accuracy']
oneR_accuracy_knn

oneR_tp_knn <- oneR_results_knn$table[4]
oneR_tp_knn
oneR_fp_knn <- oneR_results_knn$table[3]
oneR_fp_knn 
oneR_fn_knn <- oneR_results_knn$table[2]
oneR_fn_knn
oneR_tn_knn <- oneR_results_knn$table[1]
oneR_tn_knn

oneR_precision_knn <- oneR_tp_knn / (oneR_tp_knn + oneR_fp_knn)
oneR_precision_knn

oneR_recall_knn <- oneR_tp_knn / (oneR_tp_knn + oneR_fn_knn)
oneR_recall_knn

oneR_Fscore_knn <- 2 * (oneR_precision_knn * oneR_recall_knn) / (oneR_precision_knn + oneR_recall_knn)
oneR_Fscore_knn

mcc(oneR_tp_knn, oneR_tn_knn, oneR_fp_knn, oneR_fn_knn)

## Symmetric Uncertainty Attribute Classifications
# Naive Bayes
seed <- 42
set.seed(seed)

symmUncert_Model_NB <- train(Win ~ ., data = symmUncert_train_mjStats,
                       method = "naive_bayes", trControl = train_control,
                       tuneGrid = nbGrid)

symmUncert_Model_NB
plot(symmUncert_Model_NB)

symmUncert_pred_NB <- predict(symmUncert_Model_NB, newdata = symmUncert_test_mjStats)

symmUncert_results_NB <- confusionMatrix(symmUncert_pred_NB, symmUncert_test_mjStats$Win)
symmUncert_results_NB

symmUncert_accuracy_NB <- symmUncert_results_NB$overall['Accuracy']
symmUncert_accuracy_NB

symmUncert_tp_NB <- symmUncert_results_NB$table[4]
symmUncert_tp_NB
symmUncert_fp_NB <- symmUncert_results_NB$table[3]
symmUncert_fp_NB 
symmUncert_fn_NB <- symmUncert_results_NB$table[2]
symmUncert_fn_NB
symmUncert_tn_NB <- symmUncert_results_NB$table[1]
symmUncert_tn_NB

symmUncert_precision_NB <- symmUncert_tp_NB / (symmUncert_tp_NB + symmUncert_fp_NB)
symmUncert_precision_NB

symmUncert_recall_NB <- symmUncert_tp_NB / (symmUncert_tp_NB + symmUncert_fn_NB)
symmUncert_recall_NB

symmUncert_Fscore_NB <- 2 * (symmUncert_precision_NB * symmUncert_recall_NB) / (symmUncert_precision_NB + symmUncert_recall_NB)
symmUncert_Fscore_NB

mcc(symmUncert_tp_NB, symmUncert_tn_NB, symmUncert_fp_NB, symmUncert_fn_NB)

# J48 Decision Tree
seed <- 42
set.seed(seed)

symmUncert_Model_J48 <- train(Win ~ ., data = symmUncert_train_mjStats,
                        method = "J48", trControl = train_control,
                        tuneGrid = J48Grid)

symmUncert_Model_J48
plot(symmUncert_Model_J48)

symmUncert_pred_J48 <- predict(symmUncert_Model_J48, newdata = symmUncert_test_mjStats)

symmUncert_results_J48 <- confusionMatrix(symmUncert_pred_J48, symmUncert_test_mjStats$Win)
symmUncert_results_J48

symmUncert_accuracy_J48 <- symmUncert_results_J48$overall['Accuracy']
symmUncert_accuracy_J48

symmUncert_tp_J48 <- symmUncert_results_J48$table[4]
symmUncert_tp_J48
symmUncert_fp_J48 <- symmUncert_results_J48$table[3]
symmUncert_fp_J48 
symmUncert_fn_J48 <- symmUncert_results_J48$table[2]
symmUncert_fn_J48
symmUncert_tn_J48 <- symmUncert_results_J48$table[1]
symmUncert_tn_J48

symmUncert_precision_J48 <- symmUncert_tp_J48 / (symmUncert_tp_J48 + symmUncert_fp_J48)
symmUncert_precision_J48

symmUncert_recall_J48 <- symmUncert_tp_J48 / (symmUncert_tp_J48 + symmUncert_fn_J48)
symmUncert_recall_J48

symmUncert_Fscore_J48 <- 2 * (symmUncert_precision_J48 * symmUncert_recall_J48) / (symmUncert_precision_J48 + symmUncert_recall_J48)
symmUncert_Fscore_J48

mcc(symmUncert_tp_J48, symmUncert_tn_J48, symmUncert_fp_J48, symmUncert_fn_J48)

# Random Forest
seed <- 42
set.seed(seed)

symmUncert_Model_rf <- train(Win ~ ., data = symmUncert_train_mjStats,
                       method = "rf", ntree = 500,
                       tuneGrid = data.frame(mtry = mtryValues),
                       trControl = train_control)

symmUncert_Model_rf
plot(symmUncert_Model_rf)

symmUncert_pred_rf <- predict(symmUncert_Model_rf, newdata = symmUncert_test_mjStats)

symmUncert_results_rf <- confusionMatrix(symmUncert_pred_rf, symmUncert_test_mjStats$Win)
symmUncert_results_rf

symmUncert_accuracy_rf <- symmUncert_results_rf$overall['Accuracy']
symmUncert_accuracy_rf

symmUncert_tp_rf <- symmUncert_results_rf$table[4]
symmUncert_tp_rf
symmUncert_fp_rf <- symmUncert_results_rf$table[3]
symmUncert_fp_rf 
symmUncert_fn_rf <- symmUncert_results_rf$table[2]
symmUncert_fn_rf
symmUncert_tn_rf <- symmUncert_results_rf$table[1]
symmUncert_tn_rf

symmUncert_precision_rf <- symmUncert_tp_rf / (symmUncert_tp_rf + symmUncert_fp_rf)
symmUncert_precision_rf

symmUncert_recall_rf <- symmUncert_tp_rf / (symmUncert_tp_rf + symmUncert_fn_rf)
symmUncert_recall_rf

symmUncert_Fscore_rf <- 2 * (symmUncert_precision_rf * symmUncert_recall_rf) / (symmUncert_precision_rf + symmUncert_recall_rf)
symmUncert_Fscore_rf

mcc(symmUncert_tp_rf, symmUncert_tn_rf, symmUncert_fp_rf, symmUncert_fn_rf)


# Rpart
seed <- 42
set.seed(seed)

symmUncert_Model_RPart <- train(Win ~ ., data = symmUncert_train_mjStats,
                          method = "rpart", trControl = train_control,
                          tuneLength = 20)

symmUncert_Model_RPart
plot(symmUncert_Model_RPart)

symmUncert_pred_RPart <- predict(symmUncert_Model_RPart, newdata = symmUncert_test_mjStats)

symmUncert_results_RPart <- confusionMatrix(symmUncert_pred_RPart, symmUncert_test_mjStats$Win)
symmUncert_results_RPart

symmUncert_accuracy_RPart <- symmUncert_results_RPart$overall['Accuracy']
symmUncert_accuracy_RPart

symmUncert_tp_RPart <- symmUncert_results_RPart$table[4]
symmUncert_tp_RPart
symmUncert_fp_RPart <- symmUncert_results_RPart$table[3]
symmUncert_fp_RPart 
symmUncert_fn_RPart <- symmUncert_results_RPart$table[2]
symmUncert_fn_RPart
symmUncert_tn_RPart <- symmUncert_results_RPart$table[1]
symmUncert_tn_RPart

symmUncert_precision_RPart <- symmUncert_tp_RPart / (symmUncert_tp_RPart + symmUncert_fp_RPart)
symmUncert_precision_RPart

symmUncert_recall_RPart <- symmUncert_tp_RPart / (symmUncert_tp_RPart + symmUncert_fn_RPart)
symmUncert_recall_RPart

symmUncert_Fscore_RPart <- 2 * (symmUncert_precision_RPart * symmUncert_recall_RPart) / (symmUncert_precision_RPart + symmUncert_recall_RPart)
symmUncert_Fscore_RPart

mcc(symmUncert_tp_RPart, symmUncert_tn_RPart, symmUncert_fp_RPart, symmUncert_fn_RPart)

# KNN
seed <- 42
set.seed(seed)

symmUncert_Model_knn <- train(Win ~ ., data = symmUncert_train_mjStats,
                        method = "knn", trControl = train_control,
                        tuneGrid = knnGrid)

symmUncert_Model_knn
plot(symmUncert_Model_knn)

symmUncert_pred_knn <- predict(symmUncert_Model_knn, newdata = symmUncert_test_mjStats)

symmUncert_results_knn <- confusionMatrix(symmUncert_pred_knn, symmUncert_test_mjStats$Win)
symmUncert_results_knn

symmUncert_accuracy_knn <- symmUncert_results_knn$overall['Accuracy']
symmUncert_accuracy_knn

symmUncert_tp_knn <- symmUncert_results_knn$table[4]
symmUncert_tp_knn
symmUncert_fp_knn <- symmUncert_results_knn$table[3]
symmUncert_fp_knn 
symmUncert_fn_knn <- symmUncert_results_knn$table[2]
symmUncert_fn_knn
symmUncert_tn_knn <- symmUncert_results_knn$table[1]
symmUncert_tn_knn

symmUncert_precision_knn <- symmUncert_tp_knn / (symmUncert_tp_knn + symmUncert_fp_knn)
symmUncert_precision_knn

symmUncert_recall_knn <- symmUncert_tp_knn / (symmUncert_tp_knn + symmUncert_fn_knn)
symmUncert_recall_knn

symmUncert_Fscore_knn <- 2 * (symmUncert_precision_knn * symmUncert_recall_knn) / (symmUncert_precision_knn + symmUncert_recall_knn)
symmUncert_Fscore_knn

mcc(symmUncert_tp_knn, symmUncert_tn_knn, symmUncert_fp_knn, symmUncert_fn_knn)

#Classification models on whole training dataset

# Naive Bayes

seed <- 42
set.seed(seed)

Model_NB <- train(Win ~ ., data = train_mjStats, method = "naive_bayes",
                       trControl = train_control, tuneGrid = nbGrid)

Model_NB
plot(Model_NB)

pred_NB <- predict(Model_NB, newdata = test_mjStats)
confusionMatrix(pred_NB, test_mjStats$Win)


