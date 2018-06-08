###################################################################
## Predictive Modeling Workshop, Open Data Science Conference
## Max Kuhn 2015-05-30

###################################################################
## Slide 17-18

library(Fahrmeir)
data(credit)

credit$Male <-ifelse(credit$Sexo == "hombre", 1, 0)
credit$Lives_Alone <-ifelse(credit$Estc == "vive solo", 1, 0)
credit$Good_Payer <-ifelse(credit$Ppag == "pre buen pagador", 1, 0)
credit$Private_Loan <-ifelse(credit$Uso == "privado", 1, 0)
credit$Class <-ifelse(credit$Y == "buen", "Good", "Bad")
credit$Class <- factor(credit$Class, levels = c("Good", "Bad"))

credit$Y <- NULL
credit$Sexo <- NULL
credit$Uso <- NULL
credit$Ppag <- NULL
credit$Estc <- NULL

names(credit)[names(credit) == "Mes"] <- "Loan_Duration"
names(credit)[names(credit) == "DM"] <- "Loan_Amount"
names(credit)[names(credit) == "Cuenta"] <- "Credit_Quality"

library(plyr)

## to make valid R column names
trans <- c("good running" = "good_running", "bad running" = "bad_running")
credit$Credit_Quality <- revalue(credit$Credit_Quality, trans)

str(credit)

###################################################################
## Slide 23

library(caret)
set.seed(8140)
in_train <- createDataPartition(credit$Class, p = .75, list = FALSE)
head(in_train)
train_data <- credit[ in_train,]
test_data  <- credit[-in_train,]
###################################################################
## Slide 51

alt_data <- model.matrix(Class ~ ., data = train_data)
alt_data[1:4, 1:3]

###################################################################
## Slide 52

dummy_info <- dummyVars(Class ~ ., data = train_data)
dummy_info
train_dummies <- predict(dummy_info, newdata = train_data)
train_dummies[1:4, 1:3]
test_dummies <- predict(dummy_info, newdata = test_data)

###################################################################
## Slide 55

pp_values <- preProcess(train_dummies, method = c("center", "scale"))
pp_values

train_scaled <- predict(pp_values, newdata = train_dummies)
test_scaled  <- predict(pp_values, newdata = test_dummies)

###################################################################
## Make example data

data(segmentationData)

segmentationData$Cell <- NULL
segmentationData$Class <- ifelse(segmentationData$Class == "PS", "One", "Two")
segmentationData <- segmentationData[, c("EqSphereAreaCh1", "PerimCh1", "Class", "Case")]
names(segmentationData)[1:2] <- paste0("Predictor", LETTERS[1:2])
example_train <- subset(segmentationData, Case == "Train")
example_test  <- subset(segmentationData, Case == "Test")

example_train$Case <- NULL
example_test$Case  <- NULL

###################################################################
## Slide 59

dim(example_train)
dim(example_test)
head(example_train)

###################################################################
## Slide 60

library(ggplot2)
theme_set(theme_bw())
ggplot(example_train, aes(x = PredictorA, 
                      y = PredictorB,
                      color = Class)) +
  geom_point(alpha = .5, cex = 2.6) + 
  theme(legend.position = "top")

###################################################################
## Slide 61

library(reshape2)
melted <- melt(example_train, id.vars = "Class")
ggplot(melted, aes(fill = Class, 
                   x = log(value), color = Class)) +
  geom_density(alpha = .2) + theme_bw() +
  facet_wrap(~variable, scales = "free_x") + 
  theme(legend.position = "top")

###################################################################
## Slide 62

pca_pp <- preProcess(example_train[, 1:2],
                     method = "pca") # also added "center" and "scale"
pca_pp
train_pc <- predict(pca_pp, example_train[, 1:2])
test_pc <- predict(pca_pp, example_test[, 1:2])
head(test_pc, 4)

###################################################################
## Slide 63

test_pc$Class <- example_test$Class
ggplot(test_pc, aes(x = PC1, 
                    y = PC2,
                    color = Class)) +
  geom_point(alpha = .5, cex = 2.6) + 
  theme(legend.position = "top")

###################################################################
## Slide 64

test_melt <- melt(test_pc, id.vars = "Class")

ggplot(test_melt, aes(fill = Class, 
                      x = value, color = Class)) +
  geom_density(alpha = .2) +  
  facet_wrap(~variable) + xlim(c(-2, 2)) +
  theme(legend.position = "top")

###################################################################
## Slide 66

pca_ss_pp <- preProcess(example_train[, 1:2],
                        method = c("pca", "spatialSign")) 
pca_ss_pp
train_pc_ss <- predict(pca_ss_pp, example_train[, 1:2])
test_pc_ss <- predict(pca_ss_pp, example_test[, 1:2])
head(test_pc_ss, 4)

###################################################################
## Slide 67

test_pc_ss <- as.data.frame(test_pc_ss)
test_pc_ss$Class <- example_test$Class
ggplot(test_pc_ss, aes(x = PC1,
                       y = PC2,
                       color = Class)) +
  geom_point(alpha = .5, cex = 2.6) + 
  theme(legend.position = "top")

###################################################################
## Slide 72

day_values <- c("2015-05-10", "1970-11-04", "2002-03-04", "2006-01-13")
class(day_values)

library(lubridate)
days <- ymd(day_values)
str(days)

###################################################################
## Slide 73

day_of_week <- wday(days, label = TRUE)
day_of_week

year(days)
week(days)
month(days, label = TRUE)
yday(days)

###################################################################
## Slide 76

library(rpart)
rpart1 <- rpart(Class ~ .,
                data = train_data,
                control = rpart.control(maxdepth = 2))
rpart1

###################################################################
## Slide 77

library(partykit)
rpart1_plot <- as.party(rpart1)
plot(rpart1_plot)

###################################################################
## Slide 80

rpart_full <- rpart(Class ~ ., data = train_data)

###################################################################
## Slide 81

rpart_full

###################################################################
## Slide 82

rpart_full_plot <- as.party(rpart_full)
plot(rpart_full_plot)

###################################################################
## Slide 83

rpart_pred <- predict(rpart_full, newdata = test_data, type = "class")
confusionMatrix(data = rpart_pred, reference = test_data$Class)   # requires 2 factor vectors

###################################################################
## Slide 84

class_probs <- predict(rpart_full, newdata = test_data)
head(class_probs, 3)       

library(pROC)
## The roc function assumes the *second* level is the one of
## interest, so we use the 'levels' argument to change the order.
rpart_roc <- roc(response = test_data$Class, predictor = class_probs[, "Good"], 
                 levels = rev(levels(test_data$Class)))
## Get the area under the ROC curve
auc(rpart_roc)

###################################################################
## Slide 90

cv_ctrl <- trainControl(method = "repeatedcv", repeats = 5,
                        summaryFunction = twoClassSummary,
                        classProbs = TRUE)
set.seed(1735)
rpart_tune <- train(Class ~ ., data = train_data, 
                    method = "rpart",
                    tuneLength = 9,
                    metric = "ROC",
                    trControl = cv_ctrl)


###################################################################
## Slide 92

## library(doMC)
## registerDoMC(cores = 2)

###################################################################
## Slide 96

rpart_tune

###################################################################
## Slide 98

ggplot(rpart_tune)

###################################################################
## Slide 100

rpart_pred2 <- predict(rpart_tune, newdata = test_data)
confusionMatrix(rpart_pred2, test_data$Class)

###################################################################
## Slide 101

rpart_probs <- predict(rpart_tune, newdata = test_data, type = "prob")
head(rpart_probs, n = 4)
rpart_roc <- roc(response = test_data$Class, predictor = rpart_probs[, "Good"], 
                 levels = rev(levels(test_data$Class)))
auc(rpart_roc)

###################################################################
## Slide 102

plot(rpart_roc, col = "#9E0142", legacy.axes = FALSE)

###################################################################
## Slide 109

library(gbm)
# The gbm function does not accept factor response values so
# will make a copy and modify the result variable
for_gbm <- train_data
for_gbm$Class <- ifelse(for_gbm$Class == "Good", 1, 0)

set.seed(10)
gbm_fit <- gbm(formula = Class ~  .,       # Try all predictors
               distribution = "bernoulli", # For classification
               data = for_gbm,
               n.trees = 100,              # 100 boosting iterations
               interaction.depth = 1,      # How many splits in each tree
               shrinkage = 0.1,            # learning rate
               verbose = FALSE)            # Do not print the details

###################################################################
## Slide 110

gbm_pred <- predict(gbm_fit, newdata = test_data, n.trees = 100,
                    ## This calculates the class probs
                    type = "response")
head(gbm_pred)
gbm_pred <- factor(ifelse(gbm_pred > .5, "Good", "Bad"),
                   levels = c("Good", "Bad"))
head(gbm_pred)

###################################################################
## Slide 111

confusionMatrix(gbm_pred, test_data$Class)

###################################################################
## Slide 112

gbm_grid <- expand.grid(interaction.depth = seq(1, 7, by = 2),
                        n.trees = seq(100, 1000, by = 50),
                        shrinkage = c(0.0001, 0.01),
                        n.minobsinnode = 10)

###################################################################
## Slide 113

average <- function(dat, ...) mean(dat, ...)
names(formals(mean.default))
average(dat = c(1:10, 100))
average(dat = c(1:10, 100), trim = .1)

###################################################################
## Slide 114

set.seed(1735)
gbm_tune <- train(Class ~ ., data = train_data,
                  method = "gbm",
                  metric = "ROC",    
                  # Use a custom grid of tuning parameters
                  tuneGrid = gbm_grid, 
                  trControl = cv_ctrl,
                  # Remember the 'three dots' discussed previously?
                  # This options is directly passed to the gbm function.
                  verbose = FALSE)  

gbm_tune

###################################################################
## Slide 116

ggplot(gbm_tune)

###################################################################
## Slide 117

gbm_pred <- predict(gbm_tune, newdata = test_data) # Magic!
confusionMatrix(gbm_pred, test_data$Class)

###################################################################
## Slide 118

gbm_probs <- predict(gbm_tune, newdata = test_data, type = "prob")
head(gbm_probs)

###################################################################
## Slide 119

gbm_roc <- roc(response = test_data$Class, predictor = gbm_probs[, "Good"],
               levels = rev(levels(test_data$Class)))
auc(rpart_roc)
auc(gbm_roc)

plot(rpart_roc, col = "#9E0142", legacy.axes = FALSE)
plot(gbm_roc,   col = "#3288BD", legacy.axes = FALSE, add = TRUE)
legend(.6, .5, legend = c("rpart", "gbm"), 
       lty = c(1, 1),
       col = c("#9E0142", "#3288BD"))


###################################################################
## Slide 132

set.seed(1735)
svm_tune <- train(Class ~ ., data = train_data,
                  method = "svmRadial",
                  # The default grid of cost parameters go from 2^-2,
                  # 0.5, 1, ...
                  # We'll fit 10 values in that sequence via the tuneLength
                  # argument.
                  tuneLength = 10,
                  preProc = c("center", "scale"),
                  metric = "ROC",
                  trControl = cv_ctrl)

svm_tune

###################################################################
## Slide 134

svm_tune$finalModel

###################################################################
## Slide 135

ggplot(svm_tune) + scale_x_log10()

###################################################################
## Slide 136

svm_pred <- predict(svm_tune, newdata = test_data)
confusionMatrix(svm_pred, test_data$Class)

###################################################################
## Slide 137

svm_probs <- predict(svm_tune, newdata = test_data, type = "prob")
head(svm_probs)

###################################################################
## Slide 138

svm_roc <- roc(response = test_data$Class, predictor = svm_probs[, "Good"],
               levels = rev(levels(test_data$Class)))
auc(rpart_roc)
auc(gbm_roc)
auc(svm_roc)

plot(rpart_roc, col = "#9E0142", legacy.axes = FALSE)
plot(gbm_roc, col = "#3288BD", add = TRUE, legacy.axes = FALSE)
plot(svm_roc, col = "#F46D43", add = TRUE, legacy.axes = FALSE)
legend(.6, .5, legend = c("rpart", "gbm", "svm"), 
       lty = c(1, 1, 1),
       col = c("#9E0142", "#3288BD", "#F46D43"))


