# TASK:Help Santander Group identify the value of transactions for each potential customer. 
# OBJECTIVE:This is a first step that Santander needs to nail in order to personalize their services at scale.
# EVALUATION METRIC: Root Mean Squared Logarithmic Error

# Load the readr package
library(readr)

#Data Preparation
#Load training set
train <- read_csv("/Users/mellogwayo/Desktop/Santander Value Prediction Challenge/train.csv.zip") #The training set is used to train your machine learning model
test <- read_csv("/Users/mellogwayo/Desktop/Santander Value Prediction Challenge/test.csv.zip") #The testing set is used to evaluate its performance

# Display the number of rows and columns
dim(train) # 4459 rows and 4993 columns (8.2%)
dim(test)# 49342 rows and 4992 columns (91.8%)
         ### this data set is massive 

sapply(train, class)

## REFORMATING THE TRAINING DATA SET 
# Calculate the sum of each row, excluding the first two columns
total_transactions_train <- rowSums(train[, -(1:2)])
new_train <-train[, 1:2]
new_train  <- cbind(new_train, total_transactions_train)

## REFORMATING THE TESTING DATA SET 
# Calculate the sum of each row, excluding the first two columns
total_transactions_test <- rowSums(test[, -1])
ID <-data.frame(test[, 1])
new_test  <- cbind(ID, total_transactions_test)

 ## LINEAR REGRESSION MODEL
# Train a linear regression model
linear_model <- lm(target ~ total_transactions_train, data = new_train)

# Print a summary of the linear model
summary(linear_model)

# Make predictions on the test set
linear_model_predictions <- predict(linear_model, newdata = new_train)

# Calculate RMSLE for linear model
linear_model_rmsle <- sqrt(mean(log(linear_model_predictions + 1) - log(new_train$target + 1))^2)
print(paste("Root Mean Squared Logarithmic Error (RMSLE) for linear model:", linear_model_rmsle))


 ## RANDOM FOREST MODEL
# Load the randomForest package
library(randomForest)
# Train a random forest model
rf_model <- randomForest(target ~  total_transactions_train, data = new_train, ntree = 100)

# Make predictions on the test set
rf_predictions <- predict(rf_model, newdata = new_train)

# Calculate RMSLE for random forest model
rf_rmsle <- sqrt(mean(log(rf_predictions + 1) - log(new_train$target + 1))^2)
print(paste("Root Mean Squared Logarithmic Error (RMSLE) for random forest model:", rf_rmsle))

 ##SVM MODEL
# Load the e1071 package
library(e1071)
# Train an SVM model
SVM_model <- svm(target ~  total_transactions_train, data = new_train, kernel = "linear")

# Make predictions on the test set
SVM_predictions <- predict(SVM_model, newdata = new_train)

# Calculate RMSLE for SVM MODEL
SVM_rmsle <- sqrt(mean(log(SVM_predictions + 1) - log(new_train$target + 1))^2)
print(paste("Root Mean Squared Logarithmic Error (RMSLE) for SVM MODEL:", SVM_rmsle))



