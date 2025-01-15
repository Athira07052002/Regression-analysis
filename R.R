# Load required libraries
library ( readr )
library ( ggplot2 )
library(ggcorrplot)
library ( caret )
library ( glmnet )
library (caTools)
# Read the dataset
mydata<-read.csv("C:/Users/91735/OneDrive/Documents/Datasets/Diabetes.csv")
mydata
### DATA PREPROCESSING
mydataframe<-data.frame( mydata )
mydataframe
## Data Cleaning
mydataframe<- na.omit(mydata)# Remove rows with missing values
mydataframe
#feature scaling
scaled_data<-scale(mydataframe)                            
scaled_features<-scale(mydata[,c("Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome")])
scaled_features<-data.frame(scaled_features)
#Create a new data frame with scaled features
mydataframe_scaled<-data.frame(mydataframe,scaled_features)
#Remove the original unscaled columns
mydataframe_scaled[,c("Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome")]<-NULL
##Exploratory Data Analysis
summary(mydataframe)
summary(mydataframe_scaled)
#Data Visualization
correlations<-cor(mydataframe_scaled[,c("Pregnancies.1","Glucose.1","BloodPressure.1","SkinThickness.1","Insulin.1","BMI.1","DiabetesPedigreeFunction.1","Age.1","Outcome.1")])
correlation_matrix<-cor(mydataframe_scaled)
cor(correlations)
#rounding to one decimal
cor=round(cor(correlations),1)
ggcorrplot(cor)#square tiles by default
#finding the values of correlation with Diabetes Pedigree Function and other variables
cor(mydataframe_scaled$DiabetesPedigreeFunction.1,mydataframe_scaled$Glucose.1)
cor(mydataframe_scaled$DiabetesPedigreeFunction.1,mydataframe_scaled$Insulin.1)
cor(mydataframe_scaled$DiabetesPedigreeFunction.1,mydataframe_scaled$BloodPressure.1)
#correlation plot of Diabetes Pedigree Function vs Glucose
correlation_coefficient<-cor(mydataframe_scaled$DiabetesPedigreeFunction.1,mydataframe_scaled$Glucose.1)
plot(mydataframe_scaled$DiabetesPedigreeFunction.1,mydataframe_scaled$Glucose.1,xlab = "DiabetesPedigreeFunction.1",ylab = "Glucose.1",main = paste("correlation plot(r=",round(correlation_coefficient,2),")"))
#correlation plot of Diabetes Pedigree Function vs Insulin
correlation_coefficient<-cor(mydataframe_scaled$DiabetesPedigreeFunction.1,mydataframe_scaled$Insulin.1)
plot(mydataframe_scaled$DiabetesPedigreeFunction.1,mydataframe_scaled$Insulin.1,xlab = "DiabetesPedigreeFunction.1",ylab = "Insulin.1",main = paste("correlation plot(r=",round(correlation_coefficient,2),")"))
#correlation plot of Diabetes Pedigree Function vs BloodPressure
correlation_coefficient<-cor(mydataframe_scaled$DiabetesPedigreeFunction.1,mydataframe_scaled$BloodPressure.1)
plot(mydataframe_scaled$DiabetesPedigreeFunction.1,mydataframe_scaled$BloodPressure.1,xlab = "DiabetesPedigreeFunction.1",ylab = "BloodPressure.1",main = paste("correlation plot(r=",round(correlation_coefficient,2),")"))
#Regression modeling
mydatareg<-mydataframe_scaled[,c("Pregnancies.1","Glucose.1","BloodPressure.1","SkinThickness.1","Insulin.1","BMI.1","DiabetesPedigreeFunction.1","Age.1","Outcome.1")]
# Split the data into training and testing sets
set.seed (123) # For reproducibility

# For linear regression
set.seed(123)
train_index_linear<-createDataPartition(mydatareg$DiabetesPedigreeFunction.1,p=0.7,list=FALSE)
train_data_linear<-mydatareg[train_index_linear, ]
test_data_linear<-mydatareg[-train_index_linear, ]

# For ridge and lasso regression
train_idx<-sample(1:nrow(mydatareg),0.7*nrow(mydatareg))# 70% for training
train_data_ridge_lasso<-mydatareg[train_idx, ]
test_data_ridge_lasso<-mydatareg[-train_idx, ]

# Prepare the predictor and response variables
x_train<-as.matrix(train_data_ridge_lasso[,c("Pregnancies.1","Glucose.1","BloodPressure.1","SkinThickness.1","Insulin.1","BMI.1","Age.1","Outcome.1")])
y_train<-train_data_ridge_lasso$DiabetesPedigreeFunction.1

#Prepare the predictor and response variables for the testing/validation set
x_test<-as.matrix(test_data_ridge_lasso[,c("Pregnancies.1","Glucose.1","BloodPressure.1","SkinThickness.1","Insulin.1","BMI.1","Age.1","Outcome.1")])
y_test<-test_data_ridge_lasso$DiabetesPedigreeFunction.1

## linear regression
# Build the regression model
linear_model<-lm(DiabetesPedigreeFunction.1 ~ . , data = train_data_linear)
# Make predictions on the test set
predictions<-predict(linear_model,newdata=test_data_linear)
## Ridge Regression
# Fit the ridge regression model
ridge_model<-cv.glmnet(x_train,y_train,alpha=0,lambda=seq(0.001,0.1,by=0.001)) 
# tune the lambda values as needed
# Find the optimal lambda value
best_lambda_ridge<-ridge_model$lambda.min
# Refit the ridge model using the optimal lambda
final_ridge_model<-glmnet(x_train,y_train,alpha=0,lambda=best_lambda_ridge)
# Predict using the ridge model
ridge_pred<-predict(final_ridge_model,newx=x_test)
## Lasso Regression
# Fit the lasso regression model
lasso_model<-cv.glmnet(x_train,y_train,alpha= 1,lambda=seq(0.001,0.1,by=0.001))
# tune the lambda values as needed
# Find the optimal lambda value
best_lambda_lasso<-lasso_model$lambda.min
# Refit the lasso model using the optimal lambda
final_lasso_model<-glmnet(x_train,y_train,alpha=1,lambda=best_lambda_lasso)
# Predict using the lasso model
lasso_pred<-predict(final_lasso_model,newx=x_test)
### MODEL EVALUATION
# Evaluate model performance
rmse<-caret::RMSE(predictions,test_data_linear$DiabetesPedigreeFunction.1 )
ridge_rmse<-sqrt(mean((y_test-ridge_pred)^2))
lasso_rmse<-sqrt(mean((y_test-lasso_pred)^2))
r_squared<-summary(linear_model)$r.squared
ridge_r_squared<-cor(y_test,ridge_pred)^2
lasso_r_squared<-cor(y_test,lasso_pred)^2
# Print model performance metrics
cat("linear Regression:\n")
cat("Root Mean Squared Error(RMSE):",rmse,"\n")
cat("R-squared:",r_squared,"\n")
cat("Ridge Regression:\n")
cat("Root Mean Squared Error(RMSE):",ridge_rmse,"\n")
cat("R-squared:",ridge_r_squared,"\n")
cat("Lasso Regression:\n")
cat("Root Mean Squared Error(RMSE):",lasso_rmse,"\n")
cat("R-squared:",lasso_r_squared,"\n")
#Regression modeling excluding Glucose and Insulin
# Split the data into training and testing sets
set.seed (123) # For reproducibility
# For linear regression
set.seed(123)
train_index_linear<-createDataPartition(mydatareg$DiabetesPedigreeFunction.1,p=0.7,list=FALSE)
train_data_linear<-mydatareg[train_index_linear, ]
test_data_linear<-mydatareg[-train_index_linear, ]
# For ridge and lasso regression without Glucose and Insulin
train_idx<-sample(1:nrow(mydatareg),0.7*nrow(mydatareg))# 70% for training
train_data_ridge_lasso<-mydatareg[train_idx, ]
test_data_ridge_lasso<-mydatareg[-train_idx, ]

# Prepare the predictor and response variables
x_train<-as.matrix(train_data_ridge_lasso[,c("Pregnancies.1","BloodPressure.1","SkinThickness.1","BMI.1","Age.1","Outcome.1")])
y_train<-train_data_ridge_lasso$DiabetesPedigreeFunction.1

#Prepare the predictor and response variables for the testing/validation set
x_test<-as.matrix(test_data_ridge_lasso[,c("Pregnancies.1","BloodPressure.1","SkinThickness.1","BMI.1","Age.1","Outcome.1")])
y_test<-test_data_ridge_lasso$DiabetesPedigreeFunction.1
## linear regression
# Build the regression model
linear_model<-lm(DiabetesPedigreeFunction.1 ~ . , data = train_data_linear)
# Make predictions on the test set
predictions<-predict(linear_model,newdata=test_data_linear)
## Ridge Regression
# Fit the ridge regression model
ridge_model<-cv.glmnet(x_train,y_train,alpha=0,lambda=seq(0.001,0.1,by=0.001)) 
# tune the lambda values as needed
# Find the optimal lambda value
best_lambda_ridge<-ridge_model$lambda.min
# Refit the ridge model using the optimal lambda
final_ridge_model<-glmnet(x_train,y_train,alpha=0,lambda=best_lambda_ridge)
# Predict using the ridge model
ridge_pred<-predict(final_ridge_model,newx=x_test)
## Lasso Regression
# Fit the lasso regression model
lasso_model<-cv.glmnet(x_train,y_train,alpha= 1,lambda=seq(0.001,0.1,by=0.001))
# tune the lambda values as needed
# Find the optimal lambda value
best_lambda_lasso<-lasso_model$lambda.min
# Refit the lasso model using the optimal lambda
final_lasso_model<-glmnet(x_train,y_train,alpha=1,lambda=best_lambda_lasso)
# Predict using the lasso model
lasso_pred<-predict(final_lasso_model,newx=x_test)
### MODEL EVALUATION
# Evaluate model performance
rmse<-caret::RMSE(predictions,test_data_linear$DiabetesPedigreeFunction.1 )
ridge_rmse<-sqrt(mean((y_test-ridge_pred)^2))
lasso_rmse<-sqrt(mean((y_test-lasso_pred)^2))
r_squared<-summary(linear_model)$r.squared
ridge_r_squared<-cor(y_test,ridge_pred)^2
lasso_r_squared<-cor(y_test,lasso_pred)^2
# Print model performance metrics
cat("linear Regression:\n")
cat("Root Mean Squared Error(RMSE):",rmse,"\n")
cat("R-squared:",r_squared,"\n")
cat("Ridge Regression:\n")
cat("Root Mean Squared Error(RMSE):",ridge_rmse,"\n")
cat("R-squared:",ridge_r_squared,"\n")
cat("Lasso Regression:\n")
cat("Root Mean Squared Error(RMSE):",lasso_rmse,"\n")
cat("R-squared:",lasso_r_squared,"\n")
#Regression modeling without Glucose,Insulin and BloodPressure
# Split the data into training and testing sets
set.seed (123) # For reproducibility
# For linear regression
set.seed(123)
train_index_linear<-createDataPartition(mydatareg$DiabetesPedigreeFunction.1,p=0.7,list=FALSE)
train_data_linear<-mydatareg[train_index_linear, ]
test_data_linear<-mydatareg[-train_index_linear, ]
# For ridge and lasso regression
train_idx<-sample(1:nrow(mydatareg),0.7*nrow(mydatareg))# 70% for training
train_data_ridge_lasso<-mydatareg[train_idx, ]
test_data_ridge_lasso<-mydatareg[-train_idx, ]
# Prepare the predictor and response variables without Glucose,Insulin and BloodPressure
x_train<-as.matrix(train_data_ridge_lasso[,c("Pregnancies.1","SkinThickness.1","BMI.1","Age.1","Outcome.1")])
y_train<-train_data_ridge_lasso$DiabetesPedigreeFunction.1
#Prepare the predictor and response variables for the testing/validation set
x_test<-as.matrix(test_data_ridge_lasso[,c("Pregnancies.1","SkinThickness.1","BMI.1","Age.1","Outcome.1")])
y_test<-test_data_ridge_lasso$DiabetesPedigreeFunction.1
## linear regression
# Build the regression model
linear_model<-lm(DiabetesPedigreeFunction.1 ~ . , data = train_data_linear)
# Make predictions on the test set
predictions<-predict(linear_model,newdata=test_data_linear)
## Ridge Regression
# Fit the ridge regression model
ridge_model<-cv.glmnet(x_train,y_train,alpha=0,lambda=seq(0.001,0.1,by=0.001)) 
# tune the lambda values as needed
# Find the optimal lambda value
best_lambda_ridge<-ridge_model$lambda.min
# Refit the ridge model using the optimal lambda
final_ridge_model<-glmnet(x_train,y_train,alpha=0,lambda=best_lambda_ridge)
# Predict using the ridge model
ridge_pred<-predict(final_ridge_model,newx=x_test)
## Lasso Regression
# Fit the lasso regression model
lasso_model<-cv.glmnet(x_train,y_train,alpha= 1,lambda=seq(0.001,0.1,by=0.001))
# tune the lambda values as needed
# Find the optimal lambda value
best_lambda_lasso<-lasso_model$lambda.min
# Refit the lasso model using the optimal lambda
final_lasso_model<-glmnet(x_train,y_train,alpha=1,lambda=best_lambda_lasso)
# Predict using the lasso model
lasso_pred<-predict(final_lasso_model,newx=x_test)
### MODEL EVALUATION
# Evaluate model performance
rmse<-caret::RMSE(predictions,test_data_linear$DiabetesPedigreeFunction.1 )
ridge_rmse<-sqrt(mean((y_test-ridge_pred)^2))
lasso_rmse<-sqrt(mean((y_test-lasso_pred)^2))
r_squared<-summary(linear_model)$r.squared
ridge_r_squared<-cor(y_test,ridge_pred)^2
lasso_r_squared<-cor(y_test,lasso_pred)^2
# Print model performance metrics
cat("linear Regression:\n")
cat("Root Mean Squared Error(RMSE):",rmse,"\n")
cat("R-squared:",r_squared,"\n")
cat("Ridge Regression:\n")
cat("Root Mean Squared Error(RMSE):",ridge_rmse,"\n")
cat("R-squared:",ridge_r_squared,"\n")
cat("Lasso Regression:\n")
cat("Root Mean Squared Error(RMSE):",lasso_rmse,"\n")
cat("R-squared:",lasso_r_squared,"\n")
y<-mydataframe_scaled$DiabetesPedigreeFunction.1
x1<-mydataframe_scaled$Pregnancies.1
x2<-mydataframe_scaled$Glucose.1
x3<-mydataframe_scaled$BloodPressure.1
x4<-mydataframe_scaled$SkinThickness.1
x5<-mydataframe_scaled$Insulin.1
x6<-mydataframe_scaled$BMI.1
x7<-mydataframe_scaled$Age.1
x8<-mydataframe_scaled$Outcome.1
model<-lm(y~x1+x2+x3+x4+x5+x6+x7+x8)
summary(model)
