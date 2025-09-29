library(tidyverse)
library(xgboost)
library(glmtrans)
library(glmnet)
library(caret)
library(pbapply)
library(parallel)
library(MASS)
library(pROC)

source("TransLasso-functions.R")

# Builds ML algorithm (supposed to be given to us)


build_ML <- function(data_unknown, task = "auto") {
  # Splitting unknown data into train(80%) and validation(20%)
  set.seed(123)  # Set seed for reproducibility
  
  N_u <- nrow(data_unknown)
  N_train <- round(0.8 * N_u)  # 80% for training
  
  # Generate random indices
  train_indices <- sample(1:N_u, N_train, replace = FALSE)
  
  # Create training and validation sets
  data_unknown_train <- data_unknown[train_indices, ] %>% as.matrix()
  data_unknown_valid <- data_unknown[-train_indices, ] %>% as.matrix()
  
  # Determine task type (classification or regression)
  y_train <- data_unknown_train[, 1]  # First column is the target
  if (task == "auto") {
    task <- ifelse(length(unique(y_train)) == 2, "classification", "regression")
  }
  
  # Convert data to XGBoost format
  ML_data_train <- xgb.DMatrix(data = data_unknown_train[, -1] %>%  as.matrix() , label = y_train)
  ML_data_valid <- xgb.DMatrix(data = data_unknown_valid[, -1] %>% as.matrix(), label = data_unknown_valid[, 1])
  
  # Set model parameters
  if (task == "classification") {
    objective <- "binary:logistic"
    eval_metric <- "logloss"
  } else {  # Regression
    objective <- "reg:squarederror"
    eval_metric <- "rmse"
  }
  
  # Train the XGBoost model
  ML <- xgb.train(
    params = list(objective = objective, eval_metric = eval_metric,
                  max_depth = 4, max_leaves = 5000, 
                  subsample = 0.8, colsample_bytree = 0.7, eta = 0.1),
    data = ML_data_train,
    watchlist = list(train = ML_data_train, test = ML_data_valid), 
    nrounds = 5000,
    early_stopping_rounds = 50,
    verbose = 0  # Set to 1 to print the progress
  )
  
  # Predict on training and validation data
  train_pred <- predict(ML, ML_data_train)
  valid_pred <- predict(ML, ML_data_valid)
  
  # Compute evaluation metrics for training data
  if (task == "classification") {
    train_logloss <- mean(log(1 + exp(-train_pred * data_unknown_train[, 1])))
    valid_logloss <- mean(log(1 + exp(-valid_pred * data_unknown_valid[, 1])))
    cat("Train Log-Loss:", train_logloss, "\n")
    cat("Validation Log-Loss:", valid_logloss, "\n")
  } else {  # Regression
    train_rmse <- sqrt(mean((train_pred - data_unknown_train[, 1])^2))
    valid_rmse <- sqrt(mean((valid_pred - data_unknown_valid[, 1])^2))
    cat("Train RMSE:", train_rmse, "\n")
    cat("Validation RMSE:", valid_rmse, "\n")
  }
  
  return(list("ML_model" = ML, "task" = task))
}


# Gives estimates for the variances by doing a PCA on X, then lm on reduced X
sigma_hat_estimate <- function(y, X, num_components = NULL) {
  pca <- prcomp(X, center = TRUE, scale. = F)
  if (is.null(num_components)) {
    num_components <- which(cumsum(pca$sdev^2) / sum(pca$sdev^2) >= 0.95)[1]
  }
  #print(num_components)
  X_reduced <- pca$x[, 1:num_components]
  residuals <- lm(y ~ X_reduced)$residuals
  sum(residuals^2) / (length(y) - num_components - 1)
}


##### Regression functions ######

# Generates the estimate for beta using PPI method
mu_generate_ppi = function(X_tilde, X,y,Y_tilde_ML,ML){
  fx = as.matrix(predict(ML, newdata = X))
  mu_ppi = ginv(X_tilde)%*%Y_tilde_ML - ginv(X)%*%(fx - y)
  return(mu_ppi)
}

# Generates the estimate for beta using TransLasso method
mu_generate_TransLasso = function(X_tilde, X,y, Y_tilde_ML,size.A0,l1 = T){
  n.vec = c(nrow(X),nrow(X_tilde))
  X_TL = rbind(X,X_tilde)
  y_TL = rbind(y,Y_tilde_ML)
  beta.init <-
    as.numeric(glmnet(X[1:n.vec[1], ], y[1:n.vec[1]], lambda = sqrt(2 * log(ncol(X)) / n.vec[1]))$beta)
  if (size.A0 == 0) {
    beta.kA <- beta.init
  } else{
    beta.kA <- las.kA(X_TL, y_TL, A0 = 1:size.A0, n.vec = n.vec, l1=l1)$beta.kA
  }
  return(beta.kA)
}

# Generates the estimate for beta using GLMTrans method, change family for specific task
mu_generate_GLMTrans = function(X_tilde, X,y, Y_tilde_ML, family = 'gaussian'){
  GLMTrans_mod = glmtrans(target = list(x = X,y = y),
                          source = list(list(x = X_tilde,y = Y_tilde_ML)),
                          intercept = FALSE, family = family)
  beta_GLMTrans = GLMTrans_mod$beta[-1]
  return(beta_GLMTrans)
}

# Generates the estimate for beta using ridge  on only target data, change family for specific task
mu_generate_ridge = function(X,y, lambda_values = seq(0.000001, 10, length = 100), family = 'gaussian'){
  cv_ridge <- cv.glmnet(as.matrix(X), as.matrix(y), alpha = 0, 
                        intercept = FALSE, lambda = lambda_values, family = family)
  best_lambda <- cv_ridge$lambda.min
  
  # Fit the Ridge model with the best lambda
  ridge_reg <- glmnet(as.matrix(X), as.matrix(y), alpha = 0, 
                      lambda = best_lambda, intercept = FALSE, family = family)
  # Extract coefficients
  beta_ridge <- as.vector(coef(ridge_reg)[-1])  # Remove intercept term if present
  return(beta_ridge)
}

# Generates the estimate for beta using ordinary least squares regression on only target data
mu_generate_ols = function(X,y){
  beta_ols = lm(y~X-1)$coefficients
  return(beta_ols)
}

check_errors <- function(data_train, data_test, beta_hats, beta) {
  X_test <- as.matrix(data_test[,-1])
  y_test <- as.matrix(data_test[,1])
  
  # Check if n <= p for data_test and conditionally skip error calculations
  n <- nrow(data_train)
  p <- ncol(data_train) - 1  # Exclude the target variable column
  
  # Predictions for different methods
  y_blast <- X_test %*% beta_hats$mu_TL
  y_ppi <- X_test %*% beta_hats$mu_ppi 
  y_GLMTrans <- X_test %*% beta_hats$mu_GLMTrans
  y_TransLasso <- X_test %*% beta_hats$mu_TransLasso
  y_TransLasso0 <- X_test %*% beta_hats$mu_TransLasso0
  y_ridge <- X_test %*% beta_hats$ridge  
  y_ols <- if (n> p) X_test %*% beta_hats$ols else NULL
  
  # Test errors
  pred_err_blast <- sqrt(mean((y_blast - y_test)^2))
  pred_err_ppi <- sqrt(mean((y_ppi - y_test)^2)) 
  pred_err_GLMTrans <- sqrt(mean((y_GLMTrans - y_test)^2))
  pred_err_TransLasso1 <- sqrt(mean((y_TransLasso - y_test)^2))
  pred_err_TransLasso0 <- sqrt(mean((y_TransLasso0 - y_test)^2))
  pred_err_TransLasso <- min(pred_err_TransLasso1, pred_err_TransLasso0)
  best_transLasso <- which.min(c(pred_err_TransLasso1, pred_err_TransLasso0)) 
  pred_err_ridge <- sqrt(mean((y_ridge - y_test)^2)) 
  pred_err_ols <- if (!is.null(y_ols)) sqrt(mean((y_ols - y_test)^2)) else NA
  
  # Train data
  X_train <- as.matrix(data_train[,-1])
  y_train <- as.matrix(data_train[,1])
  
  # Train predictions for different methods
  y_blast_train <- X_train %*% beta_hats$mu_TL
  y_ppi_train <-  X_train %*% beta_hats$mu_ppi 
  y_GLMTrans_train <- X_train %*% beta_hats$mu_GLMTrans
  if (best_transLasso == 1) y_TransLasso_train <- X_train %*% beta_hats$mu_TransLasso
  if (best_transLasso == 2) y_TransLasso_train <- X_train %*% beta_hats$mu_TransLasso0
  y_ridge_train <- X_train %*% beta_hats$ridge 
  y_ols_train <- if (n> p) X_train %*% beta_hats$ols else NULL
  
  # Train errors
  pred_err_blast_train <- sqrt(mean((y_blast_train - y_train)^2))
  pred_err_ppi_train <-sqrt(mean((y_ppi_train - y_train)^2))
  pred_err_GLMTrans_train <- sqrt(mean((y_GLMTrans_train - y_train)^2))
  pred_err_TransLasso_train <- sqrt(mean((y_TransLasso_train - y_train)^2))
  pred_err_ridge_train <- sqrt(mean((y_ridge_train - y_train)^2))  
  pred_err_ols_train <- if (!is.null(y_ols_train)) sqrt(mean((y_ols_train - y_train)^2))  else NA
  
  # Beta errors
  error_ppi <- sqrt(mean((beta - beta_hats$mu_ppi)^2)) 
  err_GLMTrans <- sqrt(mean((beta - beta_hats$mu_GLMTrans)^2))
  if (best_transLasso == 1) err_TransLasso_train <- sqrt(mean((beta - beta_hats$mu_TransLasso)^2))
  if (best_transLasso == 2) err_TransLasso_train <- sqrt(mean((beta - beta_hats$mu_TransLasso0)^2))
  err_ours <- sqrt(mean((beta - beta_hats$mu_TL)^2))
  err_ridge <- sqrt(mean((beta - beta_hats$ridge)^2))  
  err_ols <- if (n> p) sqrt(mean((beta - beta_hats$ols)^2)) else NA 
  
  # Return results as a data frame
  res <- data.frame(
    "Method" = c("BLAST", "PPI", "GLMTrans", "TransLasso", "Ridge", "OLS"),
    "Train_error" = round(c(pred_err_blast_train, 
                            pred_err_ppi_train, 
                            pred_err_GLMTrans_train, 
                            pred_err_TransLasso_train,
                            pred_err_ridge_train,
                            pred_err_ols_train), 4),
    "Test_error" = round(c(pred_err_blast, 
                           pred_err_ppi, 
                           pred_err_GLMTrans, 
                           pred_err_TransLasso,
                           pred_err_ridge,
                           pred_err_ols), 4),
    "beta_error" = round(c(err_ours, 
                           error_ppi, 
                           err_GLMTrans, 
                           err_TransLasso_train,
                           err_ridge,
                           err_ols), 4)
  )
  
  return(res)
}

# Also adding test errors for a ML model trained right on the small labeled data 


##### Classification functions ######

mu_generate_logistic = function(X,y){
  beta_logistic = glm(y ~ X-1, data = data.frame(y = y, X = X), family = binomial())$coeff
  return(beta_logistic)
}

check_errors_classification <- function(data_train, data_test, beta_hats, thresh = "auto") {
  library(pROC)
  
  # Helper function to compute MCC
  compute_mcc <- function(y_true, y_pred) {
    TP <- sum(y_true == 1 & y_pred == 1)/100
    TN <- sum(y_true == 0 & y_pred == 0)/100
    FP <- sum(y_true == 0 & y_pred == 1)/100
    FN <- sum(y_true == 1 & y_pred == 0)/100
    
    numerator <- (TP * TN) - (FP * FN)
    
    denominator <- sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    
    if (denominator == 0) return(NA)  # Avoid division by zero
    return(numerator / denominator)
  }
  
  # Prepare train and test data
  X_test <- as.matrix(data_test[, -1])
  y_test <- as.matrix(data_test[, 1])
  
  X_train <- as.matrix(data_train[, -1])
  y_train <- as.matrix(data_train[, 1])
  
  # Model Predictions (Train)
  y_blastvar_train <- X_train %*% beta_hats$mu_TL_var %>% plogis()
  y_GLMTrans_train <- X_train %*% beta_hats$mu_GLMTrans %>% plogis()
  y_ridge_train <- X_train %*% beta_hats$ridge %>% plogis()
  y_logistic_train <- if (nrow(X_train) > ncol(X_train)) X_train %*% beta_hats$logistic %>% plogis() else NULL
  
  # ROC Curves
  roc_blastvar_train <- roc(y_train, y_blastvar_train, plotit = FALSE)
  roc_GLMTrans_train <- roc(y_train, y_GLMTrans_train, plotit = FALSE)
  roc_ridge_train <- roc(y_train, y_ridge_train, plotit = FALSE)
  roc_logistic_train <- if (!is.null(y_logistic_train)) roc(y_train, y_logistic_train, plotit = FALSE) else NULL
  
  # Threshold Selection
  if (thresh == "auto") {
    best_threshold_blastvar <- roc_blastvar_train$thresholds[which.max(roc_blastvar_train$sensitivities - (1 - roc_blastvar_train$specificities))]
    best_threshold_GLMTrans <- roc_GLMTrans_train$thresholds[which.max(roc_GLMTrans_train$sensitivities - (1 - roc_GLMTrans_train$specificities))]
    best_threshold_ridge <- roc_ridge_train$thresholds[which.max(roc_ridge_train$sensitivities - (1 - roc_ridge_train$specificities))]
    best_threshold_logistic <- if (!is.null(y_logistic_train)) roc_logistic_train$thresholds[which.max(roc_logistic_train$sensitivities - (1 - roc_logistic_train$specificities))] else NA
  } else {
    best_threshold_blastvar <- best_threshold_GLMTrans <- best_threshold_ridge <- best_threshold_logistic <- thresh
  }
  
  # Convert to Binary Predictions (Train)
  pred_blastvar_train <- ifelse(y_blastvar_train > best_threshold_blastvar, 1, 0)
  pred_GLMTrans_train <- ifelse(y_GLMTrans_train > best_threshold_GLMTrans, 1, 0)
  pred_ridge_train <- ifelse(y_ridge_train > best_threshold_ridge, 1, 0)
  pred_logistic_train <- if (!is.null(y_logistic_train)) ifelse(y_logistic_train > best_threshold_logistic, 1, 0) else NULL
  
  # Compute MCC (Train)
  mcc_blastvar_train <- compute_mcc(y_train, pred_blastvar_train)
  mcc_GLMTrans_train <- compute_mcc(y_train, pred_GLMTrans_train)
  mcc_ridge_train <- compute_mcc(y_train, pred_ridge_train)
  mcc_logistic_train <- if (!is.null(pred_logistic_train)) compute_mcc(y_train, pred_logistic_train) else NA
  
  # AUC (Train)
  auc_blastvar_train <- roc_blastvar_train$auc
  auc_GLMTrans_train <- roc_GLMTrans_train$auc
  auc_ridge_train <- roc_ridge_train$auc
  auc_logistic_train <- if (!is.null(y_logistic_train)) roc_logistic_train$auc else NA
  
  # Model Predictions (Test)
  y_blastvar <- X_test %*% beta_hats$mu_TL_var %>% plogis()
  y_GLMTrans <- X_test %*% beta_hats$mu_GLMTrans %>% plogis()
  y_ridge <- X_test %*% beta_hats$ridge %>% plogis()
  y_logistic <- if (nrow(X_test) > ncol(X_test)) X_test %*% beta_hats$logistic %>% plogis() else NULL
  
  # ROC Curves (Test)
  roc_blastvar <- roc(y_test, y_blastvar, plotit = FALSE)
  roc_GLMTrans <- roc(y_test, y_GLMTrans, plotit = FALSE)
  roc_ridge <- roc(y_test, y_ridge, plotit = FALSE)
  roc_logistic <- if (!is.null(y_logistic)) roc(y_test, y_logistic, plotit = FALSE) else NULL
  
  # Convert to Binary Predictions (Test)
  pred_blastvar <- ifelse(y_blastvar > best_threshold_blastvar, 1, 0)
  pred_GLMTrans <- ifelse(y_GLMTrans > best_threshold_GLMTrans, 1, 0)
  pred_ridge <- ifelse(y_ridge > best_threshold_ridge, 1, 0)
  pred_logistic <- if (!is.null(y_logistic)) ifelse(y_logistic > best_threshold_logistic, 1, 0) else NULL
  
  # Compute MCC (Test)
  mcc_blastvar <- compute_mcc(y_test, pred_blastvar)
  mcc_GLMTrans <- compute_mcc(y_test, pred_GLMTrans)
  mcc_ridge <- compute_mcc(y_test, pred_ridge)
  mcc_logistic <- if (!is.null(pred_logistic)) compute_mcc(y_test, pred_logistic) else NA
  
  # AUC (Test)
  auc_blastvar <- roc_blastvar$auc
  auc_GLMTrans <- roc_GLMTrans$auc
  auc_ridge <- roc_ridge$auc
  auc_logistic <- if (!is.null(y_logistic)) roc_logistic$auc else NA
  
  # Return results as a data frame
  res <- data.frame(
    "Method" = c("BLAST", "GLMTrans", "Ridge", "Logistic"),
    "Train_MCC" = round(c(mcc_blastvar_train, mcc_GLMTrans_train, mcc_ridge_train, mcc_logistic_train), 4),
    "Test_MCC" = round(c(mcc_blastvar, mcc_GLMTrans, mcc_ridge, mcc_logistic), 4),
    "AUC_Train" = round(c(auc_blastvar_train, auc_GLMTrans_train, auc_ridge_train, auc_logistic_train), 4),
    "AUC_Test" = round(c(auc_blastvar, auc_GLMTrans, auc_ridge, auc_logistic), 4),
    "Thresholds" = round(c(best_threshold_blastvar, best_threshold_GLMTrans, best_threshold_ridge, best_threshold_logistic), 4)
  )
  
  return(res)
}


