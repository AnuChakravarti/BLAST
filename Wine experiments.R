library(data.table)
library(dplyr)

source("Helper Functions.R")
source("Main Functions for BLAST.R")

set.seed(2609)
redwinedata = fread("../Data/wine+quality/winequality-red.csv") %>% 
  mutate(Intercept = 1) %>%
  relocate(Intercept) %>%
  dplyr::select(quality, everything()) #%>% 
  #mutate(across(c(2:ncol(.)), scale))

whitewinedata = fread("../Data/wine+quality/winequality-white.csv") %>%  
  mutate(Intercept = 1) %>%
  relocate(Intercept) %>%
  dplyr::select(quality, everything())# %>% 
  #mutate(across(c(2:ncol(.)), scale))

Wine_experiment = function(alpha_vals,tau_vals, unlab_samples = 1200,
                              train_samples = 100,
                           n_cv = 5,n_cores = 6,seed = 1){
  sam = sample(nrow(redwinedata), unlab_samples)
  unlabelled_data = redwinedata[sam,]
  target_data = redwinedata[-sam,]
  sam = sample(nrow(target_data), train_samples)
  
  train_data <- target_data[sam,]
  test_data <- target_data[-sam,]
  rm(sam)
  
  data = list("unlabelled_data"=unlabelled_data,
              "target_data"=target_data,
              "train_data"=train_data,
              "test_data"=test_data)
  
  ML <- build_ML(whitewinedata)$ML_model
  
  X <- train_data[,-1]  %>% as.matrix() 
  y <- train_data[,1] %>%  as.matrix()
  
  X_tilde <- unlabelled_data[,-1] %>% as.matrix()
  
  n <- nrow(train_data)
  p <- ncol(train_data) - 1  # Exclude the target variable column
  
  # Doing CV to find the best hyperparamaters
  hypers_training <- cv_loop(X, y, X_tilde, ML,
                             alpha_vals = alpha_vals,
                             k_vals = c(0),
                             tau_vals = tau_vals, 
                             n_cores = n_cores, 
                             n_cv = n_cv)
  
  best_alpha <- hypers_training$best_params$alpha
  best_k <- hypers_training$best_params$k
  best_tau <- hypers_training$best_params$tau
  
  Y_tilde_ML = as.matrix(predict(ML, newdata = X_tilde))
  
  # Generate the betas for all the methods
  mu_blast <- mu_generate_blast(X, y, X_tilde, ML,
                              alpha = best_alpha, 
                              k = best_k, 
                              tau = best_tau)$mu_blast
  mu_ppi = mu_generate_ppi(X_tilde, X,y,Y_tilde_ML,ML) 
  mu_GLMTrans = mu_generate_GLMTrans(X_tilde, X,y, Y_tilde_ML)
  mu_transLasso = mu_generate_TransLasso(X_tilde, X,y, Y_tilde_ML,size.A0 = 1)
  mu_transLasso0 = mu_generate_TransLasso(X_tilde, X,y, Y_tilde_ML,size.A0 = 0)
  mu_ridge = mu_generate_ridge(X,y, lambda_values = seq(0.000001, 10, length = 100))
  mu_ols = if(n>p) mu_generate_ols(X,y) else NULL
  
  beta_hats = list('mu_TL' = mu_blast,
                   "mu_ppi" = mu_ppi,
                   "mu_GLMTrans" = mu_GLMTrans, 
                   "mu_TransLasso" = mu_transLasso,
                   "mu_TransLasso0" = mu_transLasso0,
                   'ridge' = mu_ridge,
                   'ols' = mu_ols)
  
  
  # Testing to find errors
  X_test <- test_data[,-1]  %>% as.matrix()
  y_test <- test_data[,1] %>% as.matrix()
  
  # Computing predicted value for test using the betas
  y_blast <- X_test %*% beta_hats$mu_TL 
  y_ppi <- X_test %*% beta_hats$mu_ppi 
  y_GLMTrans <- X_test %*% beta_hats$mu_GLMTrans
  y_TransLasso <- X_test %*% beta_hats$mu_TransLasso
  y_TransLasso0 <- X_test %*% beta_hats$mu_TransLasso0
  y_ridge <- X_test %*% beta_hats$ridge  
  y_ols <- if (n> p) X_test %*% beta_hats$ols else NULL
  
  # Computing test error
  pred_err_blast <- sqrt(mean((y_blast - y_test)^2))
  pred_err_ppi <- sqrt(mean((y_ppi - y_test)^2)) 
  pred_err_GLMTrans <- sqrt(mean((y_GLMTrans - y_test)^2))
  pred_err_TransLasso1 <- sqrt(mean((y_TransLasso - y_test)^2))
  pred_err_TransLasso0 <- sqrt(mean((y_TransLasso0 - y_test)^2))
  pred_err_TransLasso <- min(pred_err_TransLasso1, pred_err_TransLasso0)
  best_transLasso <- which.min(c(pred_err_TransLasso1, pred_err_TransLasso0)) 
  pred_err_ridge <- sqrt(mean((y_ridge - y_test)^2)) 
  pred_err_ols <- if (!is.null(y_ols)) sqrt(mean((y_ols - y_test)^2)) else NA
  
  
  # Computing the predictions for the train y
  y_blast_train <- X %*% beta_hats$mu_TL
  y_ppi_train <- X %*% beta_hats$mu_ppi 
  y_GLMTrans_train <- X %*% beta_hats$mu_GLMTrans
  if (best_transLasso == 1) y_TransLasso_train <- X %*% beta_hats$mu_TransLasso
  if (best_transLasso == 2) y_TransLasso_train <- X %*% beta_hats$mu_TransLasso0
  y_ridge_train <- X %*% beta_hats$ridge 
  y_ols_train <- if (n> p) X %*% beta_hats$ols else NULL
  
  
  # Computing Train errors
  pred_err_blast_train <- sqrt(mean((y_blast_train - y)^2))
  pred_err_ppi_train <-sqrt(mean((y_ppi_train - y)^2)) 
  pred_err_GLMTrans_train <- sqrt(mean((y_GLMTrans_train - y)^2))
  pred_err_TransLasso_train <- sqrt(mean((y_TransLasso_train - y)^2))
  pred_err_ridge_train <- sqrt(mean((y_ridge_train - y)^2))  
  pred_err_ols_train <- if (!is.null(y_ols_train)) sqrt(mean((y_ols_train - y)^2))  else NA
  
  result = data.frame(
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
                           pred_err_ols), 4))
  
  return(list(data = data,beta_hats = beta_hats,result = result, best_params = hypers_training$best_params))
}

##### -------------- Experiment for given seed ---------------------############
alpha_vals = seq(0, 20, length = 50)
tau_vals = seq(0.01, 0.05, length = 50)

final = Wine_experiment(alpha_vals = alpha_vals,tau_vals = tau_vals,seed = 2609,
                        n_cv = 3)

final$result
final$best_params


##### -------------- Experiment over seeds --------------------------############

alpha_vals = seq(0, 20, length = 50)
tau_vals = seq(0.01, 0.05, length = 50)
all_errors <- list()
for(i in 1:20){
  final = Wine_experiment(alpha_vals = alpha_vals,tau_vals = tau_vals,seed = 2600+i,
                          n_cv = 3)
  all_errors[[i]] <- final$result
}

errors_df <- do.call(rbind, all_errors)
error_summary <- errors_df %>%
  group_by(Method) %>%
  summarise(
    avg_train_error = mean(Train_error),
    sd_train_error = sd(Train_error),
    avg_test_error = mean(Test_error),
    sd_test_error = sd(Test_error)
  )

# Printing the output
error_summary %>%
  arrange(avg_test_error)
