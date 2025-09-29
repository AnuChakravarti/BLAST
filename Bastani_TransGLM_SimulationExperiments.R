library(xgboost)
library(dplyr)
library(MASS)
library(pbapply)
library(parallel)

source("Helper Functions.R")
source("Main Functions for BLAST.R")


######################################################################
########### Extra Experiments 1 : Bastani Simulation #################



## -------- Data Generation ----------
 # For multivariate normal distribution

nproxy <- 1000
ngold <- 150
ntest <- 1000
d <- 100
N_source <- 1000

generate_data <- function(seed) {
  set.seed(seed) # Set seed for reproducibility
  
  # Parameters

  beta_gold <- rep(1, d) # True parameter β*gold
  trace_normalize <- function(mat) mat / sum(diag(mat)) # Trace normalization function
  
  # Generate positive-definite covariance matrix
  random_matrix <- matrix(runif(d^2, 0, 1), nrow = d)
  cov_matrix <- trace_normalize(crossprod(random_matrix)) # Positive-definite, trace normalized
  
  # Generate proxy, gold, and test data
  Xproxy <- mvrnorm(nproxy, mu = rep(0, d), Sigma = cov_matrix)
  Xgold <-  mvrnorm(ngold, mu = rep(0, d), Sigma = cov_matrix)
  Xtest <- mvrnorm(ntest, mu = rep(0, d), Sigma = cov_matrix)
  X_tilde <- mvrnorm(N_source, mu = rep(0, d), Sigma = cov_matrix)
  
  # Sparse and non-sparse δ*
  delta_sparse <- 0.1 * rbinom(d, size = 1, prob = 0.1) # Sparse δ*
  delta_nonsparse <- rnorm(d, mean = 0, sd = 0.15)      # Non-sparse δ*
  
  # Noise terms
  epsilon_proxy <- rnorm(nproxy, mean = 0, sd = 1)
  epsilon_gold <- rnorm(ngold, mean = 0, sd = 1)
  epsilon_test <- rnorm(ntest, mean = 0, sd = 1)
  
  # Responses
  Yproxy_sparse <- Xproxy %*% (beta_gold - delta_sparse) + epsilon_proxy
  Yproxy_nonsparse <- Xproxy %*% (beta_gold - delta_nonsparse) + epsilon_proxy
  Ygold <- Xgold %*% beta_gold + epsilon_gold
  Ytest <- Xtest %*% beta_gold + epsilon_test
  
  # Prepare input data
  data_unknown_nonsparse <- cbind.data.frame(Yproxy_nonsparse, Xproxy)
  data_unknown_sparse <- cbind.data.frame(Yproxy_sparse, Xproxy)
  
  # Build the ML model
  ml_model_nonsparse <- build_ML(data_unknown_nonsparse)
  ml_model_sparse <- build_ML(data_unknown_sparse)
  X_tilde_dmatrix <- xgb.DMatrix(data = X_tilde)
  
  # Predict Y_tilde_ML for sparse and non-sparse responses
  Y_tilde_ML_sparse <- predict(ml_model_sparse$ML_model, newdata = X_tilde_dmatrix)
  Y_tilde_ML_nonsparse <- predict(ml_model_nonsparse$ML_model, newdata = X_tilde_dmatrix)
  
  # Organize the data into a list
  Dat <- list(
    'data_source_nonsparse' = cbind.data.frame(Y_tilde_ML_nonsparse, X_tilde), 
    'beta_tilde_nonsparse' = (beta_gold - delta_nonsparse), 
    'ML_nonsparse' = ml_model_nonsparse,
    
    'data_source_sparse' = cbind.data.frame(Y_tilde_ML_sparse, X_tilde), 
    'beta_tilde_sparse' = (beta_gold - delta_sparse),
    'ML_sparse' = ml_model_sparse,
    
    'data_target' = cbind.data.frame(Ygold, Xgold), 
    'beta' = beta_gold,
    
    'data_test' = cbind.data.frame(Ytest, Xtest)
  )
  
  # Return the generated data
  return(Dat)
}

Dat = generate_data(1)

# Example usage:
# dat_result <- generate_data(seed = 123)
# str(dat_result)


## -------- Applying Methods -----

X = Dat$data_target[,-1] %>%  as.matrix()
y = Dat$data_target[,1] %>%  as.matrix()
X_tilde = Dat$data_source_sparse[,-1]%>%  as.matrix()
Y_tilde_ML_sparse = Dat$data_source_sparse[,1]%>%  as.matrix()
Y_tilde_ML_nonsparse = Dat$data_source_nonsparse[,1]%>%  as.matrix()
ML_sparse = Dat$ML_sparse
ML_nonsparse = Dat$ML_nonsparse



##### ----- blast ---- 

alpha_vals = seq(0, 5, length = 10)
k_vals = c(0)
tau_vals = seq(0.01, 0.5, length = 10)

##### --- non-sparse

hypers_training <- cv_loop(X, y, X_tilde, ML_sparse,
                           alpha_vals = alpha_vals,
                           k_vals = k_vals,
                           tau_vals = tau_vals, 
                           n_cores = n_cores, 
                           n_cv = 5)

best_alpha <- hypers_training$best_params$alpha
best_k <- hypers_training$best_params$k
best_tau <- hypers_training$best_params$tau
print(hypers_training$best_params)


mu_blast_sparse <- mu_generate_blast( X, y, X_tilde, ML_sparse,
                            alpha = best_alpha, 
                            k = best_k, 
                            tau = best_tau)$mu_blast

##### --- non-sparse

alpha_vals = seq(0, 2, length = 10)
k_vals = c(0)
tau_vals = seq(0.01, 0.5, length = 10)

hypers_training <- cv_loop(X, y, X_tilde, ML_nonsparse,
                           alpha_vals = alpha_vals,
                           k_vals = k_vals,
                           tau_vals = tau_vals, 
                           n_cores = n_cores, 
                           n_cv = 5)

best_alpha <- hypers_training$best_params$alpha
best_k <- hypers_training$best_params$k
best_tau <- hypers_training$best_params$tau
print(hypers_training$best_params)


mu_blast_nonsparse <- mu_generate_blast( X, y, X_tilde, ML_nonsparse,
                             alpha = best_alpha, 
                             k = best_k, 
                             tau = best_tau)$mu_blast





##### ----- Ridge ---- 

mu_ridge = mu_generate_ridge(X,y, lambda_values = seq(0.000001, 10, length = 100))%>%  unlist()

##### ----- TransLasso ---- 
mu_transLasso_sparse = mu_generate_TransLasso(X_tilde, X,y, Y_tilde_ML_sparse,size.A0 = 1)
mu_transLasso0_sparse = mu_generate_TransLasso(X_tilde, X,y, Y_tilde_ML_sparse,size.A0 = 0)

mu_transLasso_nonsparse = mu_generate_TransLasso(X_tilde, X,y, Y_tilde_ML_nonsparse,size.A0 = 1)
mu_transLasso0_nonsparse = mu_generate_TransLasso(X_tilde, X,y, Y_tilde_ML_nonsparse,size.A0 = 0)

##### ----- GLMTrans ---- 

mu_GLMTrans_sparse = mu_generate_GLMTrans(X_tilde, X,y, Y_tilde_ML_sparse) %>%  unlist()
mu_GLMTrans_nonsparse = mu_generate_GLMTrans(X_tilde, X,y, Y_tilde_ML_nonsparse)%>%  unlist()

##### ----- PPI ---- 

mu_ppi_sparse = mu_generate_ppi(X_tilde, X,y,Y_tilde_ML_sparse,ML_sparse) %>%  unlist()
mu_ppi_nonsparse = mu_generate_ppi(X_tilde, X,y,Y_tilde_ML_nonsparse,ML_nonsparse) %>%  unlist()

##### ----- OLS ---- 

mu_ols = mu_generate_OLS(X, y)




## ------- Comparing Errors ####


data_test = Dat$data_test
data_train = Dat$data_target
beta = Dat$beta

beta_hats_sparse <- list(
  'mu_TL' = mu_blast_sparse,
  "mu_ppi" = mu_ppi_sparse,
  "mu_GLMTrans" = mu_GLMTrans_sparse, 
  "mu_TransLasso" = mu_transLasso_sparse,
  "mu_TransLasso0" = mu_transLasso0_sparse,
  'ridge' = mu_ridge,
  'ols' = mu_ols
)


beta_hats_nonsparse <- list(
  'mu_TL' = mu_blast_nonsparse,
  "mu_ppi" = mu_ppi_nonsparse,
  "mu_GLMTrans" = mu_GLMTrans_nonsparse, 
  "mu_TransLasso" = mu_transLasso_nonsparse,
  "mu_TransLasso0" = mu_transLasso0_nonsparse,
  'ridge' = mu_ridge,
  'ols' = mu_ols
)



check_errors(data_train, data_test, beta_hats_nonsparse, beta)
check_errors(data_train, data_test, beta_hats_sparse, beta)










######### ---------- Loop Over Seeds ###########

results_sparse <- list()
results_nonsparse <- list()


for (seed in 1:20) {
  Dat <- generate_data(seed)
  
  X <- Dat$data_target[, -1] %>% as.matrix()
  y <- Dat$data_target[, 1] %>% as.matrix()
  X_tilde <- Dat$data_source_sparse[, -1] %>% as.matrix()
  Y_tilde_ML_sparse <- Dat$data_source_sparse[, 1] %>% as.matrix()
  Y_tilde_ML_nonsparse <- Dat$data_source_nonsparse[, 1] %>% as.matrix()
  ML_sparse <- Dat$ML_sparse
  ML_nonsparse <- Dat$ML_nonsparse
  
  alpha_vals <- seq(0, 5, length = 10)
  k_vals <- c(0)
  tau_vals <- seq(0.01, 0.5, length = 10)
  
  hypers_training <- cv_loop(X, y, X_tilde, ML_sparse,
                             alpha_vals = alpha_vals,
                             k_vals = k_vals,
                             tau_vals = tau_vals,
                             n_cores = n_cores,
                             n_cv = 5)
  
  best_alpha <- hypers_training$best_params$alpha
  best_k <- hypers_training$best_params$k
  best_tau <- hypers_training$best_params$tau
  
  mu_blast_sparse <- mu_generate_blast(X, y, X_tilde, ML_sparse,
                                     alpha = best_alpha,
                                     k = best_k,
                                     tau = best_tau)$mu_blast
  
  alpha_vals <- seq(0, 2, length = 10)
  hypers_training <- cv_loop(X, y, X_tilde, ML_nonsparse,
                             alpha_vals = alpha_vals,
                             k_vals = k_vals,
                             tau_vals = tau_vals,
                             n_cores = n_cores,
                             n_cv = 5)
  
  best_alpha <- hypers_training$best_params$alpha
  best_k <- hypers_training$best_params$k
  best_tau <- hypers_training$best_params$tau
  
  mu_blast_nonsparse <- mu_generate_blast(X, y, X_tilde, ML_nonsparse,
                                        alpha = best_alpha,
                                        k = best_k,
                                        tau = best_tau)$mu_blast
  
  mu_ridge <- mu_generate_ridge(X, y, lambda_values = seq(0.000001, 10, length = 100)) %>% unlist()
  
  mu_transLasso_sparse <- mu_generate_TransLasso(X_tilde, X, y, Y_tilde_ML_sparse, size.A0 = 1)
  mu_transLasso0_sparse <- mu_generate_TransLasso(X_tilde, X, y, Y_tilde_ML_sparse, size.A0 = 0)
  mu_transLasso_nonsparse <- mu_generate_TransLasso(X_tilde, X, y, Y_tilde_ML_nonsparse, size.A0 = 1)
  mu_transLasso0_nonsparse <- mu_generate_TransLasso(X_tilde, X, y, Y_tilde_ML_nonsparse, size.A0 = 0)
  
  mu_GLMTrans_sparse <- mu_generate_GLMTrans(X_tilde, X, y, Y_tilde_ML_sparse) %>% unlist()
  mu_GLMTrans_nonsparse <- mu_generate_GLMTrans(X_tilde, X, y, Y_tilde_ML_nonsparse) %>% unlist()
  
  mu_ppi_sparse <- mu_generate_ppi(X_tilde, X, y, Y_tilde_ML_sparse, ML_sparse) %>% unlist()
  mu_ppi_nonsparse <- mu_generate_ppi(X_tilde, X, y, Y_tilde_ML_nonsparse, ML_nonsparse) %>% unlist()
  
  mu_ols <- mu_generate_OLS(X, y)
  
  data_test <- Dat$data_test
  data_train <- Dat$data_target
  beta <- Dat$beta
  
  beta_hats_sparse <- list(
    'mu_TL' = mu_blast_sparse,
    'mu_ppi' = mu_ppi_sparse,
    'mu_GLMTrans' = mu_GLMTrans_sparse,
    'mu_TransLasso' = mu_transLasso_sparse,
    'mu_TransLasso0' = mu_transLasso0_sparse,
    'ridge' = mu_ridge,
    'ols' = mu_ols
  )
  
  beta_hats_nonsparse <- list(
    'mu_TL' = mu_blast_nonsparse,
    'mu_ppi' = mu_ppi_nonsparse,
    'mu_GLMTrans' = mu_GLMTrans_nonsparse,
    'mu_TransLasso' = mu_transLasso_nonsparse,
    'mu_TransLasso0' = mu_transLasso0_nonsparse,
    'ridge' = mu_ridge,
    'ols' = mu_ols
  )
  
  results_nonsparse[[seed]] <- check_errors(data_train, data_test, beta_hats_nonsparse, beta)
  print(results_sparse)
  results_sparse[[seed]] <- check_errors(data_train, data_test, beta_hats_sparse, beta)
  print(results_sparse)
}


summary_sparse <- do.call(rbind, results_sparse) %>%
  group_by(Method) %>%
  summarise(
    avg_train_error = mean(Train_error),
    sd_train_error = sd(Train_error),
    avg_test_error = mean(Test_error),
    sd_test_error = sd(Test_error),
    avg_beta_error = mean(beta_error),
    sd_beta_error = sd(beta_error)
  )

summary_nonsparse <- do.call(rbind, results_nonsparse) %>%
  group_by(Method) %>%
  summarise(
    avg_train_error = mean(Train_error),
    sd_train_error = sd(Train_error),
    avg_test_error = mean(Test_error),
    sd_test_error = sd(Test_error),
    avg_beta_error = mean(beta_error),
    sd_beta_error = sd(beta_error)
  )

print(summary_sparse)
print(summary_nonsparse)

dir.create("Bastani Simulation", showWarnings = FALSE)

saveRDS(summary_sparse, "Bastani Simulation/summary_sparse.rds")
saveRDS(summary_nonsparse, "Bastani Simulation/summary_nonsparse.rds")
saveRDS(results_sparse, "Bastani Simulation/results_sparse.rds")
saveRDS(results_nonsparse, "Bastani Simulation/results_nonsparse.rds")




readRDS('Bastani Simulation/summary_nonsparse.rds')
readRDS('Bastani Simulation/summary_sparse.rds')




######################################################################

########### Extra Experiments 2 : TransGLM Simulation #################




## -------- Data Generation ----------
# For multivariate normal distribution

n0 = 100
nk = 100 
p = 250
s = 75
h = 5
N_source = 1000
ntest = 1000
library(xgboost)  # For XGBoost (assuming you use this ML model)
library(caret)  # For ML model training

generate_data<- function(seed) {
  set.seed(seed)  # Set seed for reproducibility
  
  # Define the true coefficients for target
  beta_gold <- c(rep(0.5, s), rep(0, p - s))  # True coefficient vector
  
  # Covariance matrix Σ for target data
  Sigma <- outer(1:p, 1:p, FUN = function(i, j) 0.5^abs(i - j))  # AR(1) covariance
  
  # Generate target data (X_0, Y_0)
  X_target <- mvrnorm(n0, mu = rep(0, p), Sigma = Sigma)
  epsilon_target <- rnorm(n0, mean = 0, sd = 1)
  Y_target <- X_target %*% beta_gold + epsilon_target  # Linear response
  
  X_test<- mvrnorm(ntest, mu = rep(0, p), Sigma = Sigma)
  epsilon_test <- rnorm(ntest, mean = 0, sd = 1)
  Y_test <- X_test %*% beta_gold + epsilon_test
  
  # Generate source data (X_k, Y_k)
  epsilon_source <- rnorm(nk, mean = 0, sd = 1)
  epsilon_cov <- mvrnorm(nk, mu = rep(0, p), Sigma = diag(0.3^2, p)) %>%  t() # Variance perturbation
  
  # Modified covariance matrix for source data
  
  perturbation_matrix <- epsilon_cov %*% t(epsilon_cov) / nk  # Rank-1 perturbation
  
  # Modify the covariance matrix
  Sigma_source <- Sigma + perturbation_matrix
  
  
  X_source <- mvrnorm(nk, mu = rep(0, p), Sigma = Sigma_source)
  beta_source <- beta_gold + (h / p) * sample(c(-1, 1), p, replace = TRUE)  # Rademacher perturbation
  Y_source <- X_source %*% beta_source + epsilon_source  # Linear response
  
  # Generate new X_tilde data
  X_tilde <- mvrnorm(N_source, mu = rep(0, p), Sigma = Sigma)
  
  # Combine source data into one dataset
  data_unknown <- cbind(Y_source, X_source)
  
  # Build ML model for source data
  ml_model <- build_ML(data_unknown)  # XGBoost model
  
  # Predict Y_tilde_ML for X_tilde
  Y_tilde_ML <- predict(ml_model, newdata = X_tilde)
  
  # Prepare the data into a list for return
  Dat <- list(
    'data_target' = cbind.data.frame(Y_target, X_target),
    'data_source' = cbind.data.frame(Y_tilde_ML, X_tilde),
    'data_test' = cbind.data.frame(Y_test, X_test),
    'beta' = beta_gold,
    'beta_tilde' = beta_source,
    'ML' = ml_model
  )
  
  # Return the generated data
  return(Dat)
}

# Example usage:
Dat <- generate_data(seed = 1)

# Example usage:
# dat_result <- generate_data(seed = 123)
# str(dat_result)


## -------- Applying Methods -----

X = Dat$data_target[,-1] %>%  as.matrix()
y = Dat$data_target[,1] %>%  as.matrix()
X_tilde = Dat$data_source[,-1]%>%  as.matrix()
Y_tilde_ML = Dat$data_source[,1]%>%  as.matrix()
ML = Dat$ML



##### ----- blast ---- 

alpha_vals = seq(0, 0.5, length = 10)
k_vals = c(0)
tau_vals = seq(0.01, 0.25, length = 10)

hypers_training <- cv_loop(X, y, X_tilde, ML,
                           alpha_vals = alpha_vals,
                           k_vals = k_vals,
                           tau_vals = tau_vals, 
                           n_cores = n_cores, 
                           n_cv = n_cv)

best_alpha <- hypers_training$best_params$alpha
best_k <- hypers_training$best_params$k
best_tau <- hypers_training$best_params$tau
print(hypers_training$best_params)


mu_blast <- mu_generate_blast( X, y, X_tilde, ML,
                                    alpha = best_alpha, 
                                    k = best_k, 
                                    tau = best_tau)$mu_blast


##### ----- Ridge ---- 

mu_ridge = mu_generate_ridge(X,y, lambda_values = seq(0.000001, 10, length = 100))%>%  unlist()

##### ----- TransLasso ---- 
mu_transLasso = mu_generate_TransLasso(X_tilde, X,y, Y_tilde_ML,size.A0 = 1)
mu_transLasso0= mu_generate_TransLasso(X_tilde, X,y, Y_tilde_ML,size.A0 = 0)

##### ----- GLMTrans ---- 

mu_GLMTrans = mu_generate_GLMTrans(X_tilde, X,y, Y_tilde_ML) %>%  unlist()
##### ----- PPI ---- 

#mu_ppi_sparse = mu_generate_ppi(X_tilde, X,y,Y_tilde_ML,ML) %>%  unlist()

##### ----- OLS ---- 

#mu_ols = mu_generate_OLS(X, y)




## ------- Comparing Errors ####


data_test = Dat$data_test
data_train = Dat$data_target
beta = Dat$beta

beta_hats <- list(
  'mu_TL' = mu_blast,
  #"mu_ppi" = mu_ppi_sparse,
  "mu_GLMTrans" = mu_GLMTrans, 
  "mu_TransLasso" = mu_transLasso,
  "mu_TransLasso0" = mu_transLasso0,
  'ridge' = mu_ridge
  #'ols' = mu_ols
)



check_errors(data_train, data_test, beta_hats, beta)




######### ---------- Loop Over Seeds ###########



n0 = 100
nk = 100 
p = 250
s = 75
h = 25
N_source = 1000
ntest = 1000


results <- list()

for (seed in 1:20) {
  Dat <- generate_data(seed = seed)
  
  X = Dat$data_target[,-1] %>%  as.matrix()
  y = Dat$data_target[,1] %>%  as.matrix()
  X_tilde = Dat$data_source[,-1]%>%  as.matrix()
  Y_tilde_ML = Dat$data_source[,1]%>%  as.matrix()
  ML = Dat$ML
  
  
  
  ##### ----- blast ---- 
  
  alpha_vals = seq(0, 1, length = 10)
  k_vals = c(0)
  tau_vals = seq(0.01, 0.25, length = 10)
  
  
  hypers_training <- cv_loop(X, y, X_tilde, ML,
                             alpha_vals = alpha_vals,
                             k_vals = k_vals,
                             tau_vals = tau_vals, 
                             n_cores = n_cores, 
                             n_cv = n_cv)
  
  best_alpha <- hypers_training$best_params$alpha
  best_k <- hypers_training$best_params$k
  best_tau <- hypers_training$best_params$tau
  print(hypers_training$best_params)
  
  
  mu_blast <- mu_generate_blast( X, y, X_tilde, ML,
                               alpha = best_alpha, 
                               k = best_k, 
                               tau = best_tau)$mu_blast
  
  
  ##### ----- Ridge ---- 
  
  mu_ridge = mu_generate_ridge(X,y, lambda_values = seq(0.000001, 10, length = 100))%>%  unlist()
  
  ##### ----- TransLasso ---- 
  mu_transLasso = mu_generate_TransLasso(X_tilde, X,y, Y_tilde_ML,size.A0 = 1)
  mu_transLasso0= mu_generate_TransLasso(X_tilde, X,y, Y_tilde_ML,size.A0 = 0)
  
  ##### ----- GLMTrans ---- 
  
  mu_GLMTrans = mu_generate_GLMTrans(X_tilde, X,y, Y_tilde_ML) %>%  unlist()
  ##### ----- PPI ---- 
  
  #mu_ppi_sparse = mu_generate_ppi(X_tilde, X,y,Y_tilde_ML,ML) %>%  unlist()
  
  ##### ----- OLS ---- 
  
  #mu_ols = mu_generate_OLS(X, y)
  
  
  
  
  ## ------- Comparing Errors ####
  
  
  data_test = Dat$data_test
  data_train = Dat$data_target
  beta = Dat$beta
  
  beta_hats <- list(
    'mu_TL' = mu_blast,
    #"mu_ppi" = mu_ppi_sparse,
    "mu_GLMTrans" = mu_GLMTrans, 
    "mu_TransLasso" = mu_transLasso,
    "mu_TransLasso0" = mu_transLasso0,
    'ridge' = mu_ridge
    #'ols' = mu_ols
  )
  
  
  
  results[[seed]] = check_errors(data_train, data_test, beta_hats, beta)
  print(results)

}


summary_results <- do.call(rbind, results) %>%
  group_by(Method) %>%
  summarise(
    avg_train_error = mean(Train_error),
    sd_train_error = sd(Train_error),
    avg_test_error = mean(Test_error),
    sd_test_error = sd(Test_error),
    avg_beta_error = mean(beta_error),
    sd_beta_error = sd(beta_error)
  )


print(summary_results)

dir.create("GLMTrans Simulation", showWarnings = FALSE)

saveRDS(summary_results, "GLMTrans Simulation/summary_results_highbias.rds")
saveRDS(results, "Bastani Simulation/results_highbias.rds")




readRDS('GLMTrans Simulation/summary_results.rds')
readRDS('GLMTrans Simulation/summary_results_highbias.rds')











## -------- Comparing Errors -----