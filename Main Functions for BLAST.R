library(tidyverse)
library(xgboost)
library(caret)
library(MASS)

## ---------------------  Regression ---------------------------------------
# Gives the posterior parameters for our method
posterior_param = function(X, y, X_tilde, Y_tilde_ML, sigma_t_hat, alpha_tilde, k = 10e-4, tau = 10e1) {
  p = ncol(X)
  I = diag(p) 
  
  XtX = t(X) %*% X
  XtX_tilde = t(X_tilde) %*% X_tilde
  XtY = t(X) %*% y
  XtY_tilde_ML = t(X_tilde) %*% Y_tilde_ML
  
  Sigma_post_inv = XtX / sigma_t_hat + k * alpha_tilde * XtX_tilde + (1 / tau^2) * I
  Sigma_post = chol2inv(chol(Sigma_post_inv))  
  
  # Compute A and b (avoid redundant calculations)
  A = (k * alpha_tilde * XtX_tilde + (1 / tau^2) * I) %*% solve(alpha_tilde * XtX_tilde + (1 / tau^2) * I)
  #A = I
  b = alpha_tilde * XtY_tilde_ML
  
  # Calculate mu_post
  mu_post = Sigma_post %*% (XtY / sigma_t_hat + A %*% b)
  
  return(list("Sigma_post" = Sigma_post, "mu_post" = mu_post))
}

# Generates the estimate for beta (mean of estimator distribution) using our method (blast)
# additional outputs: variance of estimator distribution, estimated sigma_s_hat
mu_generate_blast = function( X,y, X_tilde, ML, alpha=1, k = 10e-4, tau = 1){
  
  p = ncol(X)
  Y_tilde_ML = as.matrix(predict(ML, newdata = X_tilde))
  I = diag(p)
  
  sigma_t_hat = sigma_hat_estimate(y,X)
  sigma_s_hat = sigma_hat_estimate(Y_tilde_ML,X_tilde)
  
  pst_params_blast = posterior_param(X, y, X_tilde, Y_tilde_ML, sigma_t_hat,
                                    alpha_tilde = alpha/sigma_s_hat, k= k, tau = tau)  
  mu_blast = pst_params_blast$mu_post
  sigma_blast = pst_params_blast$Sigma_post
  
  return(list(
    "mu_blast" = mu_blast,
    "sigma_blast" = sigma_blast, 
    'sigma_s_hat' = sigma_s_hat
  )
  )
}

# The loop for Cross Validation on the hyperparameters 
cv_loop = function( X, y,  X_tilde, ML,
                    alpha_vals = c(0.1), k_vals = c(1e-4), tau_vals = c(1),
                    n_cores = 6, n_cv = 5){
  
  param_grid = expand.grid(alpha = alpha_vals, k = k_vals, tau = tau_vals)
  Y_tilde_ML = as.matrix(predict(ML, newdata = X_tilde))
  sigma_t_hat = sigma_hat_estimate(y, X)
  sigma_s_hat = sigma_hat_estimate(Y_tilde_ML, X_tilde)
  
  compute_cvfold = function(i) {
    alpha = param_grid$alpha[i]
    k = param_grid$k[i]
    tau_1 = param_grid$tau[i]
    
    pst_params_blast = posterior_param(X, y, X_tilde, Y_tilde_ML, 
                                      sigma_t_hat, alpha_tilde = alpha / sigma_s_hat,
                                      k = k, tau = tau_1)
    mu_blast = pst_params_blast$mu_post
    val_err_ = mean((y_val - X_val %*% mu_blast)^2) %>%  sqrt()
    return(c(alpha = alpha, k = k, tau = tau_1, error = val_err_))
  }
  
  folds <- createFolds(1:nrow(X), k = n_cv, list = TRUE)
  val_err = as.data.frame(matrix(0, nrow = (nrow(param_grid)), ncol = 4))
  colnames(val_err) = c("alpha", "k" ,    "tau" ,  "error")
  
  for(cv in 1:n_cv) {
    # Create folds
    train_indices = folds[[cv]]
    X_val = X[train_indices, ]
    y_val = y[train_indices]
    
    X_train = X[-train_indices, ]
    y_train = y[-train_indices]
    
    
    # Use the cluster with pblapply
    val_errs_res = pblapply(seq_len(nrow(param_grid)), function(i) {
      compute_cvfold(i)  
    }, cl = NULL)
    
    # Process the results
    val_errs_df = do.call(rbind, val_errs_res)
    val_errs_df = as.data.frame(val_errs_df)
    val_errs_df$error <- as.numeric(val_errs_df$error)
    
    # Accumulate errors
    val_err = val_err + val_errs_df / n_cv
  }
  
  
  
  val_errs_sorted = val_err[order(val_err$error), ]
  top_val_errs = val_errs_sorted[1:min(10, nrow(val_errs_sorted)), ]
  best_err_idx = which.min(val_err$error)
  best_params = val_err[best_err_idx, 1:3]
  
  return(list(best_params = best_params, val_errs = top_val_errs))
}


## ---------------------  Classification ---------------------------------------
# Find the Variational Bayes mean and variance given y ~ Bern(g(X*beta)) and prior
VarPostParam = function(X,y,mu_prior,Sigma_prior,
                        m_init = NULL, S_init = NULL,e_init = NULL,
                        IsY01 = T, thr = 0.01, verbose = F){
  if(IsY01 == T) y = 2*y - 1
  
  n = nrow(X)
  p = ncol(X)
  I = diag(p)
  
  #Initialising the values: m = 0, Sigma = I, e = vec(1) if nothing specified
  if(is.null(m_init)) m = rep(0,p) else m_init
  if(is.null(S_init)) S = I  else S = S_init
  if(is.null(e_init)) e = rep(1,n) else e_init
  
  m_old = 100000
  S_old = 10000*I
  j = 1
  
  while((mean(abs(m_old - m)) > thr) | (mean(abs(S_old - S)) > thr)){
    m_old = m
    S_old = S
    #E-Step
    lambda_e = (1/(2*e)*(plogis(e)- 0.5)) %>%  as.vector()
    #M-Step
    Sigma_prior_inv = chol2inv(chol(Sigma_prior))
    S = chol2inv(chol(Sigma_prior_inv + 2 * t(X) %*% (diag(lambda_e) %*% X)))
    m = S%*%(Sigma_prior_inv%*%mu_prior + colSums(as.vector(y)*X)/2)

    e <- sqrt(rowSums((X %*% S) * X) + (X %*% m)^2)
    if (verbose ==T )cat("Iteration", j, "Delta:", mean(abs(m_old - m)), "\n")
    j = j + 1
  }
  return(list("mu_blast" = m,"sigma_blast" = S))
}

# Gives the posterior parameters for our method
posterior_param_class = function(X,y, X_tilde, Y_tilde_ML, alpha=1, k = 0, tau = 1, verbose = F){
  p = ncol(X)
  I = diag(p)
  
  sourceprior_Mean = rep(0,p)
  sourceprior_Sigma = (1/tau^2) * I

  
  post_params = VarPostParam(X_tilde,Y_tilde_ML,sourceprior_Mean,sourceprior_Sigma, verbose = verbose)
  
  S = post_params$sigma_blast
  m = post_params$mu_blast
  
  S_inv = chol2inv(chol(S))
  targetprior_Sigma = chol2inv(chol(alpha*(S_inv - (tau^2)*I) + (tau^2)*I))
  targetprior_Mean = targetprior_Sigma%*%(alpha*S_inv%*%m)
  
  #Scaling the variance by k
  targetprior_Sigma = chol2inv(chol(k*alpha*(S_inv - (tau^2)*I) + (tau^2)*I))

  post_params = VarPostParam(X,y,targetprior_Mean,targetprior_Sigma, verbose = verbose)
  
  return(post_params)
}

# Generates the estimate for beta (mean of estimator distribution) using our method (blast)
# additional outputs: variance of estimator distribution
mu_generate_blast_class = function( X,y, X_tilde, ML, alpha=1, k = 0, tau = 1){
  p = ncol(X)
  Y_tilde_ML = ifelse(predict(ML, newdata = X_tilde) > 0.5, 1, 0)
  I = diag(p)
  
  pst_params_blast_class = posterior_param_class(X, y, X_tilde, Y_tilde_ML,
                                                alpha = alpha, k= k, tau = tau)  
  mu_blast = pst_params_blast_class$mu_blast
  sigma_blast = pst_params_blast_class$sigma_blast
  
  return(list(
    "mu_blast" = mu_blast,
    "sigma_blast" = sigma_blast
  )
  )
}

# The loop for Cross Validation on the hyperparameters 
cv_loop_class = function(X, y,  X_tilde, ML,
                    alpha_vals = seq(2, 5, length = 5),
                    k_vals = c(0), 
                    tau_vals = seq(2, 5, length = 5),
                    n_cores = 6, n_cv = 5){
  
  param_grid = expand.grid(alpha = alpha_vals, k = k_vals, tau = tau_vals)
  Y_tilde_ML = as.matrix(predict(ML, newdata = X_tilde))
  
  compute_cvfold = function(i) {
    alpha = param_grid$alpha[i]
    k = param_grid$k[i]
    tau_1 = param_grid$tau[i]
    
    pst_params_blast = posterior_param_class(X,y, X_tilde, Y_tilde_ML,alpha = alpha,
                                      k = k, tau = tau_1)
    mu_blast = pst_params_blast$mu_blast
    
    p_val = plogis(X_val %*% mu_blast)
    val_err_ = mean(y_val * log(p_val) + (1-y_val) * (1-log(p_val)))
    return(c(alpha = alpha, k = k, tau = tau_1, error = val_err_))
  }
  
  folds <- createFolds(1:nrow(X), k = n_cv, list = TRUE)
  val_err = as.data.frame(matrix(0, nrow = (nrow(param_grid)), ncol = 4))
  colnames(val_err) = c("alpha", "k" ,    "tau" ,  "error")
  
  for(cv in 1:n_cv) {
    # Create folds
    train_indices = folds[[cv]]
    X_val = X[train_indices, ]
    y_val = y[train_indices]
    
    X_train = X[-train_indices, ]
    y_train = y[-train_indices]
    
    
    # Use the cluster with pblapply
    val_errs_res = pblapply(seq_len(nrow(param_grid)), function(i) {
      compute_cvfold(i)  
    }, cl = NULL)
    
    # Process the results
    val_errs_df = do.call(rbind, val_errs_res)
    val_errs_df = as.data.frame(val_errs_df)
    val_errs_df$error <- as.numeric(val_errs_df$error)
    
    # Accumulate errors
    val_err = val_err + val_errs_df / n_cv
  }
  
  
  
  val_errs_sorted = val_err[order(val_err$error), ]
  top_val_errs = val_errs_sorted[1:min(10, nrow(val_errs_sorted)), ]
  best_err_idx = which.min(val_err$error)
  best_params = val_err[best_err_idx, 1:3]
  
  return(list(best_params = best_params, val_errs = top_val_errs))
}


