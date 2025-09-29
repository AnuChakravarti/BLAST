source("Helper Functions.R")
source("Main Functions for BLAST.R")

### Date : 01/07/25 ###
### Note: all sigma_t_hat and sigma_s_hat are in square order ###


####################################################################################
##################### FUNCTIONS ###########################################

# Generates the datasets required for our simulations
# Input: normal(T)-Beta values generated as N(0,1) + mean, (F)-Unif(-1,1) + mean
# sparse(T)- 75% parameter coeffs 0, rest 1.
# alt (T)- parameter coeffs take both negative and positive values, abs value similar. 

# Output: data_unknown - data on which ML model is trained on, unknown to user of algorithm,
# data_source - only X values, the unlabelled large dataset 
# data_target - target data (dataset of interest) 
# data_test - independent testing data for experiments (same DGP as target data). 
# beta_tilde - beta by which the unknown data y's are generated. 
# beta - beta by which the target data y's are generated (true beta of interest)

data_generate = function (seed, hypers, normal = T, sparse = F, alt = F){
  N = hypers$N 
  n = hypers$n
  N_u = hypers$N_u
  p = hypers$p
  sigma_s = hypers$sigma_s
  sigma_t = hypers$sigma_t
  beta_0= hypers$beta_0
  beta_diff = hypers$beta_diff
  num_features <- p
  z = rbinom(num_features, size = 1, prob  = 0.25) #for sparsity 
  r = 2* (rbinom (num_features, size = 1, prob = 0.5) - 0.5) #for alternating coeff
  
  set.seed (seed)
  
  ##### SOURCE (We have the X values but not the y) #####
  # Generate features from random Gaussian distribution
  X <- matrix(rnorm(N * num_features), ncol = num_features)
  # Combine features and response into a data frame
  data_source <- data.frame(X)
  # Naming columns
  colnames(data_source) <- c(paste0("X", 1:num_features))
  
  ##### Unknown (Data ML model is trained on) #####
  # Number of features
  num_features <- p
  # True coefficients
  true_coefficients <- rnorm(num_features)  + beta_0 + beta_diff
  if(normal == F)true_coefficients <- runif(num_features, min = -1, max = 1)  * (beta_0 + beta_diff)
  if (alt == T & normal == T) true_coefficients = true_coefficients*r
  beta_tilde_s = true_coefficients
  if(sparse == T) true_coefficients = true_coefficients * z
  beta_tilde = true_coefficients
  # Generate features from random Gaussian distribution
  X <- matrix(rnorm(N_u * num_features), ncol = num_features)
  # Generate response variable based on the linear model
  y <- X %*% beta_tilde + rnorm(N_u, mean = 0, sd = sigma_s)
  # Combine features and response into a data frame
  data_unknown <- data.frame(cbind(y, X))
  # Naming columns
  colnames(data_unknown) <- c("y", paste0("X", 1:num_features))
  
  ##### Target #####
  num_features <- p
  # True coefficients
  #true_coefficients <- rnorm(num_features)  + beta_0 
  true_coefficients = beta_tilde_s - (rnorm(num_features, sd = 0.1) + beta_diff)
  #if(norm == F)true_coefficients <- runif(num_features, min = -1, max = 1)  * (beta_0)
  beta = true_coefficients
  if(sparse == T)beta = true_coefficients * z
  # Generate features from random Gaussian distribution
  X <- matrix(rnorm(n * num_features), ncol = num_features)
  # Generate response variable based on the linear model
  y <- X %*% beta  + rnorm(n, mean = 0, sd = sigma_t)
  # Combine features and response into a data frame
  data_target <- data.frame(cbind(y, X))
  # Naming columns
  colnames(data_target) <- c("y", paste0("X", 1:num_features))
  
  ##### Test data #####
  # Generate features from random Gaussian distribution
  X <- matrix(rnorm(100 * num_features), ncol = num_features)
  # Generate response variable based on the linear model
  y <- X %*% true_coefficients  + rnorm(100, mean = 0, sd = sigma_t)
  # Combine features and response into a data frame
  data_test <- data.frame(cbind(y, X))
  # Naming columns
  colnames(data_test) <- c("y", paste0("X", 1:num_features))
  
  return (list(
    'data_unknown' = data_unknown,
    
    'data_source' = data_source, 
    'beta_tilde' = beta_tilde, 
    
    'data_target' = data_target, 
    'beta' = beta,
    
    'data_test' = data_test
  ))
  
}

# Function to check errors
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
  y_ppi_train <- X_train %*% beta_hats$mu_ppi 
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


# Generating the errors and running the experiments. 
evaluate_for_seed <- function(seed = 2609, hypers, 
                              alpha_vals = seq(0, 1, length = 10), 
                              k_vals = c(0), 
                              tau_vals = seq(0.01, 0.25, length = 10),
                              n_cores = 6, n_cv = 3, low_dim = T, sparsity = F) {
  # Data Generation
  set.seed(seed)
  Dat <- data_generate(seed = seed, hypers, normal = TRUE, sparse = sparsity)
  data_unknown <- Dat$data_unknown
  data_source <- Dat$data_source
  data_target <- Dat$data_target
  data_test <- Dat$data_test
  
  #If cross validation
  data_train_target = data_target
  beta_diff = Dat$beta_tilde - Dat$beta
  beta_diff %>% hist()
  
  
  # Build ML model
  X_tilde <- as.matrix(data_source)
  ML <- build_ML(data_unknown)$ML_model
  
  X <- as.matrix(data_train_target[,-1])
  y <- as.matrix(data_train_target[,1])
  
  
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
  
  
  ### Retrain on full data ### 
  X = as.matrix(data_target[,-1])
  y = as.matrix(data_target[,1])
  
  Y_tilde_ML = as.matrix(predict(ML, newdata = X_tilde))
  # Generate predictions with fixed hyperparameters
  mu_blast <- mu_generate_blast(X, y, X_tilde, ML,
                              alpha = best_alpha, 
                              k = best_k, 
                              tau = best_tau)$mu_blast
  
  mu_ppi = mu_generate_ppi(X_tilde, X,y,Y_tilde_ML,ML) 
  mu_GLMTrans = mu_generate_GLMTrans(X_tilde, X,y, Y_tilde_ML)
  mu_transLasso = mu_generate_TransLasso(X_tilde, X,y, Y_tilde_ML,size.A0 = 1)
  mu_transLasso0 = mu_generate_TransLasso(X_tilde, X,y, Y_tilde_ML,size.A0 = 0)
  mu_ridge = mu_generate_ridge(X,y, lambda_values = seq(0.000001, 10, length = 100))
  mu_ols = if(low_dim == T) mu_generate_ols(X,y) else NULL
  
  beta_hats <- list(
    'mu_TL' = mu_blast,
    "mu_ppi" = mu_ppi,
    "mu_GLMTrans" = mu_GLMTrans, 
    "mu_TransLasso" = mu_transLasso,
    "mu_TransLasso0" = mu_transLasso0,
    'ridge' = mu_ridge,
    'ols' = mu_ols
  )
  
  # Calculate errors for each method
  errors <- check_errors(data_target, data_test, beta_hats, beta = Dat$beta)
  print(errors)
  return(errors)
}

####################################################################################
##################### TESTING ###########################################

hypers <- list(
  N_u = 1000,
  N = 1000,
  n = 30,
  p = 25,
  sigma_s = 1e-1,
  sigma_t = 1e1,
  beta_0 = 1,
  beta_diff = 0.8
)
#-------#### Testing for one seed #### -------
evaluate_for_seed(seed = 2609, hypers = hypers, low_dim = T, sparsity = F)


# Define the path to the Error_Summaries folder
summaries_folder <- "Experiments_Jan25/Sigma_t10/Error_Summaries"

# List all .rds files in the Error_Summaries folder
files <- list.files(summaries_folder, pattern = "\\.rds$", full.names = TRUE)

# Read all .rds files into a list
all_summaries <- lapply(files, readRDS)

# Optionally, you can name the list elements based on the filenames for easier access
names(all_summaries) <- basename(files)


# Print the contents of the first summary to check
print(all_summaries)


#--------#### Running code for 20 seeds #### -------

#############  Experiment Setup:
###
# In this experiment, we evaluate the model performance for a fixed set of hyperparameters (`hypers`). 
# For reproducibility and comparison, we perform 4 variations of the experiment using different combinations of settings for `alpha`, `k`, and `tau`. 
# Each variation involves running a 'seed loop' with 20 different random seeds, using the `evaluate_for_seed` function. 
# The different variations are as follows:
# 
# 1) **Default Configuration**: Run the experiment with the ALL values of  `alpha`, `k`, and `tau`. 
#    - The expected file name will be: 
#      `"Experiments_Jan25/Error_Summaries/error_summary_beta_diff_1_n_30_p_25_sigmat_1.rds"`
#    
# 2) **Alpha = 1**: Set `alpha = 1`, keeping the other parameters (`k` and `tau`) as they are in the default configuration. 
#    - The expected file name will be:
#      `"Experiments_Jan25/Error_Summaries/error_summary_beta_diff_1_n_30_p_25_sigmat_1_alpha1.rds"`
# 
# 3) **k = 1**: Set `k = 1`, keeping the other parameters (`alpha` and `tau`) as they are in the default configuration. 
#    - The expected file name will be:
#      `"Experiments_Jan25/Error_Summaries/error_summary_beta_diff_1_n_30_p_25_sigmat_1_k1.rds"`
# 
# 4) **tau = 1**: Set `tau = 1`, keeping the other parameters (`alpha` and `k`) as they are in the default configuration. 
#    - The expected file name will be:
#      `"Experiments_Jan25/Error_Summaries/error_summary_beta_diff_1_n_30_p_25_sigmat_1_tau1.rds"`
#
# By running these four variations, we can examine the impact of each parameter (`alpha`, `k`, and `tau`) on the model's performance under different random seeds.


#Define the main folder path
output_folder <- "Experiments_Jan29/sigma_t10"

# Create the main folder if it doesn't exist
if (!dir.exists(output_folder)) {
  dir.create(output_folder)
}

# Create subfolders for errors and summaries
errors_folder <- file.path(output_folder, "Errors")
summaries_folder <- file.path(output_folder, "Error_Summaries")

# Create subfolders if they don't exist
if (!dir.exists(errors_folder)) {
  dir.create(errors_folder)
}
if (!dir.exists(summaries_folder)) {
  dir.create(summaries_folder)
}

hypers <- list(
  N_u = 1000,
  N = 1000,
  n = 30,
  p = 100,
  sigma_s = 1e-1,
  sigma_t = 1e1,
  beta_0 = 1,
  beta_diff = 0.5
)

#evaluate_for_seed(seed = 1, hypers = hypers, low_dim = F, sparsity = T)


# ✅ Automatically generate the identifier from the `hypers` list
identifier <- paste0(
  "beta_diff_", hypers$beta_diff, 
  "_n_", hypers$n, 
  "_p_", hypers$p, 
  "_sigmat_", hypers$sigma_t
)

# Loop through seeds and collect errors
#seeds <- c(2609)
seeds <- c(2601:2620)

all_errors <- list()

for (seed in seeds) {
  print(seed)
  errors <- evaluate_for_seed(seed = seed, hypers = hypers,
                              alpha_vals = seq(0, 10, length = 20),
                              #alpha_vals= c(1),
                              k_vals = 10^(-c(0:9)),
                              #k_vals = c(0),
                              tau_vals = seq(0.1, 15, length = 10),
                              #tau_vals = c(1), 
                              n_cores = 6,
                              n_cv = 3)
  all_errors[[seed]] <- errors
}

# Combine all errors into a single data frame
errors_df <- do.call(rbind, all_errors)

# Calculate summary statistics


library(dplyr)
error_summary <- errors_df %>%
  group_by(Method) %>%
  summarise(
    avg_train_error = mean(Train_error),
    sd_train_error = sd(Train_error),
    avg_test_error = mean(Test_error),
    sd_test_error = sd(Test_error),
    avg_beta_error = mean(beta_error),
    sd_beta_error = sd(beta_error)
  )

# ✅ Save the `errors_df` in the Errors folder
errors_file_path <- file.path(errors_folder, paste0("errors_df_", identifier, ".rds"))
saveRDS(errors_df, file = errors_file_path)
print(paste("I saved file:", errors_file_path))

# ✅ Save the `error_summary` in the Error_Summaries folder
summary_file_path <- file.path(summaries_folder, paste0("error_summary_", identifier, ".rds"))
saveRDS(error_summary, file = summary_file_path)
print(paste("I saved file:", summary_file_path))
print((error_summary)[,1:7])







#--------#### Running code for 20 seeds for high dim n = 30, p = 100 #### -------

#############  Experiment Setup:
###
# In this experiment, we evaluate the model performance for a fixed set of hyperparameters (`hypers`). 
# For reproducibility and comparison, we perform 4 variations of the experiment using different combinations of settings for `alpha`, `k`, and `tau`. 
# Each variation involves running a 'seed loop' with 20 different random seeds, using the `evaluate_for_seed` function. 
# The different variations are as follows:
# 
# 1) **Default Configuration**: Run the experiment with the ALL values of  `alpha`, `k`, and `tau`. 
#    - The expected file name will be: 
#      `"Experiments_Jan25/Error_Summaries/error_summary_beta_diff_1_n_30_p_25_sigmat_1.rds"`
#    
# 2) **Alpha = 1**: Set `alpha = 1`, keeping the other parameters (`k` and `tau`) as they are in the default configuration. 
#    - The expected file name will be:
#      `"Experiments_Jan25/Error_Summaries/error_summary_beta_diff_1_n_30_p_25_sigmat_1_alpha1.rds"`
# 
# 3) **k = 1**: Set `k = 1`, keeping the other parameters (`alpha` and `tau`) as they are in the default configuration. 
#    - The expected file name will be:
#      `"Experiments_Jan25/Error_Summaries/error_summary_beta_diff_1_n_30_p_25_sigmat_1_k1.rds"`
# 
# 4) **tau = 1**: Set `tau = 1`, keeping the other parameters (`alpha` and `k`) as they are in the default configuration. 
#    - The expected file name will be:
#      `"Experiments_Jan25/Error_Summaries/error_summary_beta_diff_1_n_30_p_25_sigmat_1_tau1.rds"`
#
# By running these four variations, we can examine the impact of each parameter (`alpha`, `k`, and `tau`) on the model's performance under different random seeds.


#Define the main folder path
output_folder <- "Experiments_Jan25/high_dim"

# Create the main folder if it doesn't exist
if (!dir.exists(output_folder)) {
  dir.create(output_folder)
}

# Create subfolders for errors and summaries
errors_folder <- file.path(output_folder, "Errors")
summaries_folder <- file.path(output_folder, "Error_Summaries")

# Create subfolders if they don't exist
if (!dir.exists(errors_folder)) {
  dir.create(errors_folder)
}
if (!dir.exists(summaries_folder)) {
  dir.create(summaries_folder)
}


# Define the range of beta_diff values to loop over
beta_diff_values <- c(-0.5, -0.2, -0.1, 0.1, 0.2, 0.5)

# Initialize empty lists to store errors and summaries for each beta_diff value
all_errors_list <- list()
all_summaries_list <- list()

# Loop over the different beta_diff values
for (beta_diff in beta_diff_values) {
  # Set the hypers list with the current beta_diff value
  hypers <- list(
    N_u = 1000,
    N = 1000,
    n = 30,
    p = 100,
    sigma_s = 1e-1,
    sigma_t = 1e1,
    beta_0 = 1,
    beta_diff = beta_diff
  )
  
  # Generate the identifier for this iteration
  identifier <- paste0(
    "beta_diff_", hypers$beta_diff, 
    "_n_", hypers$n, 
    "_p_", hypers$p, 
    "_sigmat_", hypers$sigma_t
  )
  
  # Loop through the seeds and collect errors
  seeds <- c(1:20)
  
  # List to store errors for this beta_diff value
  all_errors <- list()
  
  # Collect errors for each seed
  for (seed in seeds) {
    print(paste("Processing seed", seed, "for beta_diff =", beta_diff))
    errors <- evaluate_for_seed(seed = seed, hypers = hypers,
                                alpha_vals = seq(0, 1, length = 20),
                                k_vals = c(0),
                                tau_vals = seq(0.1, 0.25, length = 10),
                                n_cores = 6,
                                n_cv = 3, low_dim = F, sparsity = T)
    all_errors[[paste0("seed_", seed)]] <- errors
  }
  
  # Combine all errors into a single data frame
  errors_df <- do.call(rbind, all_errors)
  
  # Save errors for this beta_diff value in the list
  all_errors_list[[paste0("beta_diff_", beta_diff)]] <- errors_df
  
  # Calculate summary statistics
  library(dplyr)
  error_summary <- errors_df %>%
    group_by(Method) %>%
    summarise(
      avg_train_error = mean(Train_error),
      sd_train_error = sd(Train_error),
      avg_test_error = mean(Test_error),
      sd_test_error = sd(Test_error),
      avg_beta_error = mean(beta_error),
      sd_beta_error = sd(beta_error)
    )
  print(error_summary)
  
  # Save the error summary for this beta_diff value in the list
  all_summaries_list[[paste0("beta_diff_", beta_diff)]] <- error_summary
  
  # Save the errors_df file
  errors_file_path <- file.path(errors_folder, paste0("errors_df_", identifier, ".rds"))
  saveRDS(errors_df, file = errors_file_path)
  print(paste("I saved file:", errors_file_path))
  
  # Save the error_summary file
  summary_file_path <- file.path(summaries_folder, paste0("error_summary_", identifier, ".rds"))
  saveRDS(error_summary, file = summary_file_path)
  print(paste("I saved file:", summary_file_path))

}


summaries_folder <- "Experiments_Jan29/Sigma_t10/Error_Summaries"

# List all .rds files in the Error_Summaries folder
files <- list.files(summaries_folder, pattern = "\\.rds$", full.names = TRUE)

# Read all .rds files into a list
all_summaries <- lapply(files, readRDS)

# Optionally, you can name the list elements based on the filenames for easier access
names(all_summaries) <- basename(files)


# Print the contents of the first summary to check
print(all_summaries)


# Print the error summary for debugging
print(error_summary[,1:7])

# Now `all_errors_list` and `all_summaries_list` contain the errors and summaries for each `beta_diff`



#### --------------- Going granular for one seed #### --------------
hypers <- list(
  N_u = 10000,
  N = 10000,
  n = 30,
  p = 25,
  sigma_s = 1e-1,
  sigma_t = 1e0,
  beta_0 = 5,
  beta_diff = 2
)

seed = 2603
Dat <- data_generate(seed = seed, hypers, normal = TRUE, sparse = FALSE, alt = FALSE)
data_unknown <- Dat$data_unknown
data_source <- Dat$data_source
data_target <- Dat$data_target
data_test <- Dat$data_test

#If cross validation
data_train_target = data_target
beta_diff = Dat$beta_tilde - Dat$beta
beta_diff %>% hist()


# Build ML model
X_tilde <- as.matrix(data_source)
ML <- build_ML(data_unknown, hypers)$ML_model

X <- as.matrix(data_target[,-1])
y <- as.matrix(data_target[,1])

hypers_training <- cv_loop(hypers, X, y, X_tilde, ML,
                           alpha_vals = seq(0, 10, length = 10),
                           #alpha_vals= c(1),
                           k_vals = 10^(-c(0:9)),
                           #k_vals = c(1),
                           tau_vals = seq(0.1, 15, length = 10),
                           #tau_vals = c(10^6), 
                           n_cores = 6,
                           n_cv = 3)

(hypers_training$best_params)

best_alpha <- hypers_training$best_params$alpha
best_k <- hypers_training$best_params$k
best_tau <- hypers_training$best_params$tau


### Retrain on full data ### 
X = as.matrix(data_target[,-1])
y = as.matrix(data_target[,1])

Y_tilde_ML = as.matrix(predict(ML, newdata = X_tilde))

# Generate predictions with fixed hyperparameters
mu_blast <- mu_generate_blast(hypers, X, y, X_tilde, ML,
                            alpha = best_alpha, 
                            k = best_k, 
                            tau = best_tau)$mu_blast

mu_ppi = mu_generate_ppi(X_tilde, X,y,Y_tilde_ML,ML)
mu_GLMTrans = mu_generate_GLMTrans(X_tilde, X,y, Y_tilde_ML)
mu_transLasso = mu_generate_TransLasso(X_tilde, X,y, Y_tilde_ML,size.A0 = 1)
mu_transLasso0 = mu_generate_TransLasso(X_tilde, X,y, Y_tilde_ML,size.A0 = 0)
mu_ridge = mu_generate_ridge(X,y, lambda_values = seq(0.000001, 10, length = 100))
mu_ols = mu_generate_ols(X,y)

beta_hats <- list(
  'mu_TL' = mu_blast,
  "mu_ppi" = mu_ppi,
  "mu_GLMTrans" = mu_GLMTrans, 
  "mu_TransLasso" = mu_transLasso,
  "mu_TransLasso0" = mu_transLasso0,
  'ridge' = mu_ridge,
  'ols' = mu_ols
)

# Calculate errors for each method
errors <- check_errors(data_target, data_test, beta_hats, beta = Dat$beta)
errors



