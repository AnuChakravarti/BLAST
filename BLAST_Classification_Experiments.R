library(xgboost)
library(dplyr)
library(MASS)
library(pbapply)
library(parallel)

source("Helper Functions.R")
source("Main Functions for BLAST.R")

################. Data Generation ################


data_generate_classification <- function(seed, hypers, normal = TRUE, sparse = FALSE, alt = FALSE) {
  set.seed(seed)
  
  # Extract hyperparameters
  N <- hypers$N
  n <- hypers$n
  N_u <- hypers$N_u
  p <- hypers$p
  sigma_s <- hypers$sigma_s
  sigma_t <- hypers$sigma_t
  beta_0 <- hypers$beta_0
  beta_diff <- hypers$beta_diff
  
  # Sparsity and alternating coefficients
  z <- rbinom(p, size = 1, prob = 0.25)  # For sparsity
  r <- 2 * (rbinom(p, size = 1, prob = 0.5) - 0.5)  # Alternating signs
  
  ##### SOURCE DATA #####
  X_source <- matrix(rnorm(N * p), ncol = p)
  data_source <- data.frame(X_source)
  colnames(data_source) <- paste0("X", 1:p)
  
  ##### UNKNOWN DATA #####
  true_coefficients <- rnorm(p) + beta_0 + beta_diff
  if (!normal) true_coefficients <- runif(p, min = -1, max = 1) * (beta_0 + beta_diff)
  if (alt && normal) true_coefficients <- true_coefficients * r
  beta_tilde_s <- true_coefficients
  if (sparse) true_coefficients <- true_coefficients * z
  beta_tilde <- true_coefficients
  
  X_unknown <- matrix(rnorm(N_u * p), ncol = p)
  y_cont <- X_unknown %*% beta_tilde
  probabilities <- plogis(y_cont)
  y <- rbinom(N_u, size = 1, prob = probabilities)
  
  data_unknown <- data.frame(y, X_unknown)
  colnames(data_unknown) <- c("y", paste0("X", 1:p))
  
  ##### TARGET DATA #####
  true_coefficients <- beta_tilde_s - (rnorm(p, sd = 0.1) + beta_diff)
  if (sparse) true_coefficients <- true_coefficients * z
  beta <- true_coefficients
  
  X_target <- matrix(rnorm(n * p), ncol = p)
  y_cont <- X_target %*% beta
  probabilities <- plogis(y_cont)
  y <- rbinom(n, size = 1, prob = probabilities)  # Fixed N_u -> n
  
  data_target <- data.frame(y, X_target)
  colnames(data_target) <- c("y", paste0("X", 1:p))
  
  ##### TEST DATA #####
  X_test <- matrix(rnorm(100 * p), ncol = p)
  y_cont <- X_test %*% beta  # Explicitly use `beta`
  probabilities <- plogis(y_cont)
  y <- rbinom(100, size = 1, prob = probabilities)  # Fixed N_u -> 100
  
  data_test <- data.frame(y, X_test)
  colnames(data_test) <- c("y", paste0("X", 1:p))
  
  return(list(
    'data_unknown' = data_unknown,
    'data_source' = data_source,
    'beta_tilde' = beta_tilde,
    'data_target' = data_target,
    'beta' = beta,
    'data_test' = data_test
  ))
}


hypers <- list(
  N_u = 1000,
  N = 1000,
  n = 30,
  p = 25,
  sigma_s = 1e-1,
  sigma_t = 1e0,
  beta_0 = 1,
  beta_diff = -0.2
)

set.seed(1)
seed = 1
Dat = data_generate_classification(seed, hypers)



## -------- Applying Methods -----
data_train = Dat$data_target
X = Dat$data_target[,-1] %>%  as.matrix()
y = Dat$data_target[,1] %>%  as.matrix()
X_tilde = Dat$data_source%>%  as.matrix()

data_unknown = Dat$data_unknown

ML <- build_ML(data_unknown)$ML_model  # XGBoost model
# Predict Y_tilde_ML for X_tilde
Y_tilde_ML <- ifelse(predict(ML, newdata = X_tilde) > 0.5, 1, 0)

data_test = Dat$data_test
X_test = Dat$data_test[,-1] %>%  as.matrix()  # Excluding the target variable column
y_test = Dat$data_test[,1] %>%  as.matrix()   # Extracting the target variable




#### ---- blastVar ----


alpha_vals = seq(2, 5, length = 5)
k_vals = c(0)
tau_vals = seq(2, 5, length = 5)

hypers_training <- cv_loop_class(X, y, X_tilde, ML,
                                 alpha_vals = alpha_vals,
                                 k_vals = k_vals,
                                 tau_vals = tau_vals,n_cv = 3,n_cores = 6)

best_alpha <- hypers_training$best_params$alpha
best_k <- hypers_training$best_params$k
best_tau <- hypers_training$best_params$tau
print(hypers_training$best_params)


mu_blastvar <- mu_generate_blast_class( X, y, X_tilde, ML,
                                   alpha = best_alpha, 
                                   k = best_k, 
                                   tau = best_tau)$mu_blast



##### ----- blast ---- 

alpha_vals = seq(0, 5, length = 20)
k_vals = c(0)
tau_vals = seq(0.001, 0.01, length = 20)

hypers_training <- cv_loop(X, y, X_tilde, ML,
                           alpha_vals = alpha_vals,
                           k_vals = k_vals,
                           tau_vals = tau_vals, 
                           n_cores = 6, 
                           n_cv = 3)

best_alpha <- hypers_training$best_params$alpha
best_k <- hypers_training$best_params$k
best_tau <- hypers_training$best_params$tau
print(hypers_training$best_params)


mu_blast <- mu_generate_blast( X, y, X_tilde, ML,
                             alpha = best_alpha, 
                             k = best_k, 
                             tau = best_tau)$mu_blast
##### ----- Others ---- 

mu_logistic = mu_generate_logistic(X,y)
mu_ridge = mu_generate_ridge(X,y, family = 'binomial')
mu_GLMTrans = mu_generate_GLMTrans(X_tilde, X,y, Y_tilde_ML, family = 'binomial')
beta_hats <- list(
  'mu_TL' = mu_blast %>%  as.matrix(),
  "mu_TL_var" = mu_blastvar  %>%  as.matrix(),
  #"mu_ppi" = mu_ppi_sparse,
  "mu_GLMTrans" = mu_GLMTrans  %>%  as.matrix(), 
  #"mu_TransLasso" = mu_transLasso,
  #"mu_TransLasso0" = mu_transLasso0,
  'ridge' = mu_ridge  %>%  as.matrix(),
  'logistic' = mu_logistic  %>%  as.matrix()
)

## -------- Comparison -----
check_errors_classification(Dat$data_target, data_test, beta_hats,thresh = 0.5)



########### Running for 20 seeds #################

# Define the different beta_diff values to loop over
beta_diff_values <- c(-0.5, -0.2, -0.1, 0.1, 0.2, 0.5)

# Create a main folder for classification results
main_folder <- "classification_results"
dir.create(main_folder, showWarnings = FALSE)

# Loop over beta_diff values
for (beta_diff in beta_diff_values) {
  
  # Update hypers with the current beta_diff
  hypers$beta_diff <- beta_diff
  
  # Create a folder for the current beta_diff inside the main folder
  folder_name <- paste0(main_folder, "/results_classification_n_", hypers$n, "p_", hypers$p, "sigmat_", hypers$sigma_t, "beta_", hypers$beta_0, "betadiff_", beta_diff)
  dir.create(folder_name, showWarnings = FALSE)
  
  # Create subfolders for errors and summaries inside the current beta_diff folder
  error_folder <- paste0(folder_name, "/errors")
  summary_folder <- paste0(folder_name, "/summaries")
  dir.create(error_folder, showWarnings = FALSE)
  dir.create(summary_folder, showWarnings = FALSE)
  
  # Initialize a list to store errors
  error_list <- list()
  
  # Loop over seeds
  seeds <- 1:20  # Adjust the number of seeds as necessary
  for (seed in seeds) {
    print(seed)
    # Set seed for reproducibility
    set.seed(seed)
    Dat = data_generate_classification(seed = seed, hypers)
  
    
    # Extract relevant data
    X = Dat$data_target[,-1] %>% as.matrix()
    y = Dat$data_target[,1] %>% as.matrix()
    X_tilde = Dat$data_source %>% as.matrix()
    
    data_unknown = Dat$data_unknown
    
    ML <- build_ML(data_unknown)$ML_model  # XGBoost model
    Y_tilde_ML <- ifelse(predict(ML, newdata = X_tilde) > 0.5, 1, 0)
    
    data_test = Dat$data_test
    X_test = Dat$data_test[,-1] %>% as.matrix()  # Excluding the target variable column
    y_test = Dat$data_test[,1] %>% as.matrix()   # Extracting the target variable
    
    
    # blastvar 
    
    alpha_vals = seq(2, 5, length = 5)
    k_vals = c(0)
    tau_vals = seq(2, 5, length = 5)
    
    hypers_training <- cv_loop_class(X, y, X_tilde, ML,
                                     alpha_vals = alpha_vals,
                                     k_vals = k_vals,
                                     tau_vals = tau_vals,
                                     n_cv = 3,n_cores = 6)
    
    best_alpha <- hypers_training$best_params$alpha
    best_k <- hypers_training$best_params$k
    best_tau <- hypers_training$best_params$tau
    print(hypers_training$best_params)
    
    
    mu_blastvar <- mu_generate_blast_class( X, y, X_tilde, ML,
                                          alpha = best_alpha, 
                                          k = best_k, 
                                          tau = best_tau)$mu_blast
    
    
    
    # Perform blast method
    # alpha_vals = seq(0, 5, length = 20)
    # k_vals = c(0)
    # tau_vals = seq(0.001, 0.01, length = 10)
    # 
    # hypers_training <- cv_loop(X, y, X_tilde, ML,
    #                            alpha_vals = alpha_vals,
    #                            k_vals = k_vals,
    #                            tau_vals = tau_vals, 
    #                            n_cores = 6, 
    #                            n_cv = 3)
    # 
    # best_alpha <- hypers_training$best_params$alpha
    # best_k <- hypers_training$best_params$k
    # best_tau <- hypers_training$best_params$tau
    # 
    # print(hypers_training$best_params)
    # 
    # mu_blast <- mu_generate_blast(X, y, X_tilde, ML,
    #                             alpha = best_alpha, 
    #                             k = best_k, 
    #                             tau = best_tau)$mu_blast
    
    # Other methods
    mu_logistic = mu_generate_logistic(X, y)
    mu_ridge = mu_generate_ridge(X, y, family = 'binomial')
    mu_GLMTrans = mu_generate_GLMTrans(X_tilde, X, y, Y_tilde_ML, family = 'binomial')
    
    beta_hats <- list(
      #'mu_TL' = mu_blast %>%  as.matrix(),
      "mu_TL_var" = mu_blastvar  %>%  as.matrix(),
      #"mu_ppi" = mu_ppi_sparse,
      "mu_GLMTrans" = mu_GLMTrans  %>%  as.matrix(), 
      #"mu_TransLasso" = mu_transLasso,
      #"mu_TransLasso0" = mu_transLasso0,
      'ridge' = mu_ridge  %>%  as.matrix(),
      'logistic' = mu_logistic  %>%  as.matrix()
    )
    
    # Compute errors
    errors <- check_errors_classification(Dat$data_target, data_test, beta_hats,thresh = "auto")
    print(errors)
    
    # Save errors for this seed
    error_list[[paste0("seed_", seed)]] <- errors
    saveRDS(errors, file = paste0(error_folder, "/errors_seed_", seed, ".rds"))
  }
  
  # Print error summaries for the current beta_diff
  summary_results <- do.call(rbind, error_list) %>%
    group_by(Method) %>%
    summarise(
      avg_train_mcc = mean(Train_MCC),
      sd_train_mcc = sd(Train_MCC),
      avg_test_mcc = mean(Test_MCC),
      sd_test_mcc = sd(Test_MCC),
      avg_train_AUC = mean(AUC_Train),
      sd_train_AUC = sd(AUC_Train),
      avg_test_AUC = mean(AUC_Test),
      sd_test_AUC = sd(AUC_Test),
      avg_thresh = mean(Thresholds)
    )
  
  print(summary_results)
  
  # Save the error summaries as an RDS file
  saveRDS(summary_results, file = paste0(summary_folder, "/error_summary.rds"))
}






##### ---- Viewing all Result  Summaries  ---- 

# Define the different beta_diff values
beta_diff_values <- c(-0.5, -0.2, -0.1, 0.1, 0.2, 0.5)

# Define the main folder where the results are saved
main_folder <- "classification_results"

# Loop over beta_diff values
for (beta_diff in beta_diff_values) {
  
  # Path to the summary file for the current beta_diff value
  summary_file <- paste0(main_folder, "/results_classification_n_", hypers$n, "p_", hypers$p, "sigmat_", hypers$sigma_t, "beta_", hypers$beta_0, "betadiff_", beta_diff, "/summaries/error_summary.rds")
  
  # Load the summary data
  summary_results <- readRDS(summary_file)
  
  # Print the summary
  print(paste("Summary for beta_diff =", beta_diff))
  print(summary_results)
}
