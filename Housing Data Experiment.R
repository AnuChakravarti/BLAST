library(data.table)
library(dplyr)

source("Helper Functions.R")
source("Main Functions for BLAST.R")

# Getting the data with demographic information on counties for 2016
countydata = fread("../Data/county_facts.csv")
countydata = countydata  %>% 
  rename(county_name = area_name) 

# Getting the data of housing index of counties for Jan2016
zillow_data = fread("../Data/Zillow_county_data.csv")[,c(1:9,202)] %>% 
  rename(ZHVI_Jan2016 = "2016-01-31" ) %>% 
  filter(!is.na(ZHVI_Jan2016)) %>% 
  mutate(fips = ifelse(MunicipalCodeFIPS>99,
                       paste(StateCodeFIPS,MunicipalCodeFIPS,sep = ""),
                       ifelse(MunicipalCodeFIPS<=9,
                              paste(StateCodeFIPS,"00",MunicipalCodeFIPS,sep = ""),
                              paste(StateCodeFIPS,"0",MunicipalCodeFIPS,sep = "")
                              )
                       )
            %>% as.numeric())  

# Creating the final dataset - the covariates are scaled, the response is in log-scale
merged_data = zillow_data %>%
  inner_join(countydata, by = "fips") %>% 
  dplyr::select(-c(RegionID,SizeRank,RegionName,RegionType,StateName,Metro,
            StateCodeFIPS,MunicipalCodeFIPS,state_abbreviation,fips,county_name)) %>% 
  mutate(#ZHVI_Jan2016 = sqrt(ZHVI_Jan2016),
         across(c(2:ncol(.)), scale) #2:ncol(.) - scales response, 3:ncol(.) - does not scale response
         ) 

##### -------------- Function for housing experiment ---------------############

####Input:
#   unlabelStates - States in the unlabelled dataset e.g. - c("MN","MI,"IN")
#   targetState  - Target states  e.g. c("IL")
#   alpha_vals - values for CV of alpha
#   tau_vals - values for CV of tau
#   train_prop - proportion of target data in the train set, rest used for testing
#   n_cv - number of folds for CV
#   n_cores - number of cores for paralellization
#   seed - the random seed

####Output:
#   data - the unknown_data, unlabelled_data, target_data, train_data, test_data
#   beta_hats - the estimated beta values by each method
#   result - the train error and test error for every method
#   best_params - the optimal parameters for our method chosen by CV

Housing_experiment = function(unlabelStates, targetState,alpha_vals,tau_vals,
                              train_prop = 0.5,n_cv = 5,n_cores = 6,seed = 1){
  set.seed(seed)
  unknown_data = merged_data %>% filter(!State %in% c(unlabelStates,targetState)) %>% 
    dplyr::select(-c(State))
  unlabelled_data = merged_data %>% filter(State %in% unlabelStates)%>% 
    dplyr::select(-c(State))
  target_data = merged_data %>% filter(State %in% targetState)%>% 
    dplyr::select(-c(State))
  
  train_data <- target_data %>% sample_frac(train_prop)  
  test_data <- anti_join(target_data, train_data) 
  
  data = list("unknown_data" = unknown_data,
              "unlabelled_data"=unlabelled_data,
              "target_data"=target_data,
              "train_data"=train_data,
              "test_data"=test_data)
  
  ML <- build_ML(unknown_data)$ML_model
  
  X <- train_data[,-1] %>% as.matrix()
  y <- train_data[,1] %>%  as.matrix()
  
  X_tilde <- as.matrix(unlabelled_data[,-1])
  
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
  X_test <- as.matrix(test_data[,-1])
  y_test <- as.matrix(test_data[,1])
  
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


##### -------------- Experiment ------------------------------------############
unlabelStates = c("MN", "WI", "IN","OH","KS","ND","IA","MO","NE","SD","MI")
targetState = c("IL")

alpha_vals = seq(0, 2, length = 10)
tau_vals = seq(1, 10, length = 50)


final = Housing_experiment(unlabelStates, targetState,alpha_vals,tau_vals,seed = 2609,n_cv = 3)
final$result %>%
  arrange(Test_error)
final$best_params


##### -------------- Experiment over all Midwest States-------------############
#Define the main folder path
output_folder <- "Experiments_House"

# Create the main folder if it doesn't exist
if (!dir.exists(output_folder)) {
  dir.create(output_folder)
}

# Create subfolders for errors and summaries
summaries_folder <- file.path(output_folder, "Log")

# Create subfolders if they don't exist
if (!dir.exists(summaries_folder)) {
  dir.create(summaries_folder)
}

# The Midwest States
States = c("IL","IN","IA","KS","MI","MN","MO","NE","ND","OH","SD","WI")
alpha_vals = seq(0, 50, length = 10)
tau_vals = seq(0.001, 0.05, length = 10)

j = 1
all_errors <- list()
for(i in 1:20){
  unlabelStates = States[-j]
  targetState = States[j]
  
  final = Housing_experiment(unlabelStates, targetState,alpha_vals,tau_vals,seed = i,n_cv = 3)
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


# Saving the error summaries
summary_file_path <- file.path(summaries_folder, paste0("error_summary_", States[j], ".rds"))
saveRDS(error_summary, file = summary_file_path)
print(paste("I saved file:", summary_file_path))


# Reading all the files:
# Define the path to the Error_Summaries folder
summaries_folder <- "Experiments_House/Error_Summaries"

# List all .rds files in the Error_Summaries folder
files <- list.files(summaries_folder, pattern = "\\.rds$", full.names = TRUE)

# Read all .rds files into a list
all_summaries <- lapply(files, readRDS)

lapply(all_summaries, function(df) df %>% arrange(avg_test_error))

# Find overall mean

all_summaries_df <- do.call(rbind, all_summaries)
all_summaries_summary <- all_summaries_df %>%
  group_by(Method) %>%
  summarise(
    train_error = mean(sd_test_error),
    test_error = mean(avg_test_error)
  )

n_unknown = numeric(12)
n_unlabelled = numeric(12)
n_train = numeric(12)
n_test = numeric(12)

for(j in 1:12){
  States = c("IL","IN","IA","KS","MI","MN","MO","NE","ND","OH","SD","WI")
  unlabelStates = States[-j]
  targetState = States[j]
  
  unknown_data = merged_data %>% filter(!State %in% c(unlabelStates,targetState)) %>% 
    dplyr::select(-c(State))
  unlabelled_data = merged_data %>% filter(State %in% unlabelStates)%>% 
    dplyr::select(-c(State))
  target_data = merged_data %>% filter(State %in% targetState)%>% 
    dplyr::select(-c(State))
  
  train_data <- target_data %>% sample_frac(0.5)  
  test_data <- anti_join(target_data, train_data) 
  
  n_unknown[j] = nrow(unknown_data)
  n_unlabelled[j] = nrow(unlabelled_data)
  n_train[j] = nrow(train_data)
  n_test[j] = nrow(train_data)
}

data.frame("Target State" = States, "n_unlabelled" = n_unlabelled,
           "n_train" = n_train, "n_test"= n_test,"n_unknown" = n_unknown)
