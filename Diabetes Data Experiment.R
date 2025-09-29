library(data.table)
library(dplyr)
library(caret)
library(data.table)
library(caret)
library(pROC)  # For AUC calculation

# Load helper functions
source("Main Functions for BLAST.R")
source('Helper Functions.R')


### ---- Feature Engineering --- #### 
# Load data and remove unnecessary rows/columns
dat <- fread('../Data/widsdatathon2021_Diabetes/TrainingWiDS2021.csv')[, -1]
dat <- dat[ethnicity != "Other/Unknown"]

# Drop columns with only one unique value
dat[, (names(which(sapply(dat, function(x) length(unique(x)) == 1)))) := NULL]

# Remove duplicate columns
duplicate_cols <- function(df) {
  col_names <- names(df)
  duplicate_pairs <- unlist(lapply(seq_along(col_names[-length(col_names)]), function(i) {
    sapply((i + 1):length(col_names), function(j) {
      if (identical(df[[col_names[i]]], df[[col_names[j]]])) col_names[j]
    })
  }))
  unique(duplicate_pairs[!is.na(duplicate_pairs)])
}
dat[, (duplicate_cols(dat)) := NULL]

# Remove unwanted rows
dat <- dat[age != 0 & !is.na(gender) & !is.na(weight) & !is.na(height)]

# Drop columns with >50% missing values
dat[, (names(which(sapply(dat, function(x) sum(is.na(x))) > 0.5 * nrow(dat)))) := NULL]

# Standardize hospital admit source
dat[, hospital_admit_source := fifelse(hospital_admit_source %in% c("Other ICU", "ICU to SDU", "Step-Down Unit (SDU)"), "SDU", 
                                       fifelse(hospital_admit_source %in% c("Other Hospital", "Observation"), "Recovery Room", hospital_admit_source))]
dat <- dat[hospital_admit_source != "Other"]

# Compute comorbidity_score and total_chronic
dat[, comorbidity_score := (aids * 23) + (cirrhosis * 4) + (hepatic_failure * 16) + (immunosuppression * 10) + 
      (leukemia * 10) + (lymphoma * 13) + (solid_tumor_with_metastasis * 11)]
dat[is.na(comorbidity_score), comorbidity_score := 0]
dat[, total_chronic := rowSums(.SD, na.rm = TRUE), .SDcols = c("aids", "cirrhosis", "hepatic_failure")]
dat[, c("aids", "cirrhosis", "hepatic_failure", "immunosuppression", "leukemia", "lymphoma", "solid_tumor_with_metastasis") := NULL]

# Fill missing values for GCS components and compute GCS Score
dat[, gcs_eyes_apache := fifelse(is.na(gcs_eyes_apache), 4, gcs_eyes_apache)]
dat[, gcs_verbal_apache := fifelse(is.na(gcs_verbal_apache), 5, gcs_verbal_apache)]
dat[, gcs_motor_apache := fifelse(is.na(gcs_motor_apache), 6, gcs_motor_apache)]
dat[, gcs_score := (2.5 * round((gcs_eyes_apache + gcs_motor_apache + gcs_verbal_apache) / 2.5)) / 2.5]
dat[, c("gcs_eyes_apache", "gcs_motor_apache", "gcs_unable_apache", "gcs_verbal_apache") := NULL]

# Handle diagnosis columns
dat[, apache_2_diagnosis := fifelse(is.na(apache_2_diagnosis), 0, as.integer(apache_2_diagnosis))]
dat[, apache_3j_diagnosis := fifelse(is.na(apache_3j_diagnosis), 0, as.integer(apache_3j_diagnosis))]
dat[, APACHE_diabetic_ketoacidosis := fifelse(apache_3j_diagnosis == 702, 1, 0)]

dat[, pregnancy_probability := fifelse(gender == 0, 0, fifelse(apache_3j_diagnosis == 1802, 1, 
  fcase(age < 20, 0.07, age < 25, 0.15, age < 30, 0.16, age < 35, 0.14, age < 40, 0.08, age < 45, 0.02, default = 0)))]

# Fill missing values from corresponding *_apache columns
dat[, `:=`(d1_hematocrit_min = fifelse(is.na(d1_hematocrit_min) & !is.na(hematocrit_apache), hematocrit_apache, d1_hematocrit_min),
           d1_temp_min = fifelse(is.na(d1_temp_min) & !is.na(temp_apache), temp_apache, d1_temp_min),
           d1_sodium_min = fifelse(is.na(d1_sodium_min) & !is.na(sodium_apache), sodium_apache, d1_sodium_min),
           d1_bun_max = fifelse(is.na(d1_bun_max) & !is.na(bun_apache), bun_apache, d1_bun_max))]
dat[, c("hematocrit_apache", "temp_apache", "sodium_apache", "bun_apache") := NULL]

# Loop through all columns in the dataset
for (col in names(dat)) {
  if (is.numeric(dat[[col]])) {
    dat[, (col) := fifelse(is.na(get(col)), median(get(col), na.rm = TRUE), get(col))]
  } else if (is.character(dat[[col]]) || is.factor(dat[[col]])) {
    most_frequent_value <- names(sort(table(dat[[col]]), decreasing = TRUE))[1]
    dat[, (col) := fifelse(is.na(get(col)), most_frequent_value, get(col))]
  }
}


sapply(dat, function(x) round(sum(is.na(x)) / length(x) * 100, 2))
sapply(dat, class)
lapply(dat, function(x) head(table(x), 10))


# Correct BMI discrepancies
dat[, bmi := fifelse(abs(bmi - (weight / (height/100)^2)) > 1, weight / (height/100)^2, bmi)]

# Remove apache diagnosis columns
dat[, (grep("apache.*diagnosis", names(dat), value = TRUE)) := NULL]


# Load the data dictionary
dat_dict <- fread('../Data/widsdatathon2021_Diabetes/DataDictionaryWiDS2021.csv')

# Create a mapping of variable names to their data types
data_types <- setNames(dat_dict$`Data Type`, dat_dict$`Variable Name`)

# Loop through all columns in dat and ensure correct data types
for (col in names(dat)) {
  if (col %in% names(data_types)) {
    if (data_types[[col]] %in% c("numeric", 'integer') ) {
      dat[, (col) := as.numeric(get(col))]  # Convert to numeric
    } else if (data_types[[col]] %in% c("binary", 'string')) {
      dat[, (col) := as.character(get(col))]  # Convert binary to character
    }
  }
}

dat$bmi = as.numeric(dat$bmi)

y = dat$diabetes_mellitus
dat[, diabetes_mellitus := NULL]

dat = cbind(y, dat)


# Print final dataset summary
sapply(dat, function(x) round(sum(is.na(x)) / length(x) * 100, 2))
sapply(dat, class)
lapply(dat, function(x) head(table(x), 10))
colnames(dat)[sapply(dat, class) == 'character']

fwrite(dat, '../Data/widsdatathon2021_Diabetes/cleaned_TrainingWiDS2021.csv', row.names = F)


### ---- Modeling --- #### 

dat = fread('../Data/widsdatathon2021_Diabetes/cleaned_TrainingWiDS2021.csv')
colnames(dat)[sapply(dat, class) == 'character']
dat[, c('hospital_id', 'gender','encounter_id', 'icu_id' ,"hospital_admit_source", "icu_admit_source", "icu_stay_type", "icu_type") := NULL]

# # One-hot encode gender with a single binary column
# dat[, gender_Male := as.integer(gender == "Male")]
# 
# # Remove original gender column
# dat[, gender := NULL]


colnames(dat)


dat_white = dat[dat$ethnicity == 'Caucasian',]
dat_white[, c('ethnicity') := NULL]

colnames(dat_white)[sapply(dat_white, class) == 'character']

############ ----  ML model on White ---- #########

# Split data into training (80%) and test (20%)
set.seed(42)
train_index <- createDataPartition(dat_white$y, p = 0.8, list = FALSE)
train_data <- dat_white[train_index, ]
test_data <- dat_white[-train_index, ]

# Train ML model on training data
ML_model <- build_ML(train_data, task = "auto")$ML_model

# Convert training and test data into gboost matrix format
dtrain <- xgb.DMatrix(data = as.matrix(train_data[, -c("y"), with = FALSE]), 
                      label = as.numeric(train_data$y))

dtest <- xgb.DMatrix(data = as.matrix(test_data[, -c("y"), with = FALSE]), 
                     label = as.numeric(test_data$y))

# Get predicted probabilities for both train and test sets
train_preds <- predict(ML_model, dtrain)
test_preds <- predict(ML_model, dtest)

# Compute AUC for training and test sets
train_auc <- auc(roc(train_data$y, train_preds))
test_auc <- auc(roc(test_data$y, test_preds))

# Print AUC values
cat("Train AUC:", round(train_auc, 4), "\n")
cat("Test AUC:", round(test_auc, 4), "\n")




############ ----  Preppring rest of the data  ---- #########

set.seed(1)
target_ethnicity <- "Asian"
eth = target_ethnicity
source_ethnicity <- "Caucasian"

# Filter for the target ethnicity and remove the ethnicity column
data_target <- dat[ethnicity == target_ethnicity]
data_target[, ethnicity := NULL]

# Step 1: Split target data into train (5%) and the rest (95%)
train_index <- createDataPartition(data_target$y, p = 0.05, list = FALSE)
data_train <- data_target[train_index]
data_rest <- data_target[-train_index]

# Step 2: Split rest into unlabeled (45%) and test (50%) of original
# 45% of total = (45 / 95) ≈ 0.4737 of the remaining 95%
unlabeled_index <- createDataPartition(data_rest$y, p = 0.4737, list = FALSE)
data_unlabeled <- data_rest[unlabeled_index]
data_test <- data_rest[-unlabeled_index]

# Step 3: Prepare matrices for training and test
X <- as.matrix(data_train[, -1, with = FALSE])     # Features for training
y <- as.matrix(data_train[, 1, with = FALSE])      # Target for training

X_test <- as.matrix(data_test[, -1, with = FALSE]) # Features for test
y_test <- as.matrix(data_test[, 1, with = FALSE])  # Target for test

# Step 4: Prepare X_tilde from unlabeled data and generate pseudo-labels
X_tilde <- as.matrix(data_unlabeled[, !'y', with = FALSE])  # Drop target
X_tilde_dmatrix <- xgb.DMatrix(data = X_tilde)
Y_tilde_ML <- ifelse(predict(ML_model, X_tilde_dmatrix) > 0.25, 1, 0)

# #### ---- blastVar ----

# 
# alpha_vals = seq(0, 1, length = 5)
# k_vals = c(0)
# tau_vals = seq(0.1, 1, length = 5)
# 
# hypers_training <- cv_loop_class(X, y, X_tilde, ML_model,
#                                  alpha_vals = alpha_vals,
#                                  k_vals = k_vals,
#                                  tau_vals = tau_vals, n_cv = 2, n_cores = 8)
# 
# best_alpha <- hypers_training$best_params$alpha
# best_k <- hypers_training$best_params$k
# best_tau <- hypers_training$best_params$tau
# print(hypers_training$best_params)
# 


##### ----- Others ---- 
mu_logistic = mu_generate_logistic(X,y)
mu_ridge = mu_generate_ridge(X,y, family = 'binomial')
mu_GLMTrans = mu_generate_GLMTrans(X_tilde, X,y, Y_tilde_ML, family = 'binomial')


best_alpha = 100
best_tau = 1000
best_k =  0
mu_blastvar <- posterior_param_class( X, y, X_tilde, Y_tilde_ML,
                                     alpha = best_alpha, 
                                     k = best_k, 
                                     tau = best_tau)$mu_blast

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


# 1. Run classification evaluation
results_df <- check_errors_classification(data_target, data_test, beta_hats)
results_df


# 2. Compute ML AUCs
X_train_dmatrix <- xgb.DMatrix(data = as.matrix(X))
X_test_dmatrix <- xgb.DMatrix(data = as.matrix(data_test[, -1]))

train_probs_ML <- predict(ML_model, X_train_dmatrix)
test_probs_ML <- predict(ML_model, X_test_dmatrix)

auc_train_ML <- pROC::auc(data_train$y, train_probs_ML)
auc_test_ML <- pROC::auc(data_test$y, test_probs_ML)

# 3. Construct ML row — note MCC and Thresholds are NA
ml_row <- data.frame(
  Method = "ML",
  Train_MCC = NA,
  Test_MCC = NA,
  AUC_Train = as.numeric(round(auc_train_ML, 4)),
  AUC_Test = as.numeric(round(auc_test_ML, 4)),
  Thresholds = NA
)

# 4. Append to results
results_combined <- rbind(results_df[,c(1,2,3)], ml_row[,c(1,4,5)])

# 5. Print or use
print(results_combined)



############ ----  Running for 20 seeds for all ethnicities ---- ############ 

# Define ethnicity groups
ethnicities <- c("African American",  "Hispanic", "Asian", "Native American" )
source_ethnicity <- "Caucasian"

# Create main folder for storing results
main_folder <- "widsdatathon2021_Diabetes/error_results"
dir.create(main_folder, showWarnings = FALSE)

# Loop over each target ethnicity
for (target_ethnicity in ethnicities) {
  cat("\n------------------------------------------------------\n")
  cat("Processing Target Ethnicity:", target_ethnicity, "\n")
  
  # Create a directory for the current ethnicity
  ethnicity_folder <- paste0(main_folder, "/", gsub(" ", "_", target_ethnicity))
  dir.create(ethnicity_folder, showWarnings = FALSE)
  
  # Initialize storage for test MCC and AUC results across 20 seeds
  test_mcc_values <- numeric(20)
  test_auc_values <- numeric(20)
  errors_list <- list()
  
  # Loop over 20 different seeds
  for (seed in 1:2) {
    set.seed(seed)
    cat("------------------------------------------------------\n")
    cat("Running for Seed:", seed, " - Ethnicity:", target_ethnicity, "\n")
    
    # Filter for the target ethnicity and remove the ethnicity column
    data_target <- dat[ethnicity == target_ethnicity]
    data_target[, ethnicity := NULL]
    
    # Step 1: Split target data into train (5%) and the rest (95%)
    train_index <- createDataPartition(data_target$y, p = 0.05, list = FALSE)
    data_train <- data_target[train_index]
    data_rest <- data_target[-train_index]
    
    # Step 2: Split rest into unlabeled (45%) and test (50%) of original
    # 45% of total = (45 / 95) ≈ 0.4737 of the remaining 95%
    unlabeled_index <- createDataPartition(data_rest$y, p = 0.4737, list = FALSE)
    data_unlabeled <- data_rest[unlabeled_index]
    data_test <- data_rest[-unlabeled_index]
    
    # Step 3: Prepare matrices for training and test
    X <- as.matrix(data_train[, -1, with = FALSE])     # Features for training
    y <- as.matrix(data_train[, 1, with = FALSE])      # Target for training
    
    X_test <- as.matrix(data_test[, -1, with = FALSE]) # Features for test
    y_test <- as.matrix(data_test[, 1, with = FALSE])  # Target for test
    
    # Step 4: Prepare X_tilde from unlabeled data and generate pseudo-labels
    X_tilde <- as.matrix(data_unlabeled[, !'y', with = FALSE])  # Drop target
    X_tilde_dmatrix <- xgb.DMatrix(data = X_tilde)
    Y_tilde_ML <- ifelse(predict(ML_model, X_tilde_dmatrix) > 0.25, 1, 0)
    
    # Generate model parameters
    cat("Generating model parameters...\n")
    mu_logistic <- mu_generate_logistic(X, y)
    mu_ridge <- mu_generate_ridge(X, y, family = 'binomial')
    mu_GLMTrans <- mu_generate_GLMTrans(X_tilde, X, y, Y_tilde_ML, family = 'binomial')
    
    # Use fixed hyperparameters for blastVar
    best_alpha <- 100
    best_tau <- 1000
    best_k <- 0
    mu_blastvar <- posterior_param_class(X, y, X_tilde, Y_tilde_ML, 
                                        alpha = best_alpha, k = best_k, tau = best_tau)$mu_blast
    
    # Store model predictions
    beta_hats <- list(
      "mu_TL_var" = replace(as.matrix(mu_blastvar), is.na(mu_blastvar), 0),
      "mu_GLMTrans" = replace(as.matrix(mu_GLMTrans), is.na(mu_GLMTrans), 0),
      "ridge" = replace(as.matrix(mu_ridge), is.na(mu_ridge), 0),
      "logistic" = replace(as.matrix(mu_logistic), is.na(mu_logistic), 0)
    )
    
    # Compute error metrics (MCC & AUC)
    cat("Computing error metrics...\n")
    errors <- check_errors_classification(data_target, data_test, beta_hats, thresh = 'auto')
    
    # Store errors in list
    errors_list[[seed]] <- errors
    print(errors)
    
    # Save individual error results as .rds inside the respective folder
    error_filename <- paste0(ethnicity_folder, "/errors_seed_", seed, ".rds")
    saveRDS(errors, file = error_filename)
  }
  
  # Create data frame for error summary
  errors_df <- do.call(rbind, lapply(errors_list, as.data.frame))
  
  # Compute summary statistics for MCC & AUC
  error_summary <- errors_df %>%
    group_by(Method) %>%
    summarise(
      avg_Test_MCC = mean(Test_MCC, na.rm = TRUE),
      sd_Test_MCC = sd(Test_MCC, na.rm = TRUE),
      avg_AUC_Test = mean(AUC_Test, na.rm = TRUE),
      sd_AUC_Test = sd(AUC_Test, na.rm = TRUE)
    )
  
  # Save summary statistics as .rds inside the respective folder
  summary_filename <- paste0(ethnicity_folder, "/summary_errors.rds")
  summary_content <- list(
    Target_Ethnicity = target_ethnicity,
    error_summary = error_summary
  )
  saveRDS(summary_content, file = summary_filename)
  
  # Print final summary for this ethnicity
  cat("\n------------------------------------------------------\n")
  cat("Final Summary for Target Ethnicity:", target_ethnicity, "\n")
  print(error_summary)
  cat("Results saved in:", ethnicity_folder, "\n")
  cat("------------------------------------------------------\n")
}

############  ------- Summary Check   ------- ############ 

library(data.table)

# Define the main directory
main_folder <- "widsdatathon2021_Diabetes/error_results"

# Get a list of all ethnicity folders
ethnicity_folders <- list.dirs(main_folder, recursive = FALSE)

# Initialize an empty list to store summaries
summary_list <- list()

# Loop through each ethnicity folder and load the summary file
for (ethnicity_folder in ethnicity_folders) {
  summary_path <- file.path(ethnicity_folder, "summary_errors.rds")
  
  if (file.exists(summary_path)) {
    summary_data <- readRDS(summary_path)
    #summary_data$Target_Ethnicity <- basename(ethnicity_folder)  # Add ethnicity as a column
    summary_list[[basename(ethnicity_folder)]] <- summary_data
  } else {
    cat("Warning: No summary found in", ethnicity_folder, "\n")
  }
}


summary_list = lapply(summary_list, '[', 2)
print(summary_list)


############  ------- Presentation  ------- ############ 

library(ggplot2)
library(data.table)

# Define the main directory
main_folder <- "widsdatathon2021_Diabetes/error_results"

# Get a list of all ethnicity folders
ethnicity_folders <- list.dirs(main_folder, recursive = FALSE)

# Initialize an empty list to store errors
errors_list <- list()

# Loop through each ethnicity and load all error results (20 seeds)
for (ethnicity_folder in ethnicity_folders) {
  ethnicity_name <- basename(ethnicity_folder)  # Extract ethnicity name
  
  # Loop through all 20 seeds
  for (seed in 1:20) {
    error_path <- file.path(ethnicity_folder, paste0("errors_seed_", seed, ".rds"))
    
    if (file.exists(error_path)) {
      error_data <- readRDS(error_path)
      error_data$Target_Ethnicity <- ethnicity_name  # Add ethnicity column
      error_data$Seed <- seed  # Add seed column
      errors_list[[paste(ethnicity_name, seed)]] <- error_data
    } else {
      cat("Warning: No error file found for", ethnicity_name, "Seed", seed, "\n")
    }
  }
}

# Combine all errors into one data table
all_errors <- rbindlist(errors_list, fill = TRUE)

# Ensure required columns exist
required_columns <- c("Target_Ethnicity", "Seed", "AUC_Test", "Method")
if (all(required_columns %in% names(all_errors))) {
  
  # Create the boxplot with enhanced formatting
  p <- ggplot(all_errors, aes(x = Method, y = AUC_Test, fill = Method)) +  # FIXED COLUMN NAME
    geom_boxplot(outlier.shape = NA, alpha = 0.7) +  # Boxplot without outlier points
    #geom_jitter(width = 0.2, alpha = 0.3, size = 1) +  # Add jittered points
    facet_wrap(~ Target_Ethnicity, scales = "free_y", ncol = 2) +  # Create facets
    labs(title = "AUC Test Distribution Across Methods and Target Ethnicities",
         x = "Method",
         y = "AUC Test") +
    theme_minimal(base_size = 14) +  # Increase base font size
    theme(
      text = element_text(face = "bold"),  # Make all text bold
      axis.text.x = element_text(angle = 45, hjust = 1, size = 14),  # Rotate x-axis labels
      axis.text.y = element_text(size = 14),
      axis.title = element_text(size = 16),
      strip.text = element_text(size = 16),  # Bigger facet labels
      plot.title = element_text(size = 18, hjust = 0.5),  # Center and enlarge title
      legend.text = element_text(size = 14),  # Bigger legend text
      legend.title = element_text(size = 16),  
      panel.spacing = unit(1.5, "lines"),  # Increase space between subplots
      panel.border = element_rect(color = "black", fill = NA, size = 1.5)  # Add border to subplots
    ) +
    ylim(0.4, 1)  # Set y-axis limits
  
  # Print the plot
  print(p)
  
} else {
  cat("Error: The dataset is missing required columns.\n")
}







############ ----  Running for 20 seeds for all ethnicities 04/02 ---- ############ 
# Updated function with PR-AUC and suppressed messages

check_errors_classification <- function(data_train, data_test, beta_hats, compute_mcc = FALSE) {
  library(pROC)
  library(PRROC)
  
  # Extract matrices
  X_train <- as.matrix(data_train[, -1])
  y_train <- data_train$y
  X_test  <- as.matrix(data_test[, -1])
  y_test  <- data_test$y
  
  # Predict probabilities for each model
  pred_probs_train <- lapply(beta_hats, function(b) plogis(X_train %*% b))
  pred_probs_test  <- lapply(beta_hats, function(b) plogis(X_test %*% b))
  
  # Compute AUCs
  auc_train <- sapply(pred_probs_train, function(p) as.numeric(pROC::auc(y_train, p)))
  auc_test  <- sapply(pred_probs_test, function(p) as.numeric(pROC::auc(y_test, p)))
  
  # Compute PR-AUCs
  pr_train <- sapply(pred_probs_train, function(p) pr.curve(scores.class0 = p[y_train == 1], 
                                                            scores.class1 = p[y_train == 0], 
                                                            curve = FALSE)$auc.integral)
  pr_test  <- sapply(pred_probs_test, function(p) pr.curve(scores.class0 = p[y_test == 1], 
                                                           scores.class1 = p[y_test == 0], 
                                                           curve = FALSE)$auc.integral)
  
  # Compute MCCs (if requested)
  mcc_train <- mcc_test <- rep(NA, length(pred_probs_train))
  if (compute_mcc) {
    binary_preds_train <- lapply(pred_probs_train, function(p) as.integer(p > 0.5))
    binary_preds_test  <- lapply(pred_probs_test,  function(p) as.integer(p > 0.5))
    
    mcc_fun <- function(y_true, y_pred) {
      TP <- sum(y_true == 1 & y_pred == 1)
      TN <- sum(y_true == 0 & y_pred == 0)
      FP <- sum(y_true == 0 & y_pred == 1)
      FN <- sum(y_true == 1 & y_pred == 0)
      num <- (TP * TN) - (FP * FN)
      den <- sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
      if (den == 0) return(NA)
      num / den
    }
    
    mcc_train <- mapply(mcc_fun, MoreArgs = list(y_true = y_train), binary_preds_train)
    mcc_test  <- mapply(mcc_fun, MoreArgs = list(y_true = y_test),  binary_preds_test)
  }
  
  # Construct output
  data.frame(
    Method     = names(beta_hats),
    AUC_Train  = round(auc_train, 4),
    AUC_Test   = round(auc_test, 4),
    PR_AUC_Train = round(pr_train, 4),
    PR_AUC_Test  = round(pr_test, 4),
    MCC_Train  = if (compute_mcc) round(mcc_train, 4) else NA,
    MCC_Test   = if (compute_mcc) round(mcc_test, 4) else NA
  )
}

suppressPackageStartupMessages({
  library(data.table)
  library(pROC)
  library(xgboost)
  library(dplyr)
  library(caret)
})

library(PRROC)
library(pROC)
library(dplyr)

alphas <- c(0.5, 10, 10, 100)
taus = c(100, 1000, 1000, 1000)
ethnicities <- c("African American", "Hispanic", "Asian", "Native American")

alpha_dict <- setNames(as.list(alphas), ethnicities)

tau_dict = setNames(as.list(taus), ethnicities)

suppressMessages(suppressWarnings({
  
  ethnicities <- c("African American",  "Hispanic", "Asian", "Native American" )
  source_ethnicity <- "Caucasian"
  main_folder <- "widsdatathon2021_Diabetes/error_results_pr"
  summary_folder <- "widsdatathon2021_Diabetes/summary_results_pr"
  
  dir.create(main_folder, showWarnings = FALSE)
  
  for (target_ethnicity in ethnicities) {
    cat("\n------------------------------------------------------\n")
    cat("-----------Processing Target Ethnicity:", target_ethnicity, "\n")
    
    ethnicity_folder <- paste0(main_folder, "/", gsub(" ", "_", target_ethnicity))
    dir.create(ethnicity_folder, showWarnings = FALSE)
    
    results_list <- list()
    
    for (seed in 1:20) {
      set.seed(seed)
      cat("-----------Processing Target Ethnicity:", target_ethnicity, "\n", 'seed: ', seed, '--------')
      data_target <- dat[ethnicity == target_ethnicity]
      data_target[, ethnicity := NULL]
      
      train_index <- createDataPartition(data_target$y, p = 0.05, list = FALSE)
      data_train <- data_target[train_index]
      data_rest <- data_target[-train_index]
      
      unlabeled_index <- createDataPartition(data_rest$y, p = 0.4737, list = FALSE)
      data_unlabeled <- data_rest[unlabeled_index]
      data_test <- data_rest[-unlabeled_index]
      
      X <- as.matrix(data_train[, -1, with = FALSE])
      y <- as.matrix(data_train[, 1])
      X_test <- as.matrix(data_test[, -1, with = FALSE])
      y_test <- as.matrix(data_test[, 1])
      
      X_tilde <- as.matrix(data_unlabeled[, !'y', with = FALSE])
      X_tilde_dmatrix <- xgb.DMatrix(data = X_tilde)
      Y_tilde_ML <- ifelse(predict(ML_model, X_tilde_dmatrix) > 0.25, 1, 0)
      
      mu_logistic <- mu_generate_logistic(X, y)
      mu_ridge <- mu_generate_ridge(X, y, family = 'binomial')
      mu_GLMTrans <- mu_generate_GLMTrans(X_tilde, X, y, Y_tilde_ML, family = 'binomial')
      
      mu_blastvar <- posterior_param_class(X, y, X_tilde, Y_tilde_ML,
                                          alpha = alpha_dict[[target_ethnicity]], k = 0, tau = tau_dict[[target_ethnicity]])$mu_blast
      
      beta_hats <- list(
        "mu_TL_var" = replace(as.matrix(mu_blastvar), is.na(mu_blastvar), 0),
        "mu_GLMTrans" = replace(as.matrix(mu_GLMTrans), is.na(mu_GLMTrans), 0),
        "ridge" = replace(as.matrix(mu_ridge), is.na(mu_ridge), 0),
        "logistic" = replace(as.matrix(mu_logistic), is.na(mu_logistic), 0)
      )
      
      res_df <- check_errors_classification(data_train, data_test, beta_hats)[, c("Method", "AUC_Train", "AUC_Test", "PR_AUC_Train", "PR_AUC_Test")]
      
      X_train_dmatrix <- xgb.DMatrix(data = as.matrix(X))
      X_test_dmatrix <- xgb.DMatrix(data = as.matrix(data_test[, -1]))
      
      train_probs_ML <- predict(ML_model, X_train_dmatrix)
      test_probs_ML <- predict(ML_model, X_test_dmatrix)
      
      pr_train <- pr.curve(scores.class0 = train_probs_ML, weights.class0 = y, curve = FALSE)
      pr_test <- pr.curve(scores.class0 = test_probs_ML, weights.class0 = y_test, curve = FALSE)
      
      auc_train_ML <- pROC::auc(y, train_probs_ML)
      auc_test_ML <- pROC::auc(y_test, test_probs_ML)
      
      ml_row <- data.frame(
        Method = "ML",
        AUC_Train = round(as.numeric(auc_train_ML), 4),
        AUC_Test = round(as.numeric(auc_test_ML), 4),
        PR_AUC_Train = round(pr_train$auc.integral, 4),
        PR_AUC_Test = round(pr_test$auc.integral, 4)
      )
      
      res_df <- rbind(res_df, ml_row)
      print(res_df)
      results_list[[seed]] <- res_df
      saveRDS(res_df, file.path(ethnicity_folder, paste0("errors_seed_", seed, ".rds")))
    }
    
    all_results <- rbindlist(results_list, fill = TRUE)
    
    summary_results <- all_results %>%
      group_by(Method) %>%
      summarise(
        avg_AUC_Train = mean(AUC_Train, na.rm = TRUE),
        sd_AUC_Train = sd(AUC_Train, na.rm = TRUE),
        avg_AUC_Test = mean(AUC_Test, na.rm = TRUE),
        sd_AUC_Test = sd(AUC_Test, na.rm = TRUE),
        avg_PR_Train = mean(PR_AUC_Train, na.rm = TRUE),
        sd_PR_Train = sd(PR_AUC_Train, na.rm = TRUE),
        avg_PR_Test = mean(PR_AUC_Test, na.rm = TRUE),
        sd_PR_Test = sd(PR_AUC_Test, na.rm = TRUE)
      ) 
    
    # Save summary statistics as .rds inside the respective folder
    summary_filename <- paste0(ethnicity_folder, "/summary_errors.rds")
    summary_content <- list(
      Target_Ethnicity = target_ethnicity,
      error_summary = summary_results
    )
    saveRDS(summary_content, file = summary_filename)
    
    # Print final summary for this ethnicity
    cat("\n------------------------------------------------------\n")
    cat("Final Summary for Target Ethnicity:", target_ethnicity, "\n")
    print(summary_results)
    cat("Results saved in:", ethnicity_folder, "\n")
    cat("------------------------------------------------------\n")
  }
}))




############  ------- Presentation 04/02 ------- ############ 
library(data.table)
library(ggplot2)

# Load seed-wise results
main_folder <- "widsdatathon2021_Diabetes/error_results_pr"
ethnicity_folders <- list.dirs(main_folder, recursive = FALSE)
all_results <- list()

for (eth_folder in ethnicity_folders) {
  ethnicity_name <- basename(eth_folder)
  seed_files <- list.files(eth_folder, pattern = "^errors_seed_\\d+\\.rds$", full.names = TRUE)
  
  for (file in seed_files) {
    res <- readRDS(file)
    res$Target_Ethnicity <- ethnicity_name
    all_results[[length(all_results) + 1]] <- res
  }
}

combined_results <- rbindlist(all_results, fill = TRUE)

# Rename method labels
rename_map <- c(
  "mu_TL_var" = "BLAST",
  "ridge" = "Ridge",
  "logistic" = "Logistic",
  "mu_GLMTrans" = "GLMTrans",
  "ML" = "Xgboost"
)

combined_results[, Method := rename_map[Method]]


ggplot(combined_results, aes(x = Method, y = AUC_Test, fill = Method)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.7) +
  facet_wrap(~ Target_Ethnicity, scales = "free_y", ncol = 2) +
  labs(title = "Test AUC Distribution Across Target Ethnicities",
       x = "Method", y = "AUC (Test)") +
  theme_minimal(base_size = 14) +
  theme(
    text = element_text(face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 12),
    axis.text.y = element_text(size = 12),
    axis.title = element_text(size = 16),
    strip.text = element_text(size = 16),
    plot.title = element_text(size = 18, hjust = 0.5),
    panel.spacing = unit(1.5, "lines"),
    panel.border = element_rect(color = "black", fill = NA, size = 1),
    
    # Legend formatting
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.box = "horizontal",
    legend.text = element_text(size = 12),
    legend.title = element_blank()
  ) +
  ylim(0.45, 0.8)

library(data.table)

# Define the summary folder path where all summary .rds files are saved
main_folder <- "widsdatathon2021_Diabetes/error_results_pr"
ethnicity_folders <- list.dirs(main_folder, recursive = FALSE)

# Initialize a list to store summary results
summary_list <- list()

# Loop through each ethnicity folder and load the summary .rds file
for (eth_folder in ethnicity_folders) {
  summary_file <- file.path(eth_folder, "summary_errors.rds")
  
  if (file.exists(summary_file)) {
    summary_content <- readRDS(summary_file)
    
    # Extract the summary data and attach ethnicity info
    summary_df <- summary_content$error_summary
    summary_df$Target_Ethnicity <- summary_content$Target_Ethnicity
    
    summary_list[[basename(eth_folder)]] <- summary_df
  } else {
    message("No summary_errors.rds found in ", eth_folder)
  }
}

# Combine all into one data.table
combined_summaries <- rbindlist(summary_list, fill = TRUE)

# View the combined summary
print(combined_summaries)





############ -------- Presentation 04/02 Housing ----- 


library(ggplot2)
library(dplyr)
library(maps)
library(ggthemes)

# 1. Prepare state and method results
midwest_states <- tolower(c("IL", "IN", "IA", "KS", "MI", "MN", "MO", "NE", "ND", "OH", "SD", "WI"))
us_map <- map_data("state") %>%
  filter(region %in% tolower(state.name[match(midwest_states, state.abb)]))

# 2. Example: Replace this with your real results
housing_summary_df <- data.frame(
  state = c("illinois", "indiana", "iowa", "kansas", "michigan", "minnesota", "missouri",
            "nebraska", "north dakota", "ohio", "south dakota", "wisconsin"),
  best_method = c("GLMTrans", "GLMTrans", "GLMTrans", "BLAST", "BLAST", "GLMTrans", "GLMTrans",
                  "GLMTrans", "TransLasso", "BLAST", "BLAST", "BLAST"),
  test_error = c(0.140, 0.188, 0.086, 0.165, 0.174, 0.171, 0.145,
                 0.283, 0.129, 0.135, 0.291, 0.114)
)

library(ggplot2)
library(dplyr)
library(maps)

# Step 1: Define Midwest states by abbreviation
midwest_abbr <- c("IL", "IN", "IA", "KS", "MI", "MN", "MO", "NE", "ND", "OH", "SD", "WI")

# Step 2: Convert to lowercase full names
midwest_states <- tolower(state.name[match(midwest_abbr, state.abb)])

# Step 3: Load US map and filter for midwest states
us_map <- map_data("state") %>%
  filter(region %in% midwest_states)

# Check result
head(us_map)



# 3. Merge map with method results
map_df <- us_map %>%
  left_join(housing_summary_df, by = c("region" = "state"))

# 4. Define method → color (blast = green)
method_colors <- c(
  "BLAST" = "forestgreen",
  "GLMTrans" = "#1f78b4",     # blue
  "Ridge" = "#e31a1c",        # red
  "TransLasso" = "#ff7f00",   # orange
  "PPI" = "#6a3d9a",          # purple
  "Xgboost" = "#b15928"       # brown
)

# 5. Plot map with fill = best_method
ggplot(map_df, aes(long, lat, group = group, fill = best_method)) +
  geom_polygon(color = "white") +
  scale_fill_manual(values = method_colors, na.value = "gray80") +
  coord_fixed(1.3) +
  theme_minimal(base_size = 14) +
  labs(title = "Best Performing Method by State (Midwest)",
       fill = "Method") +
  theme(
    legend.position = "bottom",
    legend.direction = "horizontal",
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    strip.text = element_text(size = 14, face = "bold"),
    legend.title = element_text(face = "bold")
  )

# Assuming df is your cleaned diabetes dataset
# and the relevant column is named 'ethnicity'

library(dplyr)

# Define the split proportions
split_props <- c(train = 0.05, unlabeled = 0.45, test = 0.50)
dat$eth
# Count and compute splits for each ethnicity
ethnicity_split_counts <- dat %>%
  group_by(ethnicity) %>%
  summarise(
    Total = n(),
    Train = floor(n() * split_props["train"]),
    Unlabeled = floor(n() * split_props["unlabeled"]),
    Test = n() - Train - Unlabeled
  ) %>%
  arrange(desc(Total))

print(ethnicity_split_counts)







############ -------- Presentation 04/06 Feature relevance

set.seed(1)

top5_features <- list()  # initialize the unified list

ethnicities =  c('Caucasian', "African American",  "Hispanic", "Asian", "Native American" )


alphas <- c(1, 0.5, 10, 10, 10)
taus = c(1000, 100, 1000, 1000, 1000)

alpha_dict <- setNames(as.list(alphas), ethnicities)

tau_dict = setNames(as.list(taus), ethnicities)

for (eth in ethnicities) {
  
  # Filter for the target ethnicity and remove the ethnicity column
  data_target <- dat[ethnicity == target_ethnicity]
  data_target[, ethnicity := NULL]
  
  # Step 1: Split target data into train (5%) and the rest (95%)
  train_index <- createDataPartition(data_target$y, p = 0.05, list = FALSE)
  data_train <- data_target[train_index]
  data_rest <- data_target[-train_index]
  
  # Step 2: Split rest into unlabeled (45%) and test (50%) of original
  # 45% of total = (45 / 95) ≈ 0.4737 of the remaining 95%
  unlabeled_index <- createDataPartition(data_rest$y, p = 0.4737, list = FALSE)
  data_unlabeled <- data_rest[unlabeled_index]
  data_test <- data_rest[-unlabeled_index]
  
  # Step 3: Prepare matrices for training and test
  X <- as.matrix(data_train[, -1, with = FALSE])     # Features for training
  y <- as.matrix(data_train[, 1, with = FALSE])      # Target for training
  
  X_test <- as.matrix(data_test[, -1, with = FALSE]) # Features for test
  y_test <- as.matrix(data_test[, 1, with = FALSE])  # Target for test
  
  # Step 4: Prepare X_tilde from unlabeled data and generate pseudo-labels
  X_tilde <- as.matrix(data_unlabeled[, !'y', with = FALSE])  # Drop target
  X_tilde_dmatrix <- xgb.DMatrix(data = X_tilde)
  Y_tilde_ML <- ifelse(predict(ML_model, X_tilde_dmatrix) > 0.25, 1, 0)
  
  # assume X, y, X_tilde, Y_tilde_ML are updated for each ethnicity
  blast_post <- posterior_param_class(X, y, X_tilde, Y_tilde_ML,
                                     alpha = alpha_dict[[eth]], 
                                     k = best_k, 
                                     tau = tau_dict[[target_ethnicity]])
  
  mu_blastvar <- blast_post$mu_blast
  sigma_diag <- diag(blast_post$sigma_blast)
  
  top_feats <- order(abs(mu_blastvar), decreasing = TRUE)[1:5]
  
  top5_features[[eth]] <- data.frame(
    Feature = colnames(X)[top_feats],
    Coefficient = mu_blastvar[top_feats],
    Std_Dev = sqrt(sigma_diag[top_feats])
  )
}

print(top5_features)

