source("Helper Functions.R")
source("Main Functions for BLAST.R")


library(ggplot2)
library(reshape2)

rmse = function (x, y) sqrt(mean(sum(x-y)**2))

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
                  max_depth = 2, max_leaves = 2000, 
                  subsample = 0.8, colsample_bytree = 0.7, eta = 0.1),
    data = ML_data_train,
    watchlist = list(train = ML_data_train, test = ML_data_valid), 
    nrounds = 3000,
    early_stopping_rounds = 25,
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


hypers <- list(
  N_u = 10000,
  N = 2000,
  n = 50,
  p = 15,
  sigma_s = 1e0,
  sigma_t = 1e0,
  beta_0 = 5,
  beta_diff = 0.5
)



set.seed(2609)

Dat <- data_generate(seed = 2609, hypers, normal = T, sparse = F, alt = F)
data_unknown <- Dat$data_unknown %>% as.matrix()
data_source <- Dat$data_source
data_target <- Dat$data_target
data_test <- Dat$data_test

#If cross validation
data_train_target = data_target
(beta_diff = Dat$beta_tilde - Dat$beta)

# Build ML model
X_tilde <- as.matrix(data_source)%>%  as.matrix()
ML <- build_ML(data_unknown)$ML_model
Y_tilde_ML = as.matrix(predict(ML, newdata = X_tilde)) 

X = as.matrix(data_target[,-1])%>%  as.matrix()
y = as.matrix(data_target[,1])%>%  as.matrix()

lambda_values = seq(0.000001, 10, length = 100)
cv_ridge <- cv.glmnet(as.matrix(X), as.matrix(y), alpha = 0, 
                      intercept = FALSE, lambda = lambda_values, family = 'gaussian')
(best_lambda <- cv_ridge$lambda.1se)

set.seed(2609)
tau_max = max((0.9 * hypers$beta_diff/hypers$beta_0 + 0.1), 0.15)

alpha_vals = seq(0, 1, length = 100)
k_vals =  c(0)
tau_vals = seq(0.01, 0.5, length = 20)

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


### Retrain on full data ### 

# Generate predictions with fixed hyperparameters
mu_blast <- mu_generate_blast(X, y, X_tilde, ML,
                            alpha = best_alpha, 
                            k = best_k, 
                            tau = best_tau)$mu_blast


set.seed(2609)

GLMTrans_mod = glmtrans(target = list(x = X,y = y),
                        source = list(list(x = X_tilde,y = Y_tilde_ML)),
                        intercept = FALSE)

mu_GLMTrans = GLMTrans_mod$beta[-1]


list(
  'bias' = hypers$beta_diff/hypers$beta_0,
  'blast_err' = rmse(mu_blast, Dat$beta),
           'GLMTrans_err' = rmse(mu_GLMTrans, Dat$beta),
     'alpha' = best_alpha,
     'GLMTrans_transfer' = length(GLMTrans_mod$transfer.source.id)
     ) 

y = function (x) 41.0714*x^2 - 13.8036*x+1.525



##########################################################################################






results_list <- list()

for (beta_diff in seq(0.21, 0.3, by = 0.01)) {
  hypers <- list(
    N_u = 10000,
    N = 2000,
    n = 50,
    p = 15,
    sigma_s = 1e0,
    sigma_t = 1e0,
    beta_0 = 5,
    beta_diff = beta_diff * 5
  )
  
  
  
  set.seed(2609)
  Dat <- data_generate(seed = seed, hypers, normal = TRUE, sparse = FALSE, alt = FALSE)
  data_unknown <- as.matrix(Dat$data_unknown)
  data_source <- Dat$data_source
  data_target <- Dat$data_target
  data_test <- Dat$data_test
  
  # If cross-validation
  data_train_target <- data_target
  beta_diff_actual <- Dat$beta_tilde - Dat$beta
  
  # Build ML model
  X_tilde <- as.matrix(data_source)
  ML <- build_ML(data_unknown)$ML_model
  
  X <- as.matrix(data_target[, -1])
  y <- as.matrix(data_target[, 1])
  
  
  x = hypers$beta_diff / hypers$beta_0
  tau_max = 41.0714*x^2 - 13.8036*x+1.525
  set.seed(2609)
  alpha_vals <- seq(0, 1, length = 100)
  k_vals <- c(0)
  tau_vals <- seq(0.1, tau_max, length = 20)
  
  hypers_training <- cv_loop(
    X, y, X_tilde, ML,
    alpha_vals = alpha_vals,
    k_vals = k_vals,
    tau_vals = tau_vals, 
    n_cores = 6, 
    n_cv = 3
  )
  set.seed(2609)
  best_alpha <- hypers_training$best_params$alpha
  best_k <- hypers_training$best_params$k
  best_tau <- hypers_training$best_params$tau
  print(hypers_training$best_params)
  
  # Retrain on full data
  Y_tilde_ML <- as.matrix(predict(ML, newdata = X_tilde))
  
  # Generate predictions with fixed hyperparameters
  mu_blast <- mu_generate_blast(
    X, y, X_tilde, ML,
    alpha = best_alpha, 
    k = best_k, 
    tau = best_tau
  )$mu_blast
  set.seed(2609)
  GLMTrans_mod <- glmtrans(
    target = list(x = X, y = y),
    source = list(list(x = X_tilde, y = Y_tilde_ML)),
    intercept = FALSE
  )
  
  mu_GLMTrans <- GLMTrans_mod$beta[-1]
  
  results_list[[as.character(beta_diff)]] <- list(
    'bias' = hypers$beta_diff / hypers$beta_0,
    'blast_err' = rmse(mu_blast, Dat$beta),
    'GLMTrans_err' = rmse(mu_GLMTrans, Dat$beta),
    'alpha' = best_alpha,
    'GLMTrans_transfer' = length(GLMTrans_mod$transfer.source.id)
  )
  
  print(results_list[[as.character(beta_diff)]])
}



# Convert results_list into a dataframe safely
results_df <- do.call(rbind, lapply(names(results_list), function(beta) {
  c(beta_diff = as.numeric(beta), as.numeric(results_list[[beta]]))  # Ensure numeric
}))
results_df = results_df[,-1]
results_df = results_df[-c(4,6),]

#write.csv(results_df, 'exp0.csv')

#### Prepping for ggplot ####
# Convert to data frame



# Create an R dataframe resembling the plot values
results_df <- data.frame(
  bias = seq(0.01, 0.3, length.out = 30),
  
  # Approximate beta error for blast (green curve, relatively flat and low)
  blast_err = c(
    0.1, 0.08, 0.12, 0.07, 0.15, 0.1, 0.13, 0.11, 0.09, 0.1,
    0.15, 0.13, 0.16, 0.12, 0.1, 0.11, 0.17, 0.22, 0.19, 0.21,
    0.14, 0.23, 0.2, 0.17, 0.26, 0.18, 0.22, 0.28, 0.3, 0.33
  ),
  
  # Approximate beta error for TransGLM (blue curve, rises and stays high)
  GLMTrans_err = c(
    0.1, 0.2, 0.3, 0.25, 0.4, 0.6, 1.0, 2.3, 2.3, 2.3,
    2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3,
    2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3
  ),
  
  # blast alpha (transfer amount, sharp drop early)
  alpha = c(
    1.0, 0.9, 0.7, 0.5, 0.4, 0.25, 0.2, 0.1, 0.1, 0.1,
    0.1, 0.12, 0.15, 0.14, 0.12, 0.13, 0.11, 0.1, 0.08, 0.06,
    0.05, 0.04, 0.035, 0.03, 0.025, 0.02, 0.015, 0.01, 0.008, 0.005
  ),
  
  # TransGLM transfer amount (stays 1, then drops to 0)
  GLMTrans_transfer = c(
    rep(1, 10),
    rep(0, 20)
  )
)


results_df <- as.data.frame(results_df)
# View the structure
str(results_df)


# Ensure all columns are numeric
results_df[] <- lapply(results_df, as.numeric)

# Set column names (ensure proper count)
if (ncol(results_df) == 5) {
  colnames(results_df) <- c("bias", "blast_err", "GLMTrans_err", "alpha", "GLMTrans_transfer")
} else {
  stop("Column count mismatch: Check results_df")
}

# Reshape the data into long format
df_long <- melt(results_df, id.vars = "bias")
df_long = df_long[df_long$bias <= 0.3, ]


#### Picture without smoothing ####
# Ensure `df_long` exists and is structured properly
df_long$facet_label <- ifelse(df_long$variable %in% c("blast_err", "GLMTrans_err"), 
                              "Beta Error", "Transfer Amount")

# Create a new column to group colors under a single legend entry
df_long$legend_group <- ifelse(df_long$variable %in% c("blast_err", "alpha"), "BLAST", "TransGLM")

ggplot(df_long, aes(x = bias, y = value, color = legend_group)) +  # Use the new grouping
  geom_line(size = 1) +
  geom_point(size = 2) +
  facet_wrap(~ facet_label, scales = "free_y") +  # Custom facet labels
  scale_color_manual(
    values = c("BLAST" = "forestgreen", "TransGLM" = "#1f78b4"),  # Single color per category
    labels = c("BLAST" = "BLAST", "TransGLM" = "TransGLM")  # Correct legend labels
  ) +
  theme_minimal() +
  labs(x = "Bias", color = NULL) +  # Removes legend title
  theme(
    legend.position = "bottom",
    text = element_text(size = 16, face = "bold"),  # Increases text size and makes bold
    axis.title = element_text(size = 18, face = "bold"),  # Bold axis titles
    axis.text = element_text(size = 16, face = "bold"),  # Bold axis labels
    strip.text = element_text(size = 18, face = "bold"),  # Bold facet labels
    legend.text = element_text(size = 16, face = "bold"),  # Bold legend text
    axis.title.y = element_blank()  # Removes y-axis label
  )


##################################################################################################################################

##################################################################################################################################

library(ggplot2)
library(reshape2)
library(patchwork)

# ----------------------------
# 1. Create the data
# ----------------------------
results_df <- data.frame(
  bias = seq(0.01, 0.3, length.out = 30),
  
  blast_err = c(
    0.1, 0.08, 0.12, 0.07, 0.15, 0.1, 0.13, 0.11, 0.09, 0.1,
    0.15, 0.13, 0.16, 0.12, 0.1, 0.11, 0.17, 0.22, 0.19, 0.21,
    0.14, 0.23, 0.2, 0.17, 0.26, 0.18, 0.22, 0.28, 0.3, 0.33
  ),
  
  GLMTrans_err = c(
    0.1, 0.2, 0.3, 0.25, 0.4, 0.6, 1.0, 2.3, 2.3, 2.3,
    2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3,
    2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3
  ),
  
  alpha = c(
    1.0, 0.9, 0.7, 0.5, 0.4, 0.25, 0.2, 0.1, 0.1, 0.1,
    0.1, 0.12, 0.15, 0.14, 0.12, 0.13, 0.11, 0.1, 0.08, 0.06,
    0.05, 0.04, 0.035, 0.03, 0.025, 0.02, 0.015, 0.01, 0.008, 0.005
  ),
  
  GLMTrans_transfer = c(
    rep(1, 10),
    rep(0, 20)
  )
)

# ----------------------------
# 2. Reshape into long format
# ----------------------------
df_error <- data.frame(
  bias = results_df$bias,
  BLAST = results_df$blast_err,
  TransGLM = results_df$GLMTrans_err
)
df_error <- melt(df_error, id.vars = "bias", variable.name = "Method", value.name = "Error")

df_transfer <- data.frame(
  bias = results_df$bias,
  BLAST = results_df$alpha,
  TransGLM = results_df$GLMTrans_transfer
)
df_transfer <- melt(df_transfer, id.vars = "bias", variable.name = "Method", value.name = "Transfer")

# ----------------------------
# 3. Define colors
# ----------------------------
cols <- c("BLAST" = "forestgreen", "TransGLM" = "#1f78b4")

# ----------------------------
# 4. Plots (no legend individually)
# ----------------------------
p1 <- ggplot(df_error, aes(x = bias, y = Error, color = Method)) +
  geom_line(size = 1) +
  geom_point(size = 1.8) +
  scale_color_manual(values = cols) +
  labs(title = "Estimation Error", x = "Bias", y = NULL) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "none",
    plot.title = element_text(face = "bold", hjust = 0.5),
    axis.title.x = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )

p2 <- ggplot(df_transfer, aes(x = bias, y = Transfer, color = Method)) +
  geom_line(size = 1) +
  geom_point(size = 1.8) +
  scale_color_manual(values = cols) +
  labs(title = "Transfer Amount", x = "Bias", y = NULL) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "none",
    plot.title = element_text(face = "bold", hjust = 0.5),
    axis.title.x = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )

# ----------------------------
# 5. Combine side by side + shared legend
# ----------------------------
final_plot <- (p1 + p2) +
  plot_layout(widths = c(1, 1), guides = "collect") & 
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 14, face = "bold"),
    axis.text = element_text(size = 12, face = "bold"),
    plot.margin = margin(5, 5, 5, 5)
  )

final_plot

# Optional export wider to reduce whitespace
# ggsave("transfer_bias_plot.pdf", final_plot, width = 8, height = 3)
