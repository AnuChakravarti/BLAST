library(ggplot2)
library(dplyr)

params_fn = function(XtY, XtX, tau, alpha, k){
  mu_pr = (alpha*XtX + tau)^(-1) * (alpha*XtY)
  sig_pr = (k*alpha*XtX + tau)^(-1)
  uninf_sig = 1/tau
  return(list(mu_pr = mu_pr, sig_pr = sig_pr, uninf_sig = uninf_sig))
}

XtY = 0.2
XtX = 0.1
tau = 0.8
alpha = 15

params_1 = params_fn(XtY, XtX, tau, alpha, 1)
params_2 = params_fn(XtY, XtX, tau, alpha, 0.25)
params_3 = params_fn(XtY, XtX, tau, alpha, 0)

x <- seq(-5, 10, length.out = 500)

# Helper function to create data for each facet
make_df <- function(mu, sigma, label, uninf_sig) {
  data.frame(
    x = rep(x, 2),
    density = c(
      dnorm(x, 0, uninf_sig),   # baseline prior
      dnorm(x, mu, sigma)       # pi(beta)
    ),
    dist = factor(rep(c("No transfer prior", "BLAST prior"), each = length(x)),
                  levels = c("No transfer prior", "BLAST prior")),
    facet = label
  )
}

# Build data
df <- bind_rows(
  make_df(params_1$mu_pr, params_1$sig_pr, "k = 1", params_1$uninf_sig),
  make_df(params_2$mu_pr, params_2$sig_pr, "k = 0.25", params_2$uninf_sig),
  make_df(params_3$mu_pr, params_3$sig_pr, "k = 0", params_3$uninf_sig)
)

# Make facet a factor with levels in desired order
df$facet <- factor(df$facet, levels = c("k = 1", "k = 0.25", "k = 0"))

# Faceted plot
ggplot(df, aes(x = x, y = density, color = dist)) +
  geom_line(size = 1.2) +
  facet_wrap(~facet, nrow = 3) +
  scale_color_manual(values = c("No transfer prior" = "blue", "BLAST prior" = "forestgreen")) +
  geom_vline(xintercept = 2.8, linetype = "dashed", color = "black") +
  geom_vline(xintercept = params_1$mu_pr, linetype = "dashed", color = "black") +
  #annotate("text", x = params_1$mu_pr, y = -0.04, label = expression(mu[pr]), hjust = 1) +
  #annotate("text", x = 4.5, y = -0.04, label = expression(beta^"*"), hjust = 1) +
  labs(x = NULL,y = NULL, color = "") +
  coord_cartesian(ylim = c(-0.05, max(df$density) * 1.1)) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "bottom")

