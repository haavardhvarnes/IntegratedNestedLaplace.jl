#!/usr/bin/env Rscript
# Extract R-INLA's Qprior at fixed θ and write to CSV for direct comparison
# with Julia's Q.

suppressPackageStartupMessages({library(INLA); library(Matrix)})
df <- read.csv("examples/06_brunei_school_disparities/data/areas.csv")
g  <- inla.read.graph("examples/06_brunei_school_disparities/data/areas.graph")

theta_fixed <- 2.0
res <- inla(y ~ 1 + f(area, model = "besag", graph = g, scale.model = TRUE,
                       hyper = list(prec = list(initial = theta_fixed, fixed = TRUE))),
            family = "poisson", data = df, E = df$E,
            control.compute = list(config = TRUE), silent = 2L)
cfg <- res$misc$configs$config[[1]]

cat("dim Qprior:", dim(cfg$Qprior), "\n")
cat("dim Q:", dim(cfg$Q), "\n")
cat("dim mean:", length(cfg$mean), "\n\n")

# Print the diagonal pattern of Qprior
Qp <- as.matrix(cfg$Qprior)
cat("Qprior diagonal (first 5):", round(diag(Qp)[1:5], 4), "\n")
cat("Qprior[1, 1:5]:", round(Qp[1, 1:5], 4), "\n")
cat("Qprior nnz:", sum(Qp != 0), "\n")
cat("Qprior trace:", round(sum(diag(Qp)), 4), "\n")

# What's stored at indices? Let's see the labels
cat("\ncfg$contents (latent component names and indices):\n")
print(cfg$contents)

# Save Qprior and mean
write.csv(Qp, "bench/brunei_rinla_Qprior.csv", row.names = FALSE)
write.csv(as.matrix(cfg$Q), "bench/brunei_rinla_Q.csv", row.names = FALSE)
write.csv(data.frame(mean = cfg$mean), "bench/brunei_rinla_mean.csv", row.names = FALSE)
cat("\nWrote Qprior, Q, mean to bench/\n")
