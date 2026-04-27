#!/usr/bin/env Rscript
#
# Phase 6c.1 R-INLA-side diagnostic. Mirrors `bench/brunei_dump.jl` so we
# can identify which scalar component disagrees between Julia and R-INLA
# at the same fixed θ.

suppressPackageStartupMessages({
  library(INLA)
  library(jsonlite)
})

theta_fixed <- 2.0  # log τ for the besag effect

this_script <- normalizePath(sub("--file=", "",
  grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)[1]))
HERE <- dirname(this_script)
ROOT <- normalizePath(file.path(HERE, ".."))
DATA_DIR <- file.path(ROOT, "examples", "06_brunei_school_disparities", "data")

df <- read.csv(file.path(DATA_DIR, "areas.csv"))
g  <- inla.read.graph(file.path(DATA_DIR, "areas.graph"))

# Hold θ fixed at theta_fixed; R-INLA still optimizes over the (no other) hypers.
# The `besag` model has just one log-precision hyper, so this fixes it entirely.
res <- inla(y ~ 1 + f(area, model = "besag", graph = g, scale.model = TRUE,
                       hyper = list(prec = list(initial = theta_fixed, fixed = TRUE))),
            family = "poisson", data = df, E = df$E,
            control.compute = list(config = TRUE, dic = FALSE, cpo = FALSE),
            verbose = FALSE, silent = 2L)

cat("=== R-INLA dump at θ = log τ =", theta_fixed, "===\n\n")
cat("R-INLA cpu.used Total:", as.numeric(res$cpu.used["Total"]), "s\n\n")
cat("Posterior mean of (intercept, area[1:5]):\n")
intercept_mean <- res$summary.fixed$mean
area_mean      <- res$summary.random$area$mean
cat(sprintf("  intercept  = %.6f\n", intercept_mean))
cat(sprintf("  area[1:5]  = [%s]\n", paste(round(area_mean[1:5], 6), collapse = ", ")))
cat(sprintf("  sum(area)  = %.6e\n\n", sum(area_mean)))

cat("Hyperparameter at fixed θ:\n")
print(res$summary.hyperpar[, c("mean", "sd", "mode")])
cat("\n")

# Internal config (mode at fixed θ + Q + log-density terms)
cfg <- res$misc$configs$config[[1]]
str(cfg, max.level = 1)
cat("\nlog.posterior   =", cfg$log.posterior, "\n")
if (!is.null(cfg$log.norm.const)) {
  cat("log.norm.const  =", cfg$log.norm.const, "\n")
}
if (!is.null(cfg$log.prior)) {
  cat("log.prior       =", cfg$log.prior, "\n")
}

# The configs[[1]]$mean is the constrained posterior mean for x. The Q
# stored is the "joint" precision (Q + Aᵀ D A), i.e., what we call H.
mode_x <- cfg$mean
n_x <- length(mode_x)
cat("\nlength of mode_x =", n_x, "\n")
cat("mode_x[1:6] =", paste(round(mode_x[1:6], 6), collapse = ", "), "\n")

# log p(y | x*, θ): evaluate Poisson log-likelihood at the *conditional mode*
# at θ_fixed. R-INLA's `cfg$mean` layout for `y ~ 1 + f(area, model="besag")` is
#   cfg$mean[1:42]  = u_i (area random effects)
#   cfg$mean[43]    = β   (intercept)
# so η_i = β + u_i, and the linear predictor evaluated at the data is η_i + log(E_i).
beta_cm <- mode_x[length(mode_x)]
u_cm    <- mode_x[seq_len(length(mode_x) - 1)]
eta_star <- beta_cm + u_cm[df$area] + log(df$E)
ll <- sum(df$y * eta_star - exp(eta_star))
cat(sprintf("\nβ (conditional mode at θ_fixed) = %.6f\n", beta_cm))
cat(sprintf("u[1:5] (conditional mode)       = [%s]\n",
            paste(round(u_cm[1:5], 6), collapse = ", ")))
cat(sprintf("predictor[1:5] = β + u[1:5]     = [%s]\n",
            paste(round(beta_cm + u_cm[1:5], 6), collapse = ", ")))
cat(sprintf("log p(y | x*, θ)                 = %.6f\n", ll))

# Latent prior contribution: 0.5 log|Q*| - 0.5 x*' Q x* (intrinsic)
tau <- exp(theta_fixed)
n_areas <- length(u_cm)

# Use R-INLA's own Qprior (so we can verify our Julia Q matches).
Qp <- as.matrix(cfg$Qprior)
# Quadratic at the conditional mode. R-INLA's latent layout is (u, β):
#   x = c(u, β)
x_full <- c(u_cm, beta_cm)
quad_xQx_full <- 0.5 * as.numeric(t(x_full) %*% Qp %*% x_full)
cat(sprintf("(1/2) x*' Qprior x*     = %.6f   (using R-INLA Qprior, R-INLA layout)\n",
            quad_xQx_full))

# Augmented-form intrinsic log-det. R-INLA's layout (u, β); constraint is on u
# (sum-to-zero), with normalization A_c A_c' = 1.
A_c <- matrix(c(rep(1 / sqrt(n_areas), n_areas), 0), nrow = 1)
Q_aug <- Qp + t(A_c) %*% A_c
log_det_Q_aug <- as.numeric(determinant(Q_aug, logarithm = TRUE)$modulus)
cat(sprintf("log|Qprior + A_c'A_c|   = %.6f\n", log_det_Q_aug))

# H = cfg$Q (joint posterior precision = Qprior + likelihood-Hessian)
H <- as.matrix(cfg$Q)
log_det_H <- as.numeric(determinant(H, logarithm = TRUE)$modulus)
Wc <- solve(H, t(A_c))
log_AcHinv <- log(as.numeric(A_c %*% Wc))
cat(sprintf("log|H| (cfg$Q)          = %.6f\n", log_det_H))
cat(sprintf("log(A_c H^{-1} A_c')    = %.6f\n", log_AcHinv))
cat(sprintf("log|H_c| (textbook)     = %.6f\n", log_det_H - log_AcHinv))

cat(sprintf("\nR-INLA cfg$log.posterior = %.6f   (their internal log p̂(θ|y) up to const)\n",
            cfg$log.posterior))
cat(sprintf("R-INLA mlik              = %.6f\n", res$mlik[1, 1]))

dump <- list(
  theta            = theta_fixed,
  tau              = tau,
  n_obs            = nrow(df),
  intercept        = intercept,
  u_first5         = u[1:5],
  sum_u            = sum(u),
  norm_u           = sqrt(sum(u^2)),
  ll               = ll,
  quad_xQx         = quad_xQx,
  log_det_Q_aug    = log_det_Q_aug,
  log_det_H        = log_det_H,
  log_AcHinvAct    = log_AcHinv,
  log_det_H_c      = if (is.na(log_det_H)) NA else log_det_H - log_AcHinv,
  rinla_log_post   = cfg$log.posterior,
  rinla_mlik       = res$mlik[1, 1]
)
out_path <- file.path(HERE, "brunei_dump_rinla.json")
write_json(dump, out_path, pretty = TRUE, auto_unbox = TRUE, digits = 10)
cat(sprintf("\nWrote %s\n", out_path))
