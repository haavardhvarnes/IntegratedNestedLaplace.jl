#!/usr/bin/env Rscript
#
# Phase 6g+ Phase A — extra(θ) breakdown.
#
# At each θ on the grid, compute R-INLA's `extra(θ)` empirically as
#     extra_implied(θ) = mlik(θ) - [Σ a_i - sub_logdens(0)]
# where Σ a_i is R-INLA's Taylor at sample = 0 (computed from our
# closed form for Poisson) and sub_logdens(0) is the constrained
# Gaussian density at x=0 (from R-INLA's `cfg$Q` and `cfg$mean`).
#
# Then decompose `extra_implied(θ)` against the per-block formulas
# that R-INLA's `inla.c::extra()` is supposed to emit:
#   * Predictor block:  predictor_n * (LOG_NORMC + ½ predictor_log_prec)
#   * Besag block:      (N-rankdef) * LOG_NORMC + (N-rankdef)/2 * θ
#   * (Hyperprior on log τ: SKIPPED for fixed=TRUE)
#
# Output: per-θ table with raw mlik, sum_a, sub_logdens, extra_implied,
# extra_besag (theoretical), residual = extra_implied - extra_besag.
# The residual is the missing block(s) we need to reproduce in Julia.

suppressPackageStartupMessages({
  library(INLA)
  library(jsonlite)
  library(Matrix)
})

theta_grid <- c(0, 1, 1.88, 3, 5, 7, 10)

this_script <- normalizePath(sub("--file=", "",
  grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)[1]))
HERE <- dirname(this_script)
ROOT <- normalizePath(file.path(HERE, ".."))
DATA_DIR <- file.path(ROOT, "examples", "06_brunei_school_disparities", "data")

df <- read.csv(file.path(DATA_DIR, "areas.csv"))
g  <- inla.read.graph(file.path(DATA_DIR, "areas.graph"))
n_areas <- nrow(df)

LOG_NORMC_GAUSSIAN <- -0.5 * log(2 * pi)

# Per-i Poisson Taylor at r = 0 from r_m_i, including lfact for absolute
# match against R-INLA's mlik. Same form as in
# `bench/brunei_sla_components.jl::_taylor_at_zero_loglik`.
sum_a_poisson <- function(y, r_m, offset) {
  eta_m <- r_m + offset
  lambda_m <- exp(eta_m)
  log_y_fact <- sapply(y, function(yi) if (yi == 0) 0 else sum(log(seq_len(yi))))
  f0 <- y * eta_m - lambda_m - log_y_fact
  fp <- y - lambda_m
  fpp <- -lambda_m
  fppp <- -lambda_m
  a <- f0 - fp * r_m + 0.5 * fpp * r_m^2 - (1/6) * fppp * r_m^3
  sum(a)
}

# sub_logdens(0): R-INLA's `GMRFLib_evaluate(problem)` at sample=0
# with mean=μ (joint mode) and precision=H (cfg$Q). From
# problem-setup.c::1017–1049:
#   sub_logdens(x) = -½ n log(2π) + ½ log|H| - ½ (x-μ)' H (x-μ)
#                   - ½ log|A·A'| + ½ nc log(2π)
#                   + ½ log|A H⁻¹ A'| + ½ (Aμ-b)'(A H⁻¹ A')⁻¹(Aμ-b)
sub_logdens_at_zero <- function(H_post, mu, A_user, b = 0) {
  n <- nrow(H_post)
  nc <- nrow(A_user)
  AAt <- as.numeric(A_user %*% t(A_user))
  log_det_AAt <- log(AAt)
  log_det_H <- as.numeric(determinant(H_post, logarithm = TRUE)$modulus)
  H_inv_At <- solve(H_post, t(A_user))
  AHinvAt <- as.numeric(A_user %*% H_inv_At)
  log_det_AHinvAt <- log(AHinvAt)
  Amu_b <- as.numeric(A_user %*% mu - b)
  exp_corr <- (Amu_b)^2 / AHinvAt
  quad_xMux <- as.numeric(t(mu) %*% H_post %*% mu)        # at x=0: (x-μ)'H(x-μ) = μ'Hμ
  - 0.5 * n * log(2 * pi) + 0.5 * log_det_H - 0.5 * quad_xMux -
    0.5 * log_det_AAt + 0.5 * nc * log(2 * pi) +
    0.5 * log_det_AHinvAt + 0.5 * exp_corr
}

run_one <- function(theta_val) {
  res <- inla(y ~ 1 + f(area, model = "besag", graph = g, scale.model = TRUE,
                         hyper = list(prec = list(initial = theta_val, fixed = TRUE))),
              family = "poisson", data = df, E = df$E,
              control.compute = list(config = TRUE, dic = FALSE, cpo = FALSE,
                                     return.marginals.predictor = FALSE),
              control.inla = list(strategy = "gaussian"),
              verbose = FALSE, silent = 2L)
  cfg <- res$misc$configs$config[[1]]

  # R-INLA layout: cfg$mean = (u_1..u_n_areas, β)
  mode_x <- cfg$mean
  u_cm <- mode_x[seq_len(n_areas)]
  beta_cm <- mode_x[n_areas + 1]
  r_m <- beta_cm + u_cm[df$area]

  # Symmetrize cfg$Q (upper-tri storage; M + M' - diag(M)).
  H_post <- as.matrix(cfg$Q)
  H_post <- H_post + t(H_post) - diag(diag(H_post))

  # R-INLA's actual constraint: A_user = (1, 1, ..., 1, 0) un-normalized.
  # Layout matches cfg$mean (u first, β last).
  A_user_R <- matrix(c(rep(1, n_areas), 0), nrow = 1)

  sum_a    <- sum_a_poisson(df$y, r_m, log(df$E))
  sub_log0 <- sub_logdens_at_zero(H_post, mode_x, A_user_R)
  mlik     <- res$mlik[1, 1]

  # Implied `extra(θ)` from R-INLA's code: mlik = sum_a - sub_logdens + extra
  # ⇒  extra = mlik - sum_a + sub_logdens
  extra_implied <- mlik - sum_a + sub_log0

  # Theoretical besag-block contribution from inla.c::extra() line 2986–2987:
  #   val += LOG_NORMC * (N - rankdef) + (N - rankdef)/2 * (log τ + scale_correction)
  # For besag with scale.model=TRUE: scale_correction = 0.
  # N = n_areas = 42, rankdef = 1.
  N_besag <- n_areas
  rankdef_besag <- 1
  effective_rank <- N_besag - rankdef_besag    # 41
  extra_besag <- LOG_NORMC_GAUSSIAN * effective_rank + effective_rank / 2 * theta_val

  # Residual = extra_implied - extra_besag = whatever non-besag blocks
  # contribute.
  residual <- extra_implied - extra_besag

  list(
    theta = theta_val,
    tau = exp(theta_val),
    mlik = mlik,
    sum_a = sum_a,
    sub_logdens0 = sub_log0,
    extra_implied = extra_implied,
    extra_besag = extra_besag,
    residual = residual,                       # what we still need to explain
    # Components for sanity checks
    log_det_H_post = as.numeric(determinant(H_post, logarithm = TRUE)$modulus),
    quad_xHx = 0.5 * as.numeric(t(mode_x) %*% H_post %*% mode_x),
    beta_R = beta_cm,
    sum_u_R = sum(u_cm),
    norm_u_R = sqrt(sum(u_cm^2))
  )
}

cat(sprintf("%6s  %10s  %12s  %12s  %12s  %12s  %12s  %12s\n",
            "θ", "τ", "mlik(R)", "sum_a", "sub_log0",
            "extra_impl", "extra_besag", "residual"))
results <- list()
for (t in theta_grid) {
  r <- run_one(t)
  results[[length(results) + 1]] <- r
  cat(sprintf("%6.2f  %10.2f  %12.4f  %12.4f  %12.4f  %12.4f  %12.4f  %12.4f\n",
              r$theta, r$tau, r$mlik, r$sum_a, r$sub_logdens0,
              r$extra_implied, r$extra_besag, r$residual))
}

cat("\n=== Slopes (per unit θ) anchored at θ=1.88 ===\n")
anchor_idx <- which(sapply(results, function(r) abs(r$theta - 1.88) < 1e-3))[1]
anchor <- results[[anchor_idx]]
cat(sprintf("%6s  %10s  %12s  %12s  %12s  %12s\n",
            "θ", "Δθ", "Δresidual", "ΔextraImpl", "ΔextraBesag", "Δmlik(R)"))
for (r in results) {
  dθ <- r$theta - anchor$theta
  cat(sprintf("%6.2f  %10.3f  %12.4f  %12.4f  %12.4f  %12.4f\n",
              r$theta, dθ,
              r$residual - anchor$residual,
              r$extra_implied - anchor$extra_implied,
              r$extra_besag - anchor$extra_besag,
              r$mlik - anchor$mlik))
}

# Dump JSON for downstream Julia comparison.
out_path <- file.path(HERE, "brunei_extra_breakdown.json")
write_json(list(theta_grid = theta_grid, rows = results),
           out_path, pretty = TRUE, auto_unbox = TRUE, digits = 12, null = "null")
cat(sprintf("\nWrote %s\n", out_path))
