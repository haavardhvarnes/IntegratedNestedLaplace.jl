#!/usr/bin/env Rscript
#
# Phase 6f.1 component-level diagnostic (R-INLA side). At each fixed θ on
# a grid, extract the joint mode (`cfg$mean`), prior precision
# (`cfg$Qprior`), and joint posterior precision (`cfg$Q`), then *recompute*
# all the scalar components of `log p̂(y|θ)` from these — independent of
# R-INLA's internal `mlik`. Pair with `bench/brunei_sla_components.jl`
# and `bench/brunei_sla_compare.jl` to identify the τ-slope-bearing
# component that explains the residual ~5-nat right-tail gap.

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

# R-INLA's `cfg$mean` layout for `y ~ 1 + f(area, model="besag")`:
#   indices 1..n_areas  = area random effects u_i
#   index n_areas + 1   = intercept β
# Our Julia layout swaps these (β first), so we reorder when comparing.

run_one <- function(theta_val) {
  res <- inla(y ~ 1 + f(area, model = "besag", graph = g, scale.model = TRUE,
                         hyper = list(prec = list(initial = theta_val, fixed = TRUE))),
              family = "poisson", data = df, E = df$E,
              control.compute = list(config = TRUE, dic = FALSE, cpo = FALSE,
                                     return.marginals.predictor = FALSE),
              control.inla = list(strategy = "gaussian"),
              verbose = FALSE, silent = 2L)
  cfg <- res$misc$configs$config[[1]]

  # R-INLA layout: u_1..u_n_areas, β
  mode_x <- cfg$mean
  u_cm <- mode_x[seq_len(n_areas)]
  beta_cm <- mode_x[n_areas + 1]

  # Predictor at the joint mode (without offset)
  r_m <- beta_cm + u_cm[df$area]
  # Linear predictor with offset
  eta_star <- r_m + log(df$E)
  # Exact Poisson log-likelihood at the mode
  ll_exact <- sum(df$y * eta_star - exp(eta_star))

  # H_post = cfg$Q (graph-restricted!). Note: R-INLA stores Q on the GMRF
  # graph layout, so off-diagonals from intercept (isolated node) to area
  # block are ZERO even if the true joint H has them non-zero. This is the
  # known "graph-restricted partial" issue from BRUNEI_FIX.md Phase 6c.2.
  H_post <- as.matrix(cfg$Q)

  # Q_prior — R-INLA's stored Qprior. For besag at scale.model=TRUE plus
  # `prec.intercept=0`, the intercept block is all zeros and the area
  # block is τ · scaled_besag.
  Qprior <- as.matrix(cfg$Qprior)
  # R-INLA stores Qprior and cfg$Q as upper-triangular only (the lower-tri
  # entries are zeros). Correct symmetrization is `M + M' - diag(M)`, NOT
  # `(M + M')/2` (which would halve all off-diagonals). The previous version
  # of this diagnostic had the /2 form and consequently produced
  # log-determinants that disagreed with Julia by a τ-dependent factor — a
  # red herring chase. Fixed here.
  Qprior <- Qprior + t(Qprior) - diag(diag(Qprior))
  H_post <- H_post + t(H_post) - diag(diag(H_post))

  # Diagnostic: what is the actual (β,β) entry of cfg$Qprior? Tells us
  # whether R-INLA defaults `prec.intercept = 0` to literally zero or to
  # a small jitter (e.g. 1e-8).
  Qprior_intercept_block <- Qprior[n_areas + 1, n_areas + 1]
  H_post_intercept_block <- H_post[n_areas + 1, n_areas + 1]
  H_post_intercept_area_max <- max(abs(H_post[n_areas + 1, seq_len(n_areas)]))

  # User constraint A_user: sum-to-zero on u, normalized so A_user A_user' = 1
  # Layout: (u_1..u_n_areas, β)
  A_user <- matrix(c(rep(1 / sqrt(n_areas), n_areas), 0), nrow = 1)

  # Augmented constraint A_full = [A_user; e_intercept']
  e_intercept <- matrix(c(rep(0, n_areas), 1), nrow = 1)
  A_full <- rbind(A_user, e_intercept)

  # Log-determinants on different constraint subspaces.
  log_det_H_post <- as.numeric(determinant(H_post, logarithm = TRUE)$modulus)

  Wc_user <- solve(H_post, t(A_user))
  S_user <- A_user %*% Wc_user
  log_AcHinv_user <- log(as.numeric(S_user))

  Wc_full <- solve(H_post, t(A_full))
  S_full <- A_full %*% Wc_full
  log_det_AcHinv_full <- as.numeric(determinant(S_full, logarithm = TRUE)$modulus)

  log_AcAt_user <- log(as.numeric(A_user %*% t(A_user)))   # ≈ 0
  log_det_AcAt_full <- as.numeric(determinant(A_full %*% t(A_full),
                                               logarithm = TRUE)$modulus)

  # log|H_c| under the textbook PLUS form, both subspaces
  log_det_H_c_user <- log_det_H_post + log_AcHinv_user      # textbook PLUS
  log_det_H_c_full <- log_det_H_post + log_det_AcHinv_full

  # Augmented form (Rue & Held 2005 eq. 2.30) for both subspaces
  log_det_H_aug_user <- as.numeric(determinant(H_post + t(A_user) %*% A_user,
                                                logarithm = TRUE)$modulus)
  log_det_H_aug_full <- as.numeric(determinant(H_post + t(A_full) %*% A_full,
                                                logarithm = TRUE)$modulus)

  # Same for Q_prior (which IS singular along sum-to-zero, plus possibly the
  # intercept direction depending on prec.intercept handling).
  log_det_Q_aug_user <- as.numeric(determinant(Qprior + t(A_user) %*% A_user,
                                                logarithm = TRUE)$modulus)
  log_det_Q_aug_full <- as.numeric(determinant(Qprior + t(A_full) %*% A_full,
                                                logarithm = TRUE)$modulus)

  # Quadratic forms at the mode
  x_full <- c(u_cm, beta_cm)   # R-INLA layout
  quad_xQpx <- 0.5 * as.numeric(t(x_full) %*% Qprior %*% x_full)
  quad_xHx  <- 0.5 * as.numeric(t(x_full) %*% H_post %*% x_full)

  # Sum-to-zero residual (sanity)
  sum_u <- sum(u_cm)

  # Per-coordinate η-marginal variance σ²_η (= diag(A H⁻¹ Aᵀ))
  # A is the design that maps (u, β) → η_i = β + u_{area_i}
  A_design <- matrix(0, nrow = nrow(df), ncol = length(x_full))
  for (i in seq_len(nrow(df))) {
    A_design[i, df$area[i]] <- 1   # u_{area_i}
    A_design[i, n_areas + 1] <- 1  # β
  }
  H_inv_AT <- solve(H_post, t(A_design))
  sigma2_eta <- rowSums(A_design * t(H_inv_AT))

  # log p(y_i | η_i) higher derivatives at the mode (Poisson)
  lambda_i <- exp(eta_star)
  d3 <- -lambda_i           # d³log p(y|η)/dη³ = -exp(η)
  d4 <- -lambda_i           # d⁴log p(y|η)/dη⁴ = -exp(η)

  list(
    theta = theta_val,
    tau   = exp(theta_val),
    n_obs = nrow(df),
    n_latent = length(x_full),
    # Mode
    beta_mode = beta_cm,
    u_mode_first5 = u_cm[1:5],
    u_mode_full = u_cm,                # full R-INLA mode of u (Phase 6g.1 input)
    sum_u = sum_u,
    norm_u = sqrt(sum(u_cm^2)),
    # R-INLA's mlik
    mlik_int   = res$mlik[1, 1],
    mlik_gauss = res$mlik[2, 1],
    # Likelihood
    ll_exact = ll_exact,
    # Quadratic forms at mode
    quad_xQpx = quad_xQpx,
    quad_xHx  = quad_xHx,
    # Log-determinants — R-INLA stored quantities
    log_det_H_post = log_det_H_post,
    # Constraint corrections (textbook PLUS form)
    log_det_H_c_user = log_det_H_c_user,
    log_det_H_c_full = log_det_H_c_full,
    log_AcHinv_user = log_AcHinv_user,
    log_det_AcHinv_full = log_det_AcHinv_full,
    log_AcAt_user = log_AcAt_user,
    log_det_AcAt_full = log_det_AcAt_full,
    # Augmented (Rue-Held) form
    log_det_H_aug_user = log_det_H_aug_user,
    log_det_H_aug_full = log_det_H_aug_full,
    log_det_Q_aug_user = log_det_Q_aug_user,
    log_det_Q_aug_full = log_det_Q_aug_full,
    # Intercept-block diagnostic: tells us how R-INLA handles prec.intercept=0
    Qprior_intercept_block = Qprior_intercept_block,
    H_post_intercept_block = H_post_intercept_block,
    H_post_intercept_area_max = H_post_intercept_area_max,
    # Full Qprior diagonal (for slope investigation — does the area block scale
    # exactly as τ * scaled_besag, or is there an extra τ-dependent term?)
    Qprior_diag = diag(Qprior),
    Qprior_min_offdiag = min(Qprior[upper.tri(Qprior)]),
    Qprior_max_offdiag = max(Qprior[upper.tri(Qprior)]),
    Qprior_dim = dim(Qprior),
    # Per-coordinate diagnostics
    sigma2_eta = sigma2_eta,
    lambda_i = lambda_i,
    d3 = d3,
    d4 = d4,
    eta_star = eta_star,
    r_m = r_m
  )
}

cat("Scanning θ grid for component-level diagnostic...\n")
results <- lapply(theta_grid, function(t) {
  cat(sprintf("  θ = %5.2f ... ", t))
  r <- tryCatch(run_one(t), error = function(e) {
    cat("FAILED:", conditionMessage(e), "\n")
    NULL
  })
  if (!is.null(r)) {
    cat(sprintf("mlik=%.4f  log|H|=%.4f  log|H_c|_user=%.4f  log|H_c|_full=%.4f\n",
                r$mlik_int, r$log_det_H_post, r$log_det_H_c_user, r$log_det_H_c_full))
  }
  r
})

ok <- !sapply(results, is.null)
out <- list(
  theta_grid = theta_grid[ok],
  rows = results[ok]
)
out_path <- file.path(HERE, "brunei_sla_components_rinla.json")
write_json(out, out_path, pretty = TRUE, auto_unbox = TRUE, digits = 12, null = "null")
cat(sprintf("\nWrote %s\n", out_path))

cat("\n=== Component summary across θ ===\n")
cat(sprintf("%6s  %10s  %10s  %12s  %14s  %14s  %14s\n",
            "θ", "τ", "ll_exact", "log|H|", "log|H_c|_user", "log|H_c|_full",
            "log|Q_aug|_user"))
for (r in results) {
  if (is.null(r)) next
  cat(sprintf("%6.2f  %10.2f  %10.3f  %12.3f  %14.3f  %14.3f  %14.3f\n",
              r$theta, r$tau, r$ll_exact, r$log_det_H_post,
              r$log_det_H_c_user, r$log_det_H_c_full, r$log_det_Q_aug_user))
}
