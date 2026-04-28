#!/usr/bin/env Rscript
#
# Phase B (simplified-laplace strategy) diagnostic: scan a θ grid, fix θ at
# each value via `prec.initial = θ, fixed = TRUE`, and dump R-INLA's
# `mlik[1,1]` (= log p(y|θ) under R-INLA's strategy) along with the
# per-fixed-θ summary moments. Pair with `bench/brunei_sla_diagnostic.jl`
# to identify *exactly* where R-INLA's `log p̂(y|θ)` differs from ours.

suppressPackageStartupMessages({
  library(INLA)
  library(jsonlite)
})

theta_grid <- c(-1, 0, 0.5, 1.0, 1.5, 1.88, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0)

this_script <- normalizePath(sub("--file=", "",
  grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)[1]))
HERE <- dirname(this_script)
ROOT <- normalizePath(file.path(HERE, ".."))
DATA_DIR <- file.path(ROOT, "examples", "06_brunei_school_disparities", "data")

df <- read.csv(file.path(DATA_DIR, "areas.csv"))
g  <- inla.read.graph(file.path(DATA_DIR, "areas.graph"))

run_one <- function(theta_val) {
  res <- inla(y ~ 1 + f(area, model = "besag", graph = g, scale.model = TRUE,
                         hyper = list(prec = list(initial = theta_val, fixed = TRUE))),
              family = "poisson", data = df, E = df$E,
              control.compute = list(config = TRUE, dic = FALSE, cpo = FALSE,
                                     return.marginals.predictor = FALSE),
              control.inla = list(strategy = "simplified.laplace"),
              verbose = FALSE, silent = 2L)
  cfg <- res$misc$configs$config[[1]]
  list(
    theta            = theta_val,
    tau              = exp(theta_val),
    mlik_int         = res$mlik[1, 1],   # integrated log marginal likelihood (over θ — but θ fixed here ⇒ a single Laplace value).
    mlik_gauss       = res$mlik[2, 1],   # Gaussian-only approximation
    log_posterior    = cfg$log.posterior,
    log_norm_const   = if (!is.null(cfg$log.norm.const)) cfg$log.norm.const else NA_real_,
    log_prior        = if (!is.null(cfg$log.prior))      cfg$log.prior      else NA_real_,
    intercept_mean   = res$summary.fixed$mean,
    intercept_sd     = res$summary.fixed$sd
  )
}

cat("Scanning θ grid for R-INLA's per-θ mlik...\n")
results <- lapply(theta_grid, function(t) {
  cat(sprintf("  θ = %5.2f ... ", t))
  r <- tryCatch(run_one(t), error = function(e) {
    cat("FAILED:", conditionMessage(e), "\n")
    NULL
  })
  if (!is.null(r)) {
    cat(sprintf("mlik_int=%.4f  mlik_gauss=%.4f  log.post=%.4f\n",
                r$mlik_int, r$mlik_gauss, r$log_posterior))
  }
  r
})

# Dump table.
ok <- !sapply(results, is.null)
out <- list(
  theta_grid = theta_grid[ok],
  rows       = results[ok]
)
out_path <- file.path(HERE, "brunei_sla_diagnostic_rinla.json")
write_json(out, out_path, pretty = TRUE, auto_unbox = TRUE, digits = 10, null = "null")
cat(sprintf("\nWrote %s\n", out_path))

cat("\n=== R-INLA per-θ mlik comparison: integrated vs gaussian-only ===\n")
cat(sprintf("%6s  %10s  %12s  %12s  %12s\n",
            "θ", "τ", "mlik_int", "mlik_gauss", "Δ (int-gauss)"))
for (r in results) {
  if (is.null(r)) next
  cat(sprintf("%6.2f  %10.2f  %12.4f  %12.4f  %12.4f\n",
              r$theta, r$tau, r$mlik_int, r$mlik_gauss,
              r$mlik_int - r$mlik_gauss))
}
