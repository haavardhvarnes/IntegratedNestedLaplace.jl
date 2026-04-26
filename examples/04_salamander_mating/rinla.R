#!/usr/bin/env Rscript
# Salamander mating: R-INLA reference fit.
#
# Phase 1 fixture — uses the simplified formula that matches what the current
# Julia driver can target (intercept + Cross fixed effects + IID female / male
# random effects with shared log-Gamma precision prior). The canonical
# https://r-inla.org/examples/salamander/salamander/salamander.html fit uses
# iid2d blocks per experiment + a Wishart prior; that target is reserved for
# Phase 2 once BivariateIIDModel is wired through the driver.

suppressPackageStartupMessages({
  library(INLA)
  library(jsonlite)
})

example_id <- "04_salamander_mating"
this_script <- normalizePath(sub("--file=", "", grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)[1]))
root <- normalizePath(file.path(dirname(this_script), "..", ".."))
data_path <- file.path(root, "examples", example_id, "data", "salamander.csv")
out_dir <- file.path(root, "test", "fixtures", example_id)
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
out_path <- file.path(out_dir, "rinla_reference.json")

stopifnot(file.exists(data_path))
df <- read.csv(data_path, stringsAsFactors = FALSE)
df$Female <- as.factor(df$Female)
df$Male   <- as.factor(df$Male)
df$Cross  <- as.factor(df$Cross)
stopifnot(nrow(df) == 360)

# Default INLA priors:
#   - Fixed effects: N(0, 1000) (precision 0.001)
#   - f(., model = "iid") precision: log-Gamma(1, 5e-5) by default.
formula <- Mate ~ 1 + Cross +
  f(Female, model = "iid") +
  f(Male,   model = "iid")

run <- function(verbose = FALSE) {
  t0 <- Sys.time()
  res <- inla(formula, family = "binomial", data = df,
              Ntrials = rep(1, nrow(df)),
              control.compute = list(return.marginals = FALSE),
              verbose = verbose, silent = 2L)
  t1 <- Sys.time()
  list(res = res, wall = as.numeric(t1 - t0, units = "secs"))
}

# Cold + warm runs (we record the warm one).
invisible(run())
fit <- run()

cat(sprintf("inla wall (warm)         = %.4f s\n", fit$wall))
cat(sprintf("inla cpu.used Total      = %.4f s\n", as.numeric(fit$res$cpu.used["Total"])))

named_summary <- function(s) {
  cols <- c("mean", "sd", "0.025quant", "0.5quant", "0.975quant")
  out <- list()
  for (i in seq_len(nrow(s))) {
    out[[rownames(s)[i]]] <- as.list(s[i, cols, drop = TRUE])
    names(out[[rownames(s)[i]]]) <- cols
  }
  out
}

ref <- list(
  example_id   = example_id,
  formula      = deparse(formula),
  inla_version = as.character(INLA::inla.version("version")),
  n            = nrow(df),
  fixed        = named_summary(fit$res$summary.fixed),
  hyper        = named_summary(fit$res$summary.hyperpar),
  cpu          = as.list(fit$res$cpu.used),
  wall_seconds = fit$wall
)

write_json(ref, out_path, pretty = TRUE, auto_unbox = TRUE, digits = 10)
cat(sprintf("wrote %s\n", out_path))
