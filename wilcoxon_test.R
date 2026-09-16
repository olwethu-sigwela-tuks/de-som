csv_file <- "results_30_runs_paired.csv"

if (!file.exists(csv_file)) {
  stop(paste("File not found:", csv_file, "- Please run ./ConvertToCSV first."))
}

df <- read.csv(csv_file)

alpha <- 0.05

cat("\n==================================================================\n")
cat("       WILCOXON SIGNED-RANK HYPOTHESIS TEST (30 PAIRED RUNS)       \n")
cat("==================================================================\n")
cat(sprintf("Significance Level (alpha): %.2f\n", alpha))
cat(sprintf("Sample Size (n):            %d runs\n\n", nrow(df)))

run_test <- function(metric_name, de_vec, som_vec) {
  wt <- wilcox.test(de_vec, som_vec, paired = TRUE, exact = FALSE)
  
  de_mean  <- mean(de_vec)
  de_sd    <- sd(de_vec)
  som_mean <- mean(som_vec)
  som_sd   <- sd(som_vec)
  
  is_sig <- wt$p.value < alpha
  winner <- if (!is_sig) "No Statistically Significant Difference"
            else if (de_mean < som_mean) "Differential Evolution (Lower Error)"
            else "Conventional SOM (Lower Error)"

  cat("------------------------------------------------------------------\n")
  cat(sprintf("Metric: %s\n", metric_name))
  cat("------------------------------------------------------------------\n")
  cat(sprintf("  DE   Mean (SD):   %.6f (%.6f)\n", de_mean, de_sd))
  cat(sprintf("  SOM  Mean (SD):   %.6f (%.6f)\n", som_mean, som_sd))
  cat(sprintf("  V-Statistic:      %.2f\n", wt$statistic))
  cat(sprintf("  p-value:          %.6e\n", wt$p.value))
  cat(sprintf("  Significant?      %s (p < %.2f)\n", ifelse(is_sig, "YES", "NO"), alpha))
  cat(sprintf("  Conclusion:       %s\n\n", winner))
  
  return(wt)
}

wt_test_qe  <- run_test("Test Quantization Error",  df$de_test_qe,  df$som_test_qe)
wt_test_te  <- run_test("Test Topographic Error",   df$de_test_te,  df$som_test_te)
wt_train_qe <- run_test("Train Quantization Error", df$de_train_qe, df$som_train_qe)
wt_train_te <- run_test("Train Topographic Error",  df$de_train_te, df$som_train_te)

cat("==================================================================\n")
cat("                       END OF ANALYSIS                            \n")
cat("==================================================================\n")
