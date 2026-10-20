args <- commandArgs(trailingOnly = TRUE)
dataset <- ifelse(length(args) > 0, args[1], "iris")

csv_candidates <- c(
  sprintf("results/%s/runs_30_paired.csv", dataset),
  sprintf("../../results/%s/runs_30_paired.csv", dataset),
  "results_30_runs_paired.csv",
  "../../results_30_runs_paired.csv"
)

csv_file <- ""
for (f in csv_candidates) {
  if (file.exists(f)) {
    csv_file <- f
    break
  }
}

if (csv_file == "") {
  stop(paste("File not found for dataset:", dataset, sprintf("- Please run ./ConvertToCSV %s first.", dataset)))
}

df <- read.csv(csv_file)

alpha <- 0.05

cat("\n==================================================================\n")
cat(sprintf("       WILCOXON SIGNED-RANK HYPOTHESIS TEST: %s (30 PAIRED RUNS)       \n", toupper(dataset)))
cat("==================================================================\n")
cat(sprintf("Dataset:                    %s\n", dataset))
cat(sprintf("CSV Source:                 %s\n", csv_file))
cat(sprintf("Significance Level (alpha): %.2f\n", alpha))
cat(sprintf("Sample Size (n):            %d runs\n\n", nrow(df)))

run_test <- function(metric_name, de_vec, som_vec, higher_is_better = FALSE) {
  wt <- wilcox.test(de_vec, som_vec, paired = TRUE, exact = FALSE)
  
  de_mean  <- mean(de_vec)
  de_sd    <- sd(de_vec)
  som_mean <- mean(som_vec)
  som_sd   <- sd(som_vec)
  
  is_sig <- wt$p.value < alpha
  winner <- if (!is_sig) {
    "No Statistically Significant Difference"
  } else if (!higher_is_better) {
    if (de_mean < som_mean) "Differential Evolution (Lower Error)" else "Conventional SOM (Lower Error)"
  } else {
    if (de_mean > som_mean) "Differential Evolution (Higher Score)" else "Conventional SOM (Higher Score)"
  }

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

wt_test_qe   <- run_test("Test Quantization Error",  df$de_test_qe,   df$som_test_qe,   higher_is_better = FALSE)
wt_test_te   <- run_test("Test Topographic Error",   df$de_test_te,   df$som_test_te,   higher_is_better = FALSE)

if ("de_accuracy" %in% colnames(df)) {
  wt_test_acc  <- run_test("Test Accuracy",           df$de_accuracy,  df$som_accuracy,  higher_is_better = TRUE)
  wt_test_prec <- run_test("Test Macro Precision",    df$de_precision, df$som_precision, higher_is_better = TRUE)
  wt_test_rec  <- run_test("Test Macro Recall",       df$de_recall,    df$som_recall,    higher_is_better = TRUE)
  wt_test_f1   <- run_test("Test Macro F1-Score",     df$de_f1,        df$som_f1,        higher_is_better = TRUE)
}

cat("==================================================================\n")
cat("                       END OF ANALYSIS                            \n")
cat("==================================================================\n")
