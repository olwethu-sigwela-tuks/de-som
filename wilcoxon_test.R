args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  cat("Usage: Rscript wilcoxon_test.R <csv_file>\n")
  quit(status = 1)
}

csv_file <- args[1]
data <- read.csv(csv_file)

de_data <- subset(data, Method == "DE")
som_data <- subset(data, Method == "SOM")

cat("==================================================\n")
cat("Wilcoxon Signed-Rank Test: DE-SOM vs SOM_PAK\n")
cat("Dataset CSV:", csv_file, "\n")
cat("==================================================\n\n")

# 1. Test Quantization Error (QE)
qe_test <- wilcox.test(de_data$Test_QE, som_data$Test_QE, paired = TRUE, exact = FALSE)
cat("--- Test Quantization Error (QE) ---\n")
cat(sprintf("DE-SOM  Median: %.6f (IQR: %.6f)\n", median(de_data$Test_QE), IQR(de_data$Test_QE)))
cat(sprintf("SOM_PAK Median: %.6f (IQR: %.6f)\n", median(som_data$Test_QE), IQR(som_data$Test_QE)))
cat(sprintf("p-value: %.5e\n", qe_test$p.value))
if (qe_test$p.value < 0.05) {
  cat("Result: Statistically significant difference (p < 0.05)\n\n")
} else {
  cat("Result: No statistically significant difference (p >= 0.05)\n\n")
}

# 2. Test Topographic Error (TE)
te_test <- wilcox.test(de_data$Test_TE, som_data$Test_TE, paired = TRUE, exact = FALSE)
cat("--- Test Topographic Error (TE) ---\n")
cat(sprintf("DE-SOM  Median: %.6f (IQR: %.6f)\n", median(de_data$Test_TE), IQR(de_data$Test_TE)))
cat(sprintf("SOM_PAK Median: %.6f (IQR: %.6f)\n", median(som_data$Test_TE), IQR(som_data$Test_TE)))
cat(sprintf("p-value: %.5e\n", te_test$p.value))
if (te_test$p.value < 0.05) {
  cat("Result: Statistically significant difference (p < 0.05)\n\n")
} else {
  cat("Result: No statistically significant difference (p >= 0.05)\n\n")
}