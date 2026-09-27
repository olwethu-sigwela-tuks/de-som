data_file <- "results/glass/best_params_de.txt"
lines <- readLines(data_file)
records <- list()

for (line in lines) {
  if (grepl("^NP:", line)) {
    parts <- strsplit(line, "\\|")[[1]]
    kv <- list()
    for (p in parts) {
      item <- trimws(p)
      s <- strsplit(item, ":")[[1]]
      if (length(s) == 2) {
        kv[[trimws(s[1])]] <- as.numeric(trimws(s[2]))
      }
    }
    records[[length(records) + 1]] <- data.frame(
      NP = kv[["NP"]],
      Gen = kv[["Gen"]],
      Neurons = kv[["Neurons"]],
      CR = kv[["CR"]],
      F = kv[["F"]],
      TrainError = kv[["Train Error"]],
      TestError = kv[["5-Fold Test Error"]]
    )
  }
}

df <- do.call(rbind, records)

cr_vals <- sort(unique(df$CR))
f_vals  <- sort(unique(df$F))
gen_vals <- sort(unique(df$Gen))

get_matrix <- function(gen_val, metric="TestError") {
  sub_df <- df[df$Gen == gen_val, ]
  mat <- matrix(NA, nrow=length(cr_vals), ncol=length(f_vals))
  for (i in 1:length(cr_vals)) {
    for (j in 1:length(f_vals)) {
      val <- sub_df[sub_df$CR == cr_vals[i] & sub_df$F == f_vals[j], metric]
      if (length(val) > 0) mat[i, j] <- val[1]
    }
  }
  return(mat)
}

color_palette <- colorRampPalette(c("#1a9850", "#91cf60", "#d9ef8b", "#fee08b", "#fc8d59", "#d73027"))

out_png <- "results/glass/de_hyperparameters_3d_glass.png"
png(out_png, width=2400, height=1600, res=200, bg="white")

par(mfrow=c(2, 3), mar=c(4, 4, 3, 2), oma=c(1, 1, 4, 1))

for (g in gen_vals) {
  z <- get_matrix(g, "TestError")
  
  nrz <- nrow(z)
  ncz <- ncol(z)
  zfacet <- (z[-1, -1] + z[-1, -ncz] + z[-nrz, -1] + z[-nrz, -ncz]) / 4
  facet_col <- color_palette(100)[as.numeric(cut(zfacet, 100))]
  
  res <- persp(
    x = cr_vals,
    y = f_vals,
    z = z,
    theta = 35,
    phi = 25,
    expand = 0.75,
    ltheta = 120,
    shade = 0.4,
    col = facet_col,
    border = "#2b2b2b",
    lwd = 1.2,
    ticktype = "detailed",
    xlab = "CR (Crossover Rate)",
    ylab = "F (Scaling Factor)",
    zlab = "5-Fold Test QE",
    main = sprintf("3D Error Surface: Gen = %d", g),
    cex.lab = 1.1,
    cex.axis = 0.85,
    zlim = c(0.25, 0.32)
  )
  
  opt_sub <- df[df$Gen == g, ]
  best_idx <- which.min(opt_sub$TestError)
  best_pt <- opt_sub[best_idx, ]
  
  pt_3d <- trans3d(best_pt$CR, best_pt$F, best_pt$TestError, pmat=res)
  points(pt_3d$x, pt_3d$y, col="blue", pch=19, cex=2.0)
  text(pt_3d$x, pt_3d$y, labels=sprintf("  Min: %.4f\n  (CR=%.3f, F=%.2f)", best_pt$TestError, best_pt$CR, best_pt$F), 
       pos=3, col="blue", cex=0.9, font=2)
}

for (g in gen_vals) {
  z <- get_matrix(g, "TestError")
  
  image(
    x = cr_vals,
    y = f_vals,
    z = z,
    col = rev(color_palette(50)),
    xlab = "CR (Crossover Rate)",
    ylab = "F (Scaling Factor)",
    main = sprintf("Contour Landscape: Gen = %d", g),
    cex.lab = 1.1,
    cex.axis = 0.9
  )
  contour(x = cr_vals, y = f_vals, z = z, add = TRUE, nlevels = 8, labcex = 0.8, col = "#222222", lwd = 1.2)
  
  opt_sub <- df[df$Gen == g, ]
  best_idx <- which.min(opt_sub$TestError)
  best_pt <- opt_sub[best_idx, ]
  points(best_pt$CR, best_pt$F, pch=8, col="white", cex=2.2, lwd=2.5)
  points(best_pt$CR, best_pt$F, pch=1, col="black", cex=2.5, lwd=2)
  text(best_pt$CR, best_pt$F, labels=sprintf("Min: %.4f", best_pt$TestError), pos=1, col="black", font=2, cex=0.85)
}

mtext("Differential Evolution SOM: 3D Hyperparameter Response Landscape on Glass Dataset", 
      outer=TRUE, cex=1.5, font=2, line=1.5)

dev.off()

cat(sprintf("Generated 3D visualization: %s\n", out_png))
