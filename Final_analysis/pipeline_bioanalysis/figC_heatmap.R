#!/usr/bin/env Rscript
# Fig C: 26 AD isoform switches — multi-evidence summary heatmap
# Nature Methods double-column (180 mm) format
# Packages: ggplot2, dplyr, RColorBrewer, tibble

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(RColorBrewer)
  library(tibble)
})

OUT_DIR <- tryCatch({
  args <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", args[grep("^--file=", args)])
  file.path(dirname(normalizePath(f, mustWork = FALSE)), "outputs")
}, error = function(e) "outputs")

# ─── 1. Data ──────────────────────────────────────────────────────────────────
# Tier A : novel isoform (NIC/NNIC) + M11 SUPPORTED
# Tier B : M11 SUPPORTED (known isoforms)
# Tier C : M11 UNSUPPORTED (structural domain change only)
# Within each tier, sorted by |DIFFUSE Δ| descending

df <- tribble(
  ~gene,     ~cell_type,         ~tier, ~delta,  ~n_lost, ~n_gained, ~m11,          ~ad_phyloP, ~ct_phyloP,
  # ── Tier A ────────────────────────────────────────────────────────────────
  "DLG1",    "OPC",              "A",    0.857,    3,  0,  "SUPPORTED",   NA,       2.507,
  "KIF21B",  "Excitatory",       "A",   -0.855,   3,  4,  "SUPPORTED",   4.067,    3.842,
  "NDUFS4",  "Excitatory",       "A",   -0.563,   0,  1,  "SUPPORTED",   2.263,    NA,
  # ── Tier B ────────────────────────────────────────────────────────────────
  "IFT122",  "Excitatory",       "B",    0.954,   4,  3,  "SUPPORTED",   4.826,    4.673,
  "FANCA",   "Excitatory",       "B",    0.946,   1,  0,  "SUPPORTED",  -0.493,    1.321,
  "DMD",     "Inhibitory",       "B",    0.919,   2,  1,  "SUPPORTED",   4.823,    3.350,
  "SYNE1",   "Inhibitory",       "B",    0.839,   1,  0,  "SUPPORTED",   3.450,    4.228,
  "RGS3",    "Astrocyte",        "B",    0.806,   5,  0,  "SUPPORTED",   NA,       2.687,
  "ADGRB2",  "Inhibitory",       "B",    0.800,   0,  1,  "SUPPORTED",   0.075,    1.935,
  "BSG",     "Oligodendrocyte",  "B",    0.800,   2,  0,  "SUPPORTED",  -0.473,   -0.051,
  "PTPRS",   "Astrocyte",        "B",    0.789,   1,  1,  "SUPPORTED",   NA,       3.919,
  "PTPRF",   "Inhibitory",       "B",    0.729,  10,  8,  "SUPPORTED",   2.835,    4.341,
  "SNTG1",   "Inhibitory",       "B",    0.702,   0,  1,  "SUPPORTED",   4.558,    NA,
  # ── Tier C ────────────────────────────────────────────────────────────────
  "ZCCHC17", "Oligodendrocyte",  "C",    0.965,   1,  0,  "UNSUPPORTED", 2.201,    4.169,
  "PML",     "Excitatory",       "C",    0.850,   1,  0,  "UNSUPPORTED", 0.508,    NA,
  "CCAR1",   "Inhibitory",       "C",    0.849,   0,  1,  "UNSUPPORTED", 4.204,    NA,
  "MTHFD1",  "OPC",              "C",    0.821,   0,  2,  "UNSUPPORTED", 3.864,    2.591,
  "ZNF268",  "Microglia",        "C",    0.795,   1,  1,  "UNSUPPORTED", 0.468,    NA,
  "FRMD4A",  "Excitatory",       "C",    0.788,   3,  0,  "UNSUPPORTED", 1.834,    4.534,
  "ZNF623",  "Oligodendrocyte",  "C",    0.778,   0,  1,  "UNSUPPORTED", NA,      -0.186,
  "LRPPRC",  "Oligodendrocyte",  "C",    0.758,   2,  0,  "UNSUPPORTED", 2.418,    0.096,
  "IFI16",   "Oligodendrocyte",  "C",    0.757,   2,  0,  "UNSUPPORTED",-0.089,   -0.440,
  "ASXL3",   "Excitatory",       "C",    0.749,   2,  0,  "UNSUPPORTED", 2.827,    1.104,
  "GOLGB1",  "Astrocyte",        "C",    0.735,   5,  0,  "UNSUPPORTED", 0.428,    1.481,
  "DOCK11",  "Inhibitory",       "C",    0.717,   6,  0,  "UNSUPPORTED", 3.250,    3.940,
  "ANKRD44", "Oligodendrocyte",  "C",    0.709,   0,  1,  "UNSUPPORTED", 3.712,    3.645
)

# Row ordering: Tier A→B→C, then |delta| desc within tier
df <- df %>%
  mutate(tier = factor(tier, levels = c("A","B","C")),
         abs_delta = abs(delta)) %>%
  arrange(tier, desc(abs_delta)) %>%
  mutate(y = rev(row_number()))  # top row = largest y value

N <- nrow(df)   # 26

# ─── 2. Palettes & helpers ────────────────────────────────────────────────────

CELL_COLS <- c(
  Excitatory      = "#C0392B",
  Inhibitory      = "#2475B0",
  OPC             = "#7D3C98",
  Astrocyte       = "#1E8449",
  Oligodendrocyte = "#CA6F1E",
  Microglia       = "#626567"
)

TIER_COLS  <- c(A = "#C0392B", B = "#D35400", C = "#808B96")
TIER_LABEL <- c(A = "Tier A\n(novel isoform)", B = "Tier B\n(PPI-supported)", C = "Tier C\n(structural only)")

# Sequential: lo_col → hi_col, x ∈ [lo, hi]
seq_pal <- function(x, lo, hi, col_lo, col_hi, na_col = "#E8E8E8") {
  pal  <- colorRampPalette(c(col_lo, col_hi))(200)
  norm <- pmin(pmax((x - lo)/(hi - lo), 0), 1)
  idx  <- pmax(round(norm * 199) + 1, 1)
  out  <- pal[idx]
  out[is.na(x)] <- na_col
  out
}

# Diverging: lo_col → mid_col → hi_col, x ∈ [lo, hi], midpoint at mid_x
div_pal <- function(x, lo, hi, mid_x = 0,
                    col_lo = "#C0392B", col_mid = "#FDFEFE", col_hi = "#1E8449",
                    na_col = "#E8E8E8") {
  pal  <- colorRampPalette(c(col_lo, col_mid, col_hi))(400)
  norm <- pmin(pmax((x - lo)/(hi - lo), 0), 1)
  idx  <- pmax(round(norm * 399) + 1, 1)
  out  <- pal[idx]
  out[is.na(x)] <- na_col
  out
}

# Auto text color (black on light bg, white on dark bg)
auto_txt <- function(hex_col) {
  hex_col[is.na(hex_col)] <- "#E8E8E8"
  m   <- col2rgb(hex_col) / 255
  lum <- 0.2126 * m[1,] + 0.7152 * m[2,] + 0.0722 * m[3,]
  ifelse(lum > 0.42, "#2C3E50", "#FFFFFF")
}

# ─── 3. Compute tile fills & labels ──────────────────────────────────────────

# Column 1: |DIFFUSE Δ|
f1  <- seq_pal(abs(df$delta), 0.50, 1.00, "#F3EEF9", "#5B2C8D")
l1  <- sprintf("%.3f", df$delta)

# Column 2: Domains Lost
f2  <- seq_pal(df$n_lost,    0, 10, "#FEF9F9", "#922B21")
l2  <- ifelse(df$n_lost  == 0, "–", as.character(df$n_lost))

# Column 3: Domains Gained
f3  <- seq_pal(df$n_gained,  0,  8, "#EBF5FB", "#1A5276")
l3  <- ifelse(df$n_gained == 0, "–", as.character(df$n_gained))

# Column 4: PPI Support (binary)
f4  <- ifelse(df$m11 == "SUPPORTED", "#1A7A4A", "#E8E8E8")
l4  <- ifelse(df$m11 == "SUPPORTED", "✔", "–")

# Column 5: AD-specific exon phyloP
f5  <- div_pal(df$ad_phyloP, -0.6, 5.2)
l5  <- ifelse(is.na(df$ad_phyloP), "N/A", sprintf("%.2f", df$ad_phyloP))

# Column 6: CT-specific exon phyloP
f6  <- div_pal(df$ct_phyloP, -0.6, 5.2)
l6  <- ifelse(is.na(df$ct_phyloP), "N/A", sprintf("%.2f", df$ct_phyloP))

# Assemble long-format tile data frame
make_col <- function(x_pos, fills, labels, gene, y_vals) {
  data.frame(x = x_pos, y = y_vals, fill_hex = fills,
             label = labels, gene = gene,
             stringsAsFactors = FALSE)
}
tile_df <- bind_rows(
  make_col(1, f1, l1, df$gene, df$y),
  make_col(2, f2, l2, df$gene, df$y),
  make_col(3, f3, l3, df$gene, df$y),
  make_col(4, f4, l4, df$gene, df$y),
  make_col(5, f5, l5, df$gene, df$y),
  make_col(6, f6, l6, df$gene, df$y)
)
tile_df$txt_col <- auto_txt(tile_df$fill_hex)

# ─── 4. Annotation frames ────────────────────────────────────────────────────

row_ann <- df %>%
  select(gene, cell_type, tier, y) %>%
  mutate(
    cell_hex  = CELL_COLS[cell_type],
    tier_hex  = TIER_COLS[as.character(tier)],
    ct_abbr   = case_when(
      cell_type == "Excitatory"      ~ "Exc",
      cell_type == "Inhibitory"      ~ "Inh",
      cell_type == "OPC"             ~ "OPC",
      cell_type == "Astrocyte"       ~ "Ast",
      cell_type == "Oligodendrocyte" ~ "Oli",
      TRUE                           ~ "Mic"
    )
  )

tier_ann <- df %>%
  group_by(tier) %>%
  summarise(y_lo = min(y) - 0.44, y_hi = max(y) + 0.44,
            y_mid = mean(y), .groups = "drop") %>%
  mutate(tier_hex = TIER_COLS[as.character(tier)],
         label    = TIER_LABEL[as.character(tier)])

# Tier separator y-positions (between A-B and B-C)
sep_y <- c(
  min(df$y[df$tier == "A"]) - 0.5,   # = 23.5
  min(df$y[df$tier == "B"]) - 0.5    # = 13.5
)

# Column header text
col_hdr <- data.frame(
  x     = 1:6,
  label = c("|DIFFUSE Δ|", "Domains\nLost", "Domains\nGained",
            "PPI\nSupport", "AD exon\nphyloP", "CT exon\nphyloP"),
  stringsAsFactors = FALSE
)

# Cell type legend (bottom strip)
ct_legend <- data.frame(
  label = names(CELL_COLS),
  col   = unname(CELL_COLS),
  x     = seq(0.8, 5.8, length.out = length(CELL_COLS)),
  y     = -1.2,
  stringsAsFactors = FALSE
)

# ─── 5. x-axis geometry ──────────────────────────────────────────────────────

X_CHIP  <- -0.15   # cell type chip
X_GENE  <- -0.90   # gene label right-align
X_BKT   <-  6.72  # tier bracket vertical bar
X_BLBL  <-  7.08  # tier bracket text

# ─── 6. Build figure ──────────────────────────────────────────────────────────

p <- ggplot() +

  # ── Tier background shading ──────────────────────────────────────────────
  geom_rect(
    data = tier_ann,
    aes(xmin = 0.48, xmax = 6.52, ymin = y_lo, ymax = y_hi,
        fill = I(tier_hex)),
    alpha = 0.055
  ) +

  # ── Tier horizontal separators ───────────────────────────────────────────
  geom_hline(
    yintercept = sep_y,
    color = "#AAAAAA", linewidth = 0.55, linetype = "solid"
  ) +

  # ── Main heatmap tiles ────────────────────────────────────────────────────
  geom_tile(
    data  = tile_df,
    aes(x = x, y = y, fill = I(fill_hex)),
    width = 0.93, height = 0.86,
    color = "white", linewidth = 0.30
  ) +

  # ── Cell value labels ─────────────────────────────────────────────────────
  geom_text(
    data  = tile_df,
    aes(x = x, y = y, label = label, color = I(txt_col)),
    size  = 2.50, fontface = "bold", lineheight = 0.88
  ) +

  # ── Gene name labels (left, tier-colored) ────────────────────────────────
  geom_text(
    data = row_ann,
    aes(x = X_GENE, y = y, label = gene, color = I(tier_hex)),
    hjust = 1, size = 3.05, fontface = "bold"
  ) +

  # ── Cell type chips ───────────────────────────────────────────────────────
  geom_label(
    data = row_ann,
    aes(x = X_CHIP, y = y, label = ct_abbr,
        fill  = I(cell_hex),
        color = I("white")),
    size          = 1.95, fontface = "bold",
    label.padding = unit(0.11, "lines"),
    label.r       = unit(0.08, "lines"),
    label.size    = 0
  ) +

  # ── Column headers ────────────────────────────────────────────────────────
  geom_text(
    data = col_hdr,
    aes(x = x, y = N + 1.55, label = label),
    size = 2.85, fontface = "bold", color = "#1C2833",
    vjust = 0, lineheight = 0.85
  ) +

  # ── Tier bracket: vertical bar ────────────────────────────────────────────
  geom_segment(
    data = tier_ann,
    aes(x = X_BKT, xend = X_BKT, y = y_lo, yend = y_hi,
        color = I(tier_hex)),
    linewidth = 1.15
  ) +

  # ── Tier bracket: top tick ────────────────────────────────────────────────
  geom_segment(
    data = tier_ann,
    aes(x = X_BKT - 0.08, xend = X_BKT + 0.08,
        y = y_hi, yend = y_hi,
        color = I(tier_hex)),
    linewidth = 0.80
  ) +

  # ── Tier bracket: bottom tick ─────────────────────────────────────────────
  geom_segment(
    data = tier_ann,
    aes(x = X_BKT - 0.08, xend = X_BKT + 0.08,
        y = y_lo, yend = y_lo,
        color = I(tier_hex)),
    linewidth = 0.80
  ) +

  # ── Tier bracket: label ───────────────────────────────────────────────────
  geom_text(
    data = tier_ann,
    aes(x = X_BLBL, y = y_mid, label = label, color = I(tier_hex)),
    hjust = 0, size = 2.65, fontface = "bold", lineheight = 0.88
  ) +

  # ── Cell type legend (bottom) ─────────────────────────────────────────────
  geom_label(
    data = ct_legend,
    aes(x = x, y = y, label = label,
        fill  = I(col),
        color = I("white")),
    size          = 2.05, fontface = "bold",
    label.padding = unit(0.14, "lines"),
    label.r       = unit(0.1,  "lines"),
    label.size    = 0
  ) +
  annotate("text", x = 0.1, y = -1.2, label = "Cell type:",
           hjust = 1, size = 2.4, color = "#555555", fontface = "plain") +

  # ── Column group header underlines ───────────────────────────────────────
  # DIFFUSE
  annotate("segment", x = 0.52, xend = 1.48, y = N + 0.72, yend = N + 0.72,
           color = "#5B2C8D", linewidth = 0.7) +
  # Domain change (cols 2–3)
  annotate("segment", x = 1.52, xend = 3.48, y = N + 0.72, yend = N + 0.72,
           color = "#666666", linewidth = 0.7) +
  annotate("text", x = 2.5, y = N + 1.08, label = "Domain change",
           size = 2.7, color = "#555555", fontface = "italic") +
  # PPI
  annotate("segment", x = 3.52, xend = 4.48, y = N + 0.72, yend = N + 0.72,
           color = "#1A7A4A", linewidth = 0.7) +
  # Conservation (cols 5–6)
  annotate("segment", x = 4.52, xend = 6.48, y = N + 0.72, yend = N + 0.72,
           color = "#1F618D", linewidth = 0.7) +
  annotate("text", x = 5.5, y = N + 1.08, label = "Exon conservation (phyloP)",
           size = 2.7, color = "#555555", fontface = "italic") +

  # ── Scales & coord ────────────────────────────────────────────────────────
  scale_x_continuous(limits = c(X_GENE - 0.45, X_BLBL + 1.05), expand = c(0, 0)) +
  scale_y_continuous(limits = c(-2.0, N + 3.2),  expand = c(0, 0)) +
  coord_cartesian(clip = "off") +

  # ── Theme ─────────────────────────────────────────────────────────────────
  theme_void(base_size = 9) +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    plot.margin     = margin(6, 5, 6, 5, unit = "mm"),
    plot.title      = element_text(
      size = 9, face = "bold", color = "#1A1A2E",
      margin = margin(b = 3, unit = "mm"), lineheight = 1.1
    ),
    plot.subtitle   = element_text(
      size = 7.5, color = "#4A4A6A",
      margin = margin(b = 4, unit = "mm"), lineheight = 1.25
    ),
    plot.caption    = element_text(
      size = 6.2, color = "#777777", hjust = 0, lineheight = 1.35,
      margin = margin(t = 4, unit = "mm")
    )
  ) +

  labs(
    title = paste0(
      "Figure C  |  BISECT: 26 AD isoform switches with multi-evidence annotation\n",
      "(53 input cases → 26 Stage 2 PASS → 13 M11 SUPPORTED)"
    ),
    subtitle = paste0(
      "Rows ordered by evidence tier then |DIFFUSE Δ| descending.  ",
      "FANCA* and BSG* carry negative AD-exon phyloP (accelerated evolution)."
    ),
    caption = paste0(
      "|DIFFUSE Δ|: ESM-2 function prediction score gap (CT − AD), v15d Muscle model (AUPRC 0.703).  ",
      "Domains Lost/Gained: PFAM domain count from HMMER scan.  ",
      "PPI Support (M11): STRING DB combined score ≥0 at least one hypothesized partner (experimental + coexp channels).  ",
      "phyloP: mean 100-way vertebrate conservation of isoform-specific exons; negative = accelerated evolution.  ",
      "N/A: no isoform-specific exons detected or API call failed.  ",
      "Stage 2 criterion: ≥1 domain gained or lost between CT and AD isoforms."
    )
  )

# ─── 7. Save ─────────────────────────────────────────────────────────────────

fig_w <- 7.48   # 190 mm — comfortably double-column for Nature Methods
fig_h <- 11.02  # 280 mm

pdf_out <- file.path(OUT_DIR, "figC_26case_heatmap.pdf")
png_out <- file.path(OUT_DIR, "figC_26case_heatmap.png")

# cairo_pdf for proper Unicode rendering
tryCatch({
  ggsave(pdf_out, plot = p, width = fig_w, height = fig_h,
         device = cairo_pdf)
  cat(sprintf("Saved PDF: %s\n", pdf_out))
}, error = function(e) {
  ggsave(pdf_out, plot = p, width = fig_w, height = fig_h, useDingbats = FALSE)
  cat(sprintf("Saved PDF (fallback): %s\n", pdf_out))
})

ggsave(png_out, plot = p, width = fig_w, height = fig_h, dpi = 300, bg = "white")
cat(sprintf("Saved PNG: %s\n", png_out))
