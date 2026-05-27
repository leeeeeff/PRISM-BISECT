#!/usr/bin/env Rscript
# Fig A: BISECT pipeline funnel + DIFFUSE Δ vs STRING score scatter
# Nature Methods double-column (190 mm wide, 120 mm tall)
# Packages: ggplot2, dplyr, tibble, RColorBrewer, grid (base R)

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tibble)
  library(grid)
})

OUT_DIR <- tryCatch({
  args <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", args[grep("^--file=", args)])
  file.path(dirname(normalizePath(f, mustWork = FALSE)), "outputs")
}, error = function(e) "outputs")

# ─── Data ─────────────────────────────────────────────────────────────────────

df <- tribble(
  ~gene,     ~cell_type,         ~tier, ~delta,  ~n_lost, ~n_gained, ~verdict,      ~string_top,
  "DLG1",    "OPC",              "A",    0.857,   3,  0,  "SUPPORTED",    999,
  "KIF21B",  "Excitatory",       "A",   -0.855,  3,  4,  "SUPPORTED",    765,
  "NDUFS4",  "Excitatory",       "A",   -0.563,  0,  1,  "SUPPORTED",    999,
  "IFT122",  "Excitatory",       "B",    0.954,  4,  3,  "SUPPORTED",    999,
  "FANCA",   "Excitatory",       "B",    0.946,  1,  0,  "SUPPORTED",    999,
  "DMD",     "Inhibitory",       "B",    0.919,  2,  1,  "SUPPORTED",    999,
  "SYNE1",   "Inhibitory",       "B",    0.839,  1,  0,  "SUPPORTED",    999,
  "RGS3",    "Astrocyte",        "B",    0.806,  5,  0,  "SUPPORTED",    996,
  "ADGRB2",  "Inhibitory",       "B",    0.800,  0,  1,  "SUPPORTED",    900,
  "BSG",     "Oligodendrocyte",  "B",    0.800,  2,  0,  "SUPPORTED",    999,
  "PTPRS",   "Astrocyte",        "B",    0.789,  1,  1,  "SUPPORTED",    999,
  "PTPRF",   "Inhibitory",       "B",    0.729, 10,  8,  "SUPPORTED",    997,
  "SNTG1",   "Inhibitory",       "B",    0.702,  0,  1,  "SUPPORTED",    992,
  "ZCCHC17", "Oligodendrocyte",  "C",    0.965,  1,  0,  "UNSUPPORTED",  760,
  "PML",     "Excitatory",       "C",    0.850,  1,  0,  "UNSUPPORTED",  999,
  "CCAR1",   "Inhibitory",       "C",    0.849,  0,  1,  "UNSUPPORTED",  935,
  "MTHFD1",  "OPC",              "C",    0.821,  0,  2,  "UNSUPPORTED",  999,
  "ZNF268",  "Microglia",        "C",    0.795,  1,  1,  "UNSUPPORTED",  747,
  "FRMD4A",  "Excitatory",       "C",    0.788,  3,  0,  "UNSUPPORTED",  919,
  "ZNF623",  "Oligodendrocyte",  "C",    0.778,  0,  1,  "UNSUPPORTED",  520,
  "LRPPRC",  "Oligodendrocyte",  "C",    0.758,  2,  0,  "UNSUPPORTED",  999,
  "IFI16",   "Oligodendrocyte",  "C",    0.757,  2,  0,  "UNSUPPORTED",  988,
  "ASXL3",   "Excitatory",       "C",    0.749,  2,  0,  "UNSUPPORTED",  917,
  "GOLGB1",  "Astrocyte",        "C",    0.735,  5,  0,  "UNSUPPORTED",  997,
  "DOCK11",  "Inhibitory",       "C",    0.717,  6,  0,  "UNSUPPORTED",  937,
  "ANKRD44", "Oligodendrocyte",  "C",    0.709,  0,  1,  "UNSUPPORTED",  964
) %>%
  mutate(
    abs_delta   = abs(delta),
    n_change    = n_lost + n_gained,
    tier        = factor(tier, levels = c("A","B","C")),
    verdict_fac = factor(verdict, levels = c("SUPPORTED","UNSUPPORTED"))
  )

# Label set: Tier A + PTPRF (extreme domain) + KIF21B (lowest SUPPORTED STRING)
LABEL_GENES <- c("DLG1","KIF21B","NDUFS4","IFT122","FANCA","PTPRF")

# Manual nudges to avoid overlap (x, y in data units)
nudges <- tribble(
  ~gene,     ~nx,    ~ny,
  "DLG1",     0.003, -35,
  "KIF21B",   0.003, -35,
  "NDUFS4",  -0.003, -35,
  "IFT122",   0.000, -35,
  "FANCA",   -0.003, -35,
  "PTPRF",   -0.030,  30
)

label_df <- df %>%
  filter(gene %in% LABEL_GENES) %>%
  left_join(nudges, by = "gene")

# ─── Colour constants ─────────────────────────────────────────────────────────

COL_SUPP   <- "#1A7A4A"
COL_UNSUPP <- "#AAB7B8"
COL_A      <- "#C0392B"
COL_B      <- "#D35400"
COL_C      <- "#808B96"

TIER_COLS  <- c(A = COL_A, B = COL_B, C = COL_C)
VERDICT_COLS <- c(SUPPORTED = COL_SUPP, UNSUPPORTED = COL_UNSUPP)

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL A — Pipeline funnel
# ═══════════════════════════════════════════════════════════════════════════════

# Coordinate system: x ∈ [0,1], y ∈ [0,10] (top = 10)
# Three boxes + two arrows + criterion text

BOX_W <- 0.62; BOX_CX <- 0.50
B1 <- c(y_lo = 7.9,  y_hi = 9.9,  n = 53,  label = "53 input cases",       sub = "DTU-significant, |DIFFUSE Δ| ranked",  col = "#E8EAF6")
B2 <- c(y_lo = 4.3,  y_hi = 6.3,  n = 26,  label = "26  Stage 2 PASS",     sub = "≥1 PFAM domain gained or lost",    col = "#E3F2FD")
B3 <- c(y_lo = 0.7,  y_hi = 2.7,  n = 13,  label = "13  M11 SUPPORTED",    sub = "STRING PPI hypothesis confirmed",       col = "#E8F5E9")

boxes <- bind_rows(
  as_tibble(as.list(B1)) %>% mutate(border = "#9FA8DA"),
  as_tibble(as.list(B2)) %>% mutate(border = "#42A5F5"),
  as_tibble(as.list(B3)) %>% mutate(border = "#66BB6A")
) %>%
  mutate(across(c(y_lo, y_hi, n), as.numeric),
         y_mid = (y_lo + y_hi) / 2)

# Arrow segments and criterion labels
arrows_df <- data.frame(
  y_start = c(as.numeric(B1["y_lo"]) - 0.02, as.numeric(B2["y_lo"]) - 0.02),
  y_end   = c(as.numeric(B2["y_hi"]) + 0.02, as.numeric(B3["y_hi"]) + 0.02),
  crit    = c("Stage 2 filter\n(domain change ≥1)",
              "M11 filter\n(STRING PPI validation)")
)

# Stage 2 cell-type mini bars within box 2
ct_stage2 <- data.frame(
  ct  = c("Excitatory","Inhibitory","Oligodendrocyte","Astrocyte","OPC","Microglia"),
  n   = c(7, 7, 6, 3, 2, 1),
  col = c("#C0392B","#2475B0","#CA6F1E","#1E8449","#7D3C98","#626567"),
  stringsAsFactors = FALSE
) %>% mutate(frac = n / sum(n))

CT_BAR_Y  <- 4.55  # bottom of bar strip
CT_BAR_H  <- 0.28
CT_BAR_X0 <- BOX_CX - BOX_W/2 + 0.02
CT_BAR_X1 <- BOX_CX + BOX_W/2 - 0.02
CT_BAR_W  <- CT_BAR_X1 - CT_BAR_X0

ct_stage2 <- ct_stage2 %>%
  mutate(x_lo = CT_BAR_X0 + cumsum(c(0, head(frac,-1))) * CT_BAR_W,
         x_hi = x_lo + frac * CT_BAR_W)

# Tier mini bars within box 3
tier_b3 <- data.frame(
  tier = c("A","B","C"), n = c(3,10,0), col = unname(TIER_COLS[c("A","B","C")]),
  stringsAsFactors = FALSE
) %>% filter(n > 0) %>% mutate(frac = n / sum(n))

TIER_BAR_Y  <- 0.95
TIER_BAR_H  <- 0.28
tier_b3 <- tier_b3 %>%
  mutate(x_lo = CT_BAR_X0 + cumsum(c(0, head(frac,-1))) * CT_BAR_W,
         x_hi = x_lo + frac * CT_BAR_W)

p_funnel <- ggplot() +

  # ── Boxes ──────────────────────────────────────────────────────────────────
  geom_rect(
    data = boxes,
    aes(xmin = BOX_CX - BOX_W/2, xmax = BOX_CX + BOX_W/2,
        ymin = y_lo, ymax = y_hi, fill = I(col)),
    color = NA
  ) +
  geom_rect(
    data = boxes,
    aes(xmin = BOX_CX - BOX_W/2, xmax = BOX_CX + BOX_W/2,
        ymin = y_lo, ymax = y_hi),
    fill = NA, color = boxes$border, linewidth = 0.6
  ) +

  # ── Count numbers ───────────────────────────────────────────────────────────
  geom_text(
    data = boxes,
    aes(x = BOX_CX, y = y_mid + 0.35, label = as.character(n)),
    size = 10, fontface = "bold", color = "#1A1A2E"
  ) +
  geom_text(
    data = boxes,
    aes(x = BOX_CX, y = y_mid - 0.25, label = label),
    size = 3.2, fontface = "bold", color = "#2C3E50"
  ) +
  geom_text(
    data = boxes,
    aes(x = BOX_CX, y = y_mid - 0.75, label = sub),
    size = 2.4, color = "#555555", fontface = "italic"
  ) +

  # ── Cell-type mini bar (box 2) ──────────────────────────────────────────────
  geom_rect(
    data = ct_stage2,
    aes(xmin = x_lo, xmax = x_hi,
        ymin = CT_BAR_Y, ymax = CT_BAR_Y + CT_BAR_H,
        fill = I(col)),
    color = "white", linewidth = 0.2
  ) +
  annotate("text", x = CT_BAR_X0, y = CT_BAR_Y + CT_BAR_H + 0.12,
           label = "by cell type", hjust = 0, size = 2.0, color = "#777777") +

  # ── Tier mini bar (box 3) ───────────────────────────────────────────────────
  geom_rect(
    data = tier_b3,
    aes(xmin = x_lo, xmax = x_hi,
        ymin = TIER_BAR_Y, ymax = TIER_BAR_Y + TIER_BAR_H,
        fill = I(col)),
    color = "white", linewidth = 0.2
  ) +
  annotate("text", x = CT_BAR_X0, y = TIER_BAR_Y + TIER_BAR_H + 0.12,
           label = "A: 3  B: 10", hjust = 0, size = 2.0, color = "#777777") +

  # ── Arrows ──────────────────────────────────────────────────────────────────
  geom_segment(
    data = arrows_df,
    aes(x = BOX_CX, xend = BOX_CX, y = y_start, yend = y_end),
    arrow = arrow(length = unit(0.10,"inches"), type = "closed"),
    color = "#555555", linewidth = 0.7
  ) +
  geom_label(
    data = arrows_df,
    aes(x = BOX_CX + 0.26, y = (y_start + y_end)/2, label = crit),
    size = 2.3, color = "#333333", fill = "white",
    label.padding = unit(0.12,"lines"), label.size = 0.3,
    label.r = unit(0.08,"lines"), lineheight = 0.9
  ) +

  # ── Loss annotation (funnel neck) ───────────────────────────────────────────
  annotate("text", x = BOX_CX - 0.33, y = (as.numeric(B1["y_lo"]) + as.numeric(B2["y_hi"]))/2,
           label = "27 filtered", size = 2.2, color = "#999999", hjust = 1, fontface = "italic") +
  annotate("text", x = BOX_CX - 0.33, y = (as.numeric(B2["y_lo"]) + as.numeric(B3["y_hi"]))/2,
           label = "13 filtered", size = 2.2, color = "#999999", hjust = 1, fontface = "italic") +

  scale_x_continuous(limits = c(0, 1.05), expand = c(0,0)) +
  scale_y_continuous(limits = c(0, 10.6), expand = c(0,0)) +
  coord_cartesian(clip = "off") +
  theme_void() +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    plot.margin     = margin(4, 2, 4, 4, "mm"),
    plot.title      = element_text(size = 8.5, face = "bold", color = "#1A1A2E",
                                   margin = margin(b=3, unit="mm"))
  ) +
  labs(title = "a  BISECT pipeline")

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL B — Scatter: |DIFFUSE Δ| vs STRING top score
# ═══════════════════════════════════════════════════════════════════════════════

# Background zone: high Δ + SUPPORTED → "high-confidence region"
DELTA_THRESH <- 0.80

p_scatter <- ggplot(df, aes(x = abs_delta, y = string_top)) +

  # ── Background zone ─────────────────────────────────────────────────────────
  annotate("rect",
           xmin = DELTA_THRESH, xmax = 1.02,
           ymin = 870, ymax = 1020,
           fill = "#E8F5E9", alpha = 0.7) +
  annotate("text",
           x = DELTA_THRESH + 0.005, y = 1010,
           label = "High-confidence\nzone",
           hjust = 0, vjust = 1, size = 2.5,
           color = "#1A7A4A", fontface = "italic", lineheight = 0.9) +

  # ── Reference line: Δ threshold ─────────────────────────────────────────────
  geom_vline(xintercept = DELTA_THRESH, linetype = "dashed",
             color = "#AAAAAA", linewidth = 0.5) +
  annotate("text", x = DELTA_THRESH - 0.005, y = 525,
           label = sprintf("|Delta| = %.2f", DELTA_THRESH),
           hjust = 1, size = 2.1, color = "#AAAAAA") +

  # ── Reference line: STRING threshold ────────────────────────────────────────
  geom_hline(yintercept = 700, linetype = "dashed",
             color = "#AAAAAA", linewidth = 0.5) +
  annotate("text", x = 0.455, y = 710,
           label = "STRING = 700", hjust = 0, size = 2.1, color = "#AAAAAA") +

  # ── Points ──────────────────────────────────────────────────────────────────
  geom_point(
    data = df %>% filter(verdict == "UNSUPPORTED"),
    aes(shape = tier, size = n_change),
    color = COL_UNSUPP, alpha = 0.75, stroke = 0.4
  ) +
  geom_point(
    data = df %>% filter(verdict == "SUPPORTED"),
    aes(shape = tier, size = n_change),
    color = COL_SUPP, alpha = 0.90, stroke = 0.5
  ) +

  # ── PTPRF special ring (extreme domain change) ───────────────────────────────
  geom_point(
    data = df %>% filter(gene == "PTPRF"),
    shape = 21, size = 8, fill = NA,
    color = COL_B, stroke = 0.9, alpha = 0.8
  ) +

  # ── Labels ──────────────────────────────────────────────────────────────────
  geom_text(
    data = label_df,
    aes(x = abs_delta + nx, y = string_top + ny,
        label = gene, color = I(TIER_COLS[as.character(tier)])),
    size = 2.6, fontface = "bold.italic", hjust = 0.5
  ) +

  # ── Legends ─────────────────────────────────────────────────────────────────
  scale_shape_manual(
    name   = "Tier",
    values = c(A = 17, B = 16, C = 15),
    labels = c(A = "A (novel isoform)", B = "B (PPI-supported)", C = "C (structural only)")
  ) +
  scale_size_continuous(
    name   = "Domain\nchanges",
    range  = c(2.0, 7.0),
    breaks = c(1, 5, 10, 18)
  ) +

  # ── Manual verdict legend ────────────────────────────────────────────────────
  annotate("point", x = 0.462, y = 650, shape = 16, size = 3.5, color = COL_SUPP) +
  annotate("text",  x = 0.475, y = 650, label = "M11 SUPPORTED (n=13)",
           hjust = 0, size = 2.5, color = COL_SUPP, fontface = "bold") +
  annotate("point", x = 0.462, y = 615, shape = 16, size = 3.5, color = COL_UNSUPP) +
  annotate("text",  x = 0.475, y = 615, label = "M11 UNSUPPORTED (n=13)",
           hjust = 0, size = 2.5, color = "#777777") +

  # ── Axes & theme ────────────────────────────────────────────────────────────
  scale_x_continuous(
    name   = "|DIFFUSE Δ|  (CT vs AD isoform prediction gap)",
    limits = c(0.45, 1.02), breaks = seq(0.5, 1.0, 0.1),
    expand = c(0, 0)
  ) +
  scale_y_continuous(
    name   = "STRING top combined score (M11)",
    limits = c(500, 1045), breaks = c(500, 600, 700, 800, 900, 1000),
    expand = c(0, 0)
  ) +
  coord_cartesian(clip = "off") +
  theme_classic(base_size = 9) +
  theme(
    plot.background  = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white"),
    panel.grid.major = element_line(color = "#F0F0F0", linewidth = 0.3),
    axis.title       = element_text(size = 8, color = "#333333"),
    axis.text        = element_text(size = 7.5, color = "#444444"),
    axis.line        = element_line(color = "#888888", linewidth = 0.4),
    axis.ticks       = element_line(color = "#888888", linewidth = 0.3),
    legend.title     = element_text(size = 7.5, face = "bold"),
    legend.text      = element_text(size = 7.0),
    legend.key.size  = unit(0.35, "cm"),
    legend.position  = "inside",
    legend.position.inside = c(0.98, 0.28),
    legend.justification = c(1, 0),
    legend.background = element_rect(fill = "white", color = "#DDDDDD",
                                     linewidth = 0.3),
    legend.margin    = margin(3, 5, 3, 5),
    plot.title       = element_text(size = 8.5, face = "bold", color = "#1A1A2E",
                                    margin = margin(b = 3, unit = "mm")),
    plot.margin      = margin(4, 5, 4, 2, "mm")
  ) +
  labs(title = "b  |DIFFUSE Δ| versus STRING interaction evidence")

# ═══════════════════════════════════════════════════════════════════════════════
# Combine panels using grid viewports
# ═══════════════════════════════════════════════════════════════════════════════

fig_w <- 7.87   # 200 mm
fig_h <- 4.72   # 120 mm

pdf_out <- file.path(OUT_DIR, "figA_pipeline.pdf")
png_out <- file.path(OUT_DIR, "figA_pipeline.png")

save_combined <- function(device_fn, ...) {
  device_fn(...)
  grid.newpage()
  # Viewports: centre-based (x, y = centre of viewport in npc)
  print(p_funnel,  vp = viewport(x = 0.16, y = 0.5, width = 0.32, height = 1.0))
  print(p_scatter, vp = viewport(x = 0.66, y = 0.5, width = 0.68, height = 1.0))
  dev.off()
}

tryCatch(
  save_combined(cairo_pdf, pdf_out, width = fig_w, height = fig_h),
  error = function(e) save_combined(pdf, pdf_out, width = fig_w, height = fig_h,
                                    useDingbats = FALSE)
)
cat(sprintf("Saved PDF: %s\n", pdf_out))

save_combined(png, png_out, width = round(fig_w * 300), height = round(fig_h * 300),
              res = 300, bg = "white")
cat(sprintf("Saved PNG: %s\n", png_out))
