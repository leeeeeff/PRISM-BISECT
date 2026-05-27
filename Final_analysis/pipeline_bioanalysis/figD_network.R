#!/usr/bin/env Rscript
# Fig D: 13 SUPPORTED isoform switches converge onto 4 biological pathways
# Left: manual network (4 pathway clusters, gene nodes, STRING edges)
# Right: SUPPORTED rate by cell type (all 26 Stage2 cases)
# Nature Methods double-column (180 mm × 170 mm)

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

# ─── Shared constants ─────────────────────────────────────────────────────────

CELL_COLS <- c(
  Excitatory      = "#C0392B",
  Inhibitory      = "#2475B0",
  OPC             = "#7D3C98",
  Astrocyte       = "#1E8449",
  Oligodendrocyte = "#CA6F1E",
  Microglia       = "#626567"
)
TIER_COLS <- c(A = "#C0392B", B = "#D35400")

CLUST_FILLS <- c(
  "1" = "#FFF3E0",   # WD40/motor — warm orange
  "2" = "#E3F2FD",   # Spectrin/DAPC — cool blue
  "3" = "#E8F5E9",   # Signaling/phosphatase — green
  "4" = "#F3E5F5"    # Organelle/specialized — purple
)
CLUST_BORDERS <- c("1"="#E67E22","2"="#2475B0","3"="#1E8449","4"="#7D3C98")
CLUST_TITLES <- c(
  "1" = "WD40 β-propeller\nredistribution",
  "2" = "Spectrin /\nDAPC complex",
  "3" = "Phosphatase &\nG-protein signaling",
  "4" = "Organelle &\nspecialised function"
)

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL a — Network
# ═══════════════════════════════════════════════════════════════════════════════

# ── Gene nodes (manually placed in cluster quadrants) ─────────────────────────
# Cluster bounding boxes (xmin, xmax, ymin, ymax):
#   1 = top-left  [0.3, 4.2] × [5.2, 9.5]
#   2 = top-right [4.8, 8.7] × [5.2, 9.5]
#   3 = bot-left  [0.3, 4.2] × [0.5, 4.8]
#   4 = bot-right [4.8, 8.7] × [0.5, 4.8]

nodes <- tribble(
  ~gene,    ~cl, ~x,   ~y,    ~delta,  ~cell_type,        ~tier,
  # Cluster 1: WD40/motor (Excitatory)
  "IFT122", "1", 1.8,  8.4,   0.954, "Excitatory",       "B",
  "KIF21B", "1", 3.3,  7.0,   0.855, "Excitatory",       "A",
  # Cluster 2: Spectrin/DAPC (Inhibitory)
  "DMD",    "2", 5.6,  8.3,   0.919, "Inhibitory",       "B",
  "SNTG1",  "2", 7.2,  8.8,   0.702, "Inhibitory",       "B",
  "SYNE1",  "2", 7.2,  7.0,   0.839, "Inhibitory",       "B",
  # Cluster 3: Signaling (Inhibitory + Astrocyte)
  "PTPRF",  "3", 1.2,  3.8,   0.729, "Inhibitory",       "B",
  "PTPRS",  "3", 3.2,  3.6,   0.789, "Astrocyte",        "B",
  "RGS3",   "3", 2.8,  2.0,   0.806, "Astrocyte",        "B",
  "ADGRB2", "3", 1.0,  1.4,   0.800, "Inhibitory",       "B",
  # Cluster 4: Organelle/specialised (mixed)
  "FANCA",  "4", 5.4,  3.8,   0.946, "Excitatory",       "B",
  "NDUFS4", "4", 7.0,  3.2,   0.563, "Excitatory",       "A",
  "BSG",    "4", 7.8,  4.5,   0.800, "Oligodendrocyte",  "B",
  "DLG1",   "4", 5.6,  1.8,   0.857, "OPC",              "A"
) %>%
  mutate(
    cell_hex = CELL_COLS[cell_type],
    tier_hex = TIER_COLS[tier],
    node_r   = scales::rescale(abs(delta), to = c(3.5, 8.0), from = c(0.5, 1.0))
  )

# ── Edges ─────────────────────────────────────────────────────────────────────
# (1) Within-cluster STRING-confirmed
edges_str <- tribble(
  ~from,  ~to,     ~score, ~type,
  "DMD",  "SNTG1", 992,    "STRING"
)

# (2) Gene → external STRING top partner (dashed, outside cluster)
ext_partners <- tribble(
  ~gene,    ~partner,  ~x_ext, ~y_ext, ~score,
  "IFT122", "IFT140",   2.5,   10.0,   999,
  "DMD",    "DAG1",     5.0,   10.0,   999,
  "FANCA",  "FANCF",    4.8,    0.0,   999,
  "BSG",    "SLC16A1",  8.8,    4.0,   999,
  "NDUFS4", "NDUFS6",   8.8,    2.4,   999,
  "PTPRF",  "PPFIA1",   0.0,    3.0,   997,
  "PTPRS",  "PPFIA1",   0.0,    3.0,   985
) %>%
  left_join(select(nodes, gene, x, y), by = "gene")

# ── Cluster bounding boxes ────────────────────────────────────────────────────
clust_boxes <- tribble(
  ~cl,  ~xmin, ~xmax, ~ymin, ~ymax,
  "1",  0.30,  4.20,  5.20,  9.50,
  "2",  4.80,  8.70,  5.20,  9.50,
  "3",  0.30,  4.20,  0.50,  4.80,
  "4",  4.80,  8.70,  0.50,  4.80
) %>%
  mutate(
    fill   = CLUST_FILLS[cl],
    border = CLUST_BORDERS[cl],
    title  = CLUST_TITLES[cl],
    title_x = (xmin + xmax) / 2,
    title_y = ymax - 0.32
  )

# ── Build edge data frames ────────────────────────────────────────────────────
# STRING edges
make_edge <- function(from_g, to_g) {
  f <- nodes %>% filter(gene == from_g)
  t <- nodes %>% filter(gene == to_g)
  data.frame(x=f$x, y=f$y, xend=t$x, yend=t$y)
}

str_edge_df <- edges_str %>%
  rowwise() %>%
  mutate(x=nodes$x[nodes$gene==from], y=nodes$y[nodes$gene==from],
         xend=nodes$x[nodes$gene==to], yend=nodes$y[nodes$gene==to]) %>%
  ungroup()

# PTPRF — PTPRS parallel pathway arrow (within cluster 3)
ptprf <- nodes %>% filter(gene=="PTPRF")
ptprs <- nodes %>% filter(gene=="PTPRS")

# ── Build plot ────────────────────────────────────────────────────────────────

# PPFIA1 external node is shared by PTPRF and PTPRS
ppfia1_xy <- c(x=0.0, y=3.0)

p_net <- ggplot() +

  # ── Cluster background boxes ──────────────────────────────────────────────
  geom_rect(
    data = clust_boxes,
    aes(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, fill=I(fill)),
    color = NA, alpha = 1
  ) +
  geom_rect(
    data = clust_boxes,
    aes(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax),
    fill  = NA, color = clust_boxes$border,
    linewidth = 0.7, linetype = "solid"
  ) +

  # ── Cluster titles ────────────────────────────────────────────────────────
  geom_label(
    data = clust_boxes,
    aes(x=title_x, y=title_y, label=title),
    fill  = clust_boxes$border, color = "white",
    size  = 2.8, fontface = "bold", lineheight = 0.88,
    label.padding = unit(0.15, "lines"),
    label.r = unit(0.08, "lines"), label.size = 0
  ) +

  # ── External partner edges (dashed, gray) ────────────────────────────────
  geom_segment(
    data = ext_partners %>% filter(partner != "PPFIA1"),
    aes(x=x, y=y, xend=x_ext, yend=y_ext),
    linetype="dashed", color="#AAAAAA", linewidth=0.45,
    arrow=arrow(length=unit(0.07,"inches"), type="open")
  ) +
  # PPFIA1 (shared by PTPRF and PTPRS)
  geom_segment(
    data = ext_partners %>% filter(partner == "PPFIA1"),
    aes(x=x, y=y, xend=x_ext, yend=y_ext),
    linetype="dashed", color="#1E8449", linewidth=0.45,
    arrow=arrow(length=unit(0.07,"inches"), type="open")
  ) +

  # ── External partner node labels ─────────────────────────────────────────
  geom_label(
    data = ext_partners %>% distinct(partner, x_ext, y_ext) %>%
           filter(partner != "PPFIA1"),
    aes(x=x_ext, y=y_ext, label=partner),
    size=2.2, color="#666666", fill="white",
    label.padding=unit(0.1,"lines"), label.r=unit(0.1,"lines"),
    label.size=0.3
  ) +
  # PPFIA1 shared label
  annotate("label", x=ppfia1_xy["x"], y=ppfia1_xy["y"],
           label="PPFIA1\n(Liprin-α)", size=2.2, color="#1E8449",
           fill="white", label.padding=unit(0.1,"lines"),
           label.r=unit(0.1,"lines"), label.size=0.4) +

  # ── STRING-confirmed within-cluster edge ─────────────────────────────────
  geom_segment(
    data = str_edge_df,
    aes(x=x, y=y, xend=xend, yend=yend),
    color="#2475B0", linewidth=1.2, alpha=0.65
  ) +
  annotate("text",
           x = (str_edge_df$x + str_edge_df$xend)/2 + 0.3,
           y = (str_edge_df$y + str_edge_df$yend)/2 + 0.15,
           label="STRING\n992", size=1.9, color="#2475B0",
           fontface="italic", lineheight=0.85) +

  # ── PTPRF — PTPRS parallel arrow (within cluster 3) ──────────────────────
  annotate("segment",
           x=ptprf$x+0.25, y=ptprf$y, xend=ptprs$x-0.25, yend=ptprs$y,
           linetype="dotted", color="#1E8449", linewidth=0.7,
           arrow=arrow(length=unit(0.07,"inches"), ends="both", type="open")) +
  annotate("text",
           x=(ptprf$x+ptprs$x)/2, y=(ptprf$y+ptprs$y)/2+0.25,
           label="parallel", size=1.9, color="#1E8449", fontface="italic") +

  # ── Gene nodes ────────────────────────────────────────────────────────────
  geom_point(
    data = nodes,
    aes(x=x, y=y, size=I(node_r), fill=I(cell_hex)),
    shape=21, color="white", stroke=0.8
  ) +

  # ── Tier ring (Tier A only) ───────────────────────────────────────────────
  geom_point(
    data = nodes %>% filter(tier == "A"),
    aes(x=x, y=y, size=I(node_r + 2.5)),
    shape=21, fill=NA, color=TIER_COLS["A"], stroke=1.0
  ) +

  # ── Gene labels ───────────────────────────────────────────────────────────
  geom_text(
    data = nodes,
    aes(x=x, y=y-node_r*0.028-0.35, label=gene, color=I(cell_hex)),
    size=2.6, fontface="bold"
  ) +

  # ── DIFFUSE Δ sub-labels ──────────────────────────────────────────────────
  geom_text(
    data = nodes,
    aes(x=x, y=y-node_r*0.028-0.60,
        label=sprintf("Δ=%.2f", delta)),
    size=1.9, color="#777777"
  ) +

  # ── Special markers ───────────────────────────────────────────────────────
  # FANCA: accelerated evolution
  annotate("text",
           x = nodes$x[nodes$gene=="FANCA"] + 0.45,
           y = nodes$y[nodes$gene=="FANCA"] + 0.5,
           label = "phyloP\n−0.49*",
           size=2.0, color="#C0392B", fontface="bold.italic", lineheight=0.85) +
  # BSG: accelerated evolution
  annotate("text",
           x = nodes$x[nodes$gene=="BSG"] + 0.52,
           y = nodes$y[nodes$gene=="BSG"] + 0.5,
           label = "phyloP\n−0.47†",
           size=2.0, color="#C0392B", fontface="bold.italic", lineheight=0.85) +
  # KIF21B: Tier A label
  annotate("text",
           x = nodes$x[nodes$gene=="KIF21B"] - 0.5,
           y = nodes$y[nodes$gene=="KIF21B"] + 0.55,
           label = "Tier A", size=2.0, color=TIER_COLS["A"], fontface="bold") +
  # NDUFS4: Tier A
  annotate("text",
           x = nodes$x[nodes$gene=="NDUFS4"] - 0.5,
           y = nodes$y[nodes$gene=="NDUFS4"] + 0.55,
           label = "Tier A", size=2.0, color=TIER_COLS["A"], fontface="bold") +
  # DLG1: Tier A
  annotate("text",
           x = nodes$x[nodes$gene=="DLG1"] - 0.5,
           y = nodes$y[nodes$gene=="DLG1"] + 0.55,
           label = "Tier A", size=2.0, color=TIER_COLS["A"], fontface="bold") +

  # ── Cell-type legend (bottom) ─────────────────────────────────────────────
  {
    ct_shown <- c("Excitatory","Inhibitory","Astrocyte","OPC","Oligodendrocyte")
    ct_x <- seq(0.5, 7.5, length.out=length(ct_shown))
    ct_y <- rep(-0.55, length(ct_shown))
    list(
      annotate("point", x=ct_x, y=ct_y,
               size=3.5, color=CELL_COLS[ct_shown], shape=16),
      annotate("text",  x=ct_x+0.2, y=ct_y,
               label=ct_shown, hjust=0, size=2.1, color="#444444")
    )
  } +

  # ── Axes & theme ─────────────────────────────────────────────────────────
  scale_x_continuous(limits=c(-0.8, 9.5), expand=c(0,0)) +
  scale_y_continuous(limits=c(-0.9, 10.4), expand=c(0,0)) +
  coord_cartesian(clip="off") +
  labs(title="a  13 M11-SUPPORTED switches converge onto 4 pathway themes") +
  theme_void(base_size=9) +
  theme(
    plot.background = element_rect(fill="white", color=NA),
    plot.title      = element_text(size=8.5, face="bold", color="#1A1A2E",
                                   margin=margin(b=3, unit="mm")),
    plot.margin     = margin(4, 2, 4, 4, "mm")
  )

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL b — SUPPORTED rate by cell type (all 26 Stage2 cases)
# ═══════════════════════════════════════════════════════════════════════════════

ct_data <- tribble(
  ~cell_type,         ~n_total, ~n_supp,
  "Excitatory",       7,        4,
  "Inhibitory",       7,        5,
  "Oligodendrocyte",  6,        1,
  "Astrocyte",        3,        2,
  "OPC",              2,        1,
  "Microglia",        1,        0
) %>%
  mutate(
    n_unsupp   = n_total - n_supp,
    pct_supp   = n_supp / n_total,
    cell_type  = factor(cell_type,
                        levels=c("Excitatory","Inhibitory","Astrocyte",
                                 "OPC","Oligodendrocyte","Microglia")),
    cell_hex   = CELL_COLS[as.character(cell_type)]
  )

ct_long <- bind_rows(
  ct_data %>% mutate(verdict="SUPPORTED",   n=n_supp),
  ct_data %>% mutate(verdict="UNSUPPORTED", n=n_unsupp)
) %>%
  mutate(verdict=factor(verdict, levels=c("UNSUPPORTED","SUPPORTED")))

p_ct <- ggplot(ct_long, aes(x=cell_type, y=n)) +

  # ── Bars ────────────────────────────────────────────────────────────────
  geom_col(
    data = ct_long %>% filter(verdict=="UNSUPPORTED"),
    aes(fill=I(cell_hex)), alpha=0.22, width=0.68, color="white"
  ) +
  geom_col(
    data = ct_long %>% filter(verdict=="SUPPORTED"),
    aes(fill=I(cell_hex)), alpha=0.92, width=0.68, color="white"
  ) +

  # ── Count label inside bar ───────────────────────────────────────────────
  geom_text(
    data = ct_long %>% filter(verdict=="SUPPORTED" & n>0),
    aes(y=n-0.08, label=n, color=I("white")),
    vjust=1, size=3.5, fontface="bold"
  ) +

  # ── Combined n/total (% rate) label outside bar ───────────────────────
  geom_text(
    data = ct_data,
    aes(y = n_total + 0.20,
        label = sprintf("%d/%d\n(%d%%)", n_supp, n_total, round(pct_supp*100)),
        color = I(cell_hex)),
    hjust = 0, vjust = 0.5, size = 2.6, fontface = "bold", lineheight = 0.85
  ) +

  # ── Legend ──────────────────────────────────────────────────────────────
  annotate("rect", xmin=0.5, xmax=0.85, ymin=8.4, ymax=8.7,
           fill="#555555", alpha=0.90) +
  annotate("text", x=0.90, y=8.55, label="SUPPORTED",
           hjust=0, size=2.3, color="#555555", fontface="bold") +
  annotate("rect", xmin=0.5, xmax=0.85, ymin=7.9, ymax=8.2,
           fill="#CCCCCC", alpha=0.40) +
  annotate("text", x=0.90, y=8.05, label="UNSUPPORTED",
           hjust=0, size=2.3, color="#AAAAAA") +

  scale_y_continuous(limits=c(0, 9.5), expand=c(0,0),
                     breaks=0:7, name="Stage 2 PASS cases (n)") +
  scale_x_discrete(name=NULL) +
  coord_flip(clip="off") +
  labs(title="b  M11 SUPPORTED rate\nby cell type") +
  theme_classic(base_size=9) +
  theme(
    plot.background  = element_rect(fill="white", color=NA),
    panel.background = element_rect(fill="white"),
    panel.grid.major.x = element_line(color="#F0F0F0", linewidth=0.3),
    axis.text.y = element_text(size=8, color="#444444", face="bold"),
    axis.text.x = element_text(size=7.5, color="#555555"),
    axis.title.x= element_text(size=8, color="#444444"),
    axis.line   = element_line(color="#888888", linewidth=0.4),
    axis.ticks  = element_line(color="#888888", linewidth=0.3),
    plot.title  = element_text(size=8.5, face="bold", color="#1A1A2E",
                               margin=margin(b=3, unit="mm")),
    plot.margin = margin(4, 5, 4, 2, "mm")
  )

# ═══════════════════════════════════════════════════════════════════════════════
# Combine and save
# ═══════════════════════════════════════════════════════════════════════════════

fig_w <- 7.09   # 180 mm
fig_h <- 6.69   # 170 mm

pdf_out <- file.path(OUT_DIR, "figD_network.pdf")
png_out <- file.path(OUT_DIR, "figD_network.png")

save_fig <- function(device_fn, ...) {
  device_fn(...)
  grid.newpage()
  print(p_net, vp=viewport(x=0.00, y=0, width=0.60, height=1.0,
                            just=c("left","bottom")))
  print(p_ct,  vp=viewport(x=0.60, y=0, width=0.40, height=1.0,
                            just=c("left","bottom")))
  dev.off()
}

tryCatch(
  save_fig(cairo_pdf, pdf_out, width=fig_w, height=fig_h),
  error=function(e) save_fig(pdf, pdf_out, width=fig_w, height=fig_h,
                              useDingbats=FALSE)
)
cat(sprintf("Saved PDF: %s\n", pdf_out))

save_fig(png, png_out,
         width=round(fig_w*300), height=round(fig_h*300), res=300, bg="white")
cat(sprintf("Saved PNG: %s\n", png_out))
