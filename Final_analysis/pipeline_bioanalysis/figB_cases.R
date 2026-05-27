#!/usr/bin/env Rscript
# Fig B: Tier A case deep-dive — KIF21B (domain architecture + pLDDT) and NDUFS4
# Nature Methods double-column (180 mm × 200 mm)
# Packages: ggplot2, dplyr, tibble, grid

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

# ─── Shared colour constants ──────────────────────────────────────────────────

COL_CT  <- "#C0392B"   # CT isoform red
COL_AD  <- "#2475B0"   # AD isoform blue
COL_NA  <- "#BDC3C7"   # neutral gray

# ─── Helper: draw a single protein diagram ───────────────────────────────────
# domains: data.frame(start, end, name, color, label_y_offset)
# aa_total: total length in aa
# y_center: y position of the backbone

make_protein_panel <- function(domains, aa_total, y_center = 0,
                                label_size = 2.5, backbone_col = "#7F8C8D") {
  # Backbone
  bb <- annotate("segment",
                 x = 0, xend = aa_total,
                 y = y_center, yend = y_center,
                 color = backbone_col, linewidth = 1.0)
  # End caps
  ec_lo <- annotate("segment", x = 0, xend = 0,
                    y = y_center - 0.06, yend = y_center + 0.06,
                    color = backbone_col, linewidth = 1.0)
  ec_hi <- annotate("segment", x = aa_total, xend = aa_total,
                    y = y_center - 0.06, yend = y_center + 0.06,
                    color = backbone_col, linewidth = 1.0)
  list(backbone = bb, ec_lo = ec_lo, ec_hi = ec_hi)
}

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL a — KIF21B domain architecture
# ═══════════════════════════════════════════════════════════════════════════════
# y=2: CT isoform (418 aa, kinesin motor)
# y=1: AD isoform (710 aa, WD40 scaffold)

# CT domains (approximate positions from UniProt KIF21B / NIC alignment)
kif_ct_domains <- tribble(
  ~start, ~end,  ~name,             ~fill,    ~label,
  1,      335,   "Kinesin motor",   "#E74C3C","Kinesin\nmotor\n(P-loop, Sw-I, Sw-II)",
  1,       30,   "P-loop",          "#C0392B","P-loop",
  335,    418,   "DUF5082",         "#E67E22","DUF5082"
)

# AD domains: 14 WD40-type repeats spread across 710 aa
# Group into 8 visual blocks (ANAPC4_WD40, NBCH_WD40, Nup160)
wd40_types <- c("ANAPC4\nWD40","NBCH\nWD40","Nup160","WD40",
                "ANAPC4\nWD40","WD40","Nup160","WD40")
wd40_fills <- c("#1F618D","#2E86C1","#2471A3","#85C1E9",
                "#1F618D","#85C1E9","#2471A3","#5DADE2")
wd40_width <- 710 / 8   # ≈89 aa per visual block

kif_ad_domains <- tibble(
  idx   = seq_along(wd40_types),
  start = (idx - 1) * wd40_width,
  end   = idx * wd40_width,
  name  = wd40_types,
  fill  = wd40_fills
)

# y positions
Y_KIF_CT <- 1.60
Y_KIF_AD <- 0.75
DOM_H    <- 0.32

p_kif_arch <- ggplot() +

  # ── CT backbone ─────────────────────────────────────────────────────────────
  annotate("segment", x=0, xend=418, y=Y_KIF_CT, yend=Y_KIF_CT,
           color="#7F8C8D", linewidth=0.8) +
  annotate("segment", x=c(0,418), xend=c(0,418),
           y=Y_KIF_CT-0.04, yend=Y_KIF_CT+0.04,
           color="#7F8C8D", linewidth=0.8) +

  # ── CT kinesin motor (large box) ────────────────────────────────────────────
  geom_rect(data=tibble(x=1,xe=335,y=Y_KIF_CT-DOM_H/2,ye=Y_KIF_CT+DOM_H/2),
            aes(xmin=x,xmax=xe,ymin=y,ymax=ye), fill="#E74C3C", color="white",
            linewidth=0.3, alpha=0.92) +
  annotate("text", x=168, y=Y_KIF_CT, label="Kinesin motor domain",
           size=3.0, color="white", fontface="bold") +

  # ── CT sub-motif annotations ─────────────────────────────────────────────────
  annotate("text", x=25,  y=Y_KIF_CT+DOM_H/2+0.09, label="P-loop",
           size=2.0, color=COL_CT, fontface="italic") +
  annotate("segment", x=25, xend=25, y=Y_KIF_CT+DOM_H/2, yend=Y_KIF_CT+DOM_H/2+0.06,
           color=COL_CT, linewidth=0.5) +
  annotate("text", x=175, y=Y_KIF_CT+DOM_H/2+0.09, label="Switch-I",
           size=2.0, color=COL_CT, fontface="italic") +
  annotate("segment", x=175, xend=175, y=Y_KIF_CT+DOM_H/2, yend=Y_KIF_CT+DOM_H/2+0.06,
           color=COL_CT, linewidth=0.5) +
  annotate("text", x=255, y=Y_KIF_CT+DOM_H/2+0.09, label="Switch-II",
           size=2.0, color=COL_CT, fontface="italic") +
  annotate("segment", x=255, xend=255, y=Y_KIF_CT+DOM_H/2, yend=Y_KIF_CT+DOM_H/2+0.06,
           color=COL_CT, linewidth=0.5) +

  # ── CT DUF5082 ──────────────────────────────────────────────────────────────
  geom_rect(data=tibble(x=340,xe=418,y=Y_KIF_CT-DOM_H/2,ye=Y_KIF_CT+DOM_H/2),
            aes(xmin=x,xmax=xe,ymin=y,ymax=ye), fill="#E67E22", color="white",
            linewidth=0.3, alpha=0.92) +
  annotate("text", x=379, y=Y_KIF_CT, label="DUF\n5082",
           size=2.2, color="white", fontface="bold", lineheight=0.85) +

  # ── CT length label ──────────────────────────────────────────────────────────
  annotate("text", x=430, y=Y_KIF_CT, label="418 aa",
           size=2.5, color="#555555", hjust=0) +

  # ── CT isoform label ─────────────────────────────────────────────────────────
  annotate("text", x=-10, y=Y_KIF_CT, label="CT\n(NIC)", hjust=1,
           size=2.8, color=COL_CT, fontface="bold", lineheight=0.85) +
  annotate("text", x=-10, y=Y_KIF_CT-0.26, label="tr293004",
           hjust=1, size=2.0, color="#888888") +

  # ── AD backbone ─────────────────────────────────────────────────────────────
  annotate("segment", x=0, xend=710, y=Y_KIF_AD, yend=Y_KIF_AD,
           color="#7F8C8D", linewidth=0.8) +
  annotate("segment", x=c(0,710), xend=c(0,710),
           y=Y_KIF_AD-0.04, yend=Y_KIF_AD+0.04,
           color="#7F8C8D", linewidth=0.8) +

  # ── AD WD40 blocks ──────────────────────────────────────────────────────────
  geom_rect(data = kif_ad_domains,
            aes(xmin=start, xmax=end,
                ymin=Y_KIF_AD-DOM_H/2, ymax=Y_KIF_AD+DOM_H/2,
                fill=I(fill)),
            color="white", linewidth=0.3, alpha=0.92) +
  geom_text(data = kif_ad_domains,
            aes(x=(start+end)/2, y=Y_KIF_AD, label=name),
            size=1.9, color="white", fontface="bold", lineheight=0.80) +

  # ── AD β-propeller label ─────────────────────────────────────────────────────
  annotate("text", x=355, y=Y_KIF_AD-DOM_H/2-0.12,
           label="WD40 β-propeller scaffold (dominant-negative coiled-coil via KIF21B-201)",
           size=2.2, color="#1F618D", fontface="italic", hjust=0.5) +

  # ── AD length label ──────────────────────────────────────────────────────────
  annotate("text", x=720, y=Y_KIF_AD, label="710 aa",
           size=2.5, color="#555555", hjust=0) +

  # ── AD isoform label ─────────────────────────────────────────────────────────
  annotate("text", x=-10, y=Y_KIF_AD, label="AD\n(NNIC)", hjust=1,
           size=2.8, color=COL_AD, fontface="bold", lineheight=0.85) +
  annotate("text", x=-10, y=Y_KIF_AD-0.26, label="tr292978",
           hjust=1, size=2.0, color="#888888") +

  # ── DIFFUSE Δ annotation ──────────────────────────────────────────────────────
  annotate("text", x=355, y=0.28,
           label="DIFFUSE Δ = −0.855  (CT function high → AD WD40 scaffold: motor loss)",
           size=2.5, color="#6C3483", fontface="bold.italic", hjust=0.5) +

  # ── Axis + theme ─────────────────────────────────────────────────────────────
  scale_x_continuous(limits=c(-80, 790), expand=c(0,0),
                     breaks=seq(0,700,100), labels=seq(0,700,100)) +
  scale_y_continuous(limits=c(0.1, 2.15), expand=c(0,0)) +
  coord_cartesian(clip="off") +
  labs(x="Residue (aa)", y=NULL,
       title="a  KIF21B: kinesin motor → WD40 scaffold switch (Excitatory, AD)") +
  theme_classic(base_size=9) +
  theme(
    plot.background  = element_rect(fill="white", color=NA),
    panel.background = element_rect(fill="white"),
    panel.grid       = element_blank(),
    axis.line.y      = element_blank(),
    axis.ticks.y     = element_blank(),
    axis.text.y      = element_blank(),
    axis.title.x     = element_text(size=8, color="#444444"),
    axis.text.x      = element_text(size=7.5, color="#555555"),
    axis.line.x      = element_line(color="#888888", linewidth=0.4),
    axis.ticks.x     = element_line(color="#888888", linewidth=0.3),
    plot.title       = element_text(size=8.5, face="bold", color="#1A1A2E",
                                    margin=margin(b=3, unit="mm")),
    plot.margin      = margin(4, 5, 2, 8, "mm")
  )

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL b — KIF21B pLDDT trace
# ═══════════════════════════════════════════════════════════════════════════════

plddt <- read.csv("/tmp/kif21b_plddt.csv", stringsAsFactors=FALSE) %>%
  mutate(across(everything(), function(x) suppressWarnings(as.numeric(x))))

ct_trace <- plddt %>% filter(!is.na(residue_ct) & !is.na(plddt_ct)) %>%
  select(x = residue_ct, y = plddt_ct)
ad_trace <- plddt %>% filter(!is.na(residue_ad) & !is.na(plddt_ad)) %>%
  select(x = residue_ad, y = plddt_ad)

mean_ct <- mean(ct_trace$y, na.rm=TRUE)
mean_ad <- mean(ad_trace$y, na.rm=TRUE)

p_kif_plddt <- ggplot() +

  # ── pLDDT quality bands ──────────────────────────────────────────────────────
  annotate("rect", xmin=0, xmax=760, ymin=90, ymax=101,
           fill="#EAF2FF", alpha=0.5) +
  annotate("rect", xmin=0, xmax=760, ymin=70, ymax=90,
           fill="#FFF9E6", alpha=0.4) +
  annotate("text", x=755, y=95.5, label="Very high (>90)", hjust=1,
           size=2.0, color="#1F618D", fontface="italic") +
  annotate("text", x=755, y=79.5, label="Confident (70–90)", hjust=1,
           size=2.0, color="#B7950B", fontface="italic") +

  # ── Domain boundary lines ────────────────────────────────────────────────────
  annotate("segment", x=335, xend=335, y=60, yend=101,
           linetype="dashed", color="#E74C3C", linewidth=0.5, alpha=0.7) +
  annotate("text", x=335, y=62, label="Motor\nboundary",
           size=1.9, color="#E74C3C", hjust=0.5, lineheight=0.8) +
  annotate("segment", x=370, xend=620, y=62.5, yend=62.5,
           color="#2475B0", linewidth=0.5, alpha=0.7) +
  annotate("text", x=495, y=61, label="WD40 core (ESMAtlas)",
           size=1.9, color="#2475B0", hjust=0.5) +

  # ── CT trace (raw + smoothed) ────────────────────────────────────────────────
  geom_ribbon(data=ct_trace, aes(x=x, ymin=70, ymax=y),
              fill=COL_CT, alpha=0.12) +
  geom_line(data=ct_trace,  aes(x=x, y=y),
            color=COL_CT, linewidth=0.35, alpha=0.4) +
  geom_smooth(data=ct_trace, aes(x=x, y=y), method="loess", span=0.12,
              se=FALSE, color=COL_CT, linewidth=1.2) +

  # ── AD trace (raw + smoothed) ────────────────────────────────────────────────
  geom_ribbon(data=ad_trace, aes(x=x, ymin=70, ymax=y),
              fill=COL_AD, alpha=0.12) +
  geom_line(data=ad_trace,  aes(x=x, y=y),
            color=COL_AD, linewidth=0.35, alpha=0.4) +
  geom_smooth(data=ad_trace, aes(x=x, y=y), method="loess", span=0.20,
              se=FALSE, color=COL_AD, linewidth=1.2) +

  # ── Mean reference lines ─────────────────────────────────────────────────────
  annotate("segment", x=1, xend=380, y=mean_ct, yend=mean_ct,
           linetype="dashed", color=COL_CT, linewidth=0.65, alpha=0.8) +
  annotate("segment", x=370, xend=620, y=mean_ad, yend=mean_ad,
           linetype="dashed", color=COL_AD, linewidth=0.65, alpha=0.8) +
  annotate("text", x=190,  y=mean_ct+1.5,
           label=sprintf("μ=%.1f", mean_ct),
           size=2.6, color=COL_CT, fontface="bold") +
  annotate("text", x=495, y=mean_ad+1.5,
           label=sprintf("μ=%.1f", mean_ad),
           size=2.6, color=COL_AD, fontface="bold") +

  # ── Legend ───────────────────────────────────────────────────────────────────
  annotate("segment", x=430, xend=455, y=71.5, yend=71.5,
           color=COL_CT, linewidth=1.2) +
  annotate("text", x=460, y=71.5, label="CT kinesin (aa 1–380)",
           hjust=0, size=2.4, color=COL_CT, fontface="bold") +
  annotate("segment", x=430, xend=455, y=68.5, yend=68.5,
           color=COL_AD, linewidth=1.2) +
  annotate("text", x=460, y=68.5, label="AD WD40 core (aa 370–620)",
           hjust=0, size=2.4, color=COL_AD, fontface="bold") +

  scale_x_continuous(limits=c(0, 760), breaks=seq(0,700,100), expand=c(0,0)) +
  scale_y_continuous(limits=c(60, 101), breaks=c(70,80,90,100), expand=c(0,0)) +
  labs(x="Residue (aa)", y="ESMAtlas pLDDT",
       title="b  Structural confidence: both folds are high-quality (μ>93)") +
  theme_classic(base_size=9) +
  theme(
    plot.background  = element_rect(fill="white", color=NA),
    panel.background = element_rect(fill="white"),
    panel.grid.major = element_line(color="#F5F5F5", linewidth=0.3),
    axis.title       = element_text(size=8, color="#444444"),
    axis.text        = element_text(size=7.5, color="#555555"),
    axis.line        = element_line(color="#888888", linewidth=0.4),
    axis.ticks       = element_line(color="#888888", linewidth=0.3),
    plot.title       = element_text(size=8.5, face="bold", color="#1A1A2E",
                                    margin=margin(b=3, unit="mm")),
    plot.margin      = margin(2, 5, 4, 8, "mm")
  )

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL c — NDUFS4 domain architecture
# ═══════════════════════════════════════════════════════════════════════════════

Y_ND_CT <- 1.60
Y_ND_AD <- 0.75

p_ndufs4_arch <- ggplot() +

  # ── CT backbone ─────────────────────────────────────────────────────────────
  annotate("segment", x=0, xend=175, y=Y_ND_CT, yend=Y_ND_CT,
           color="#7F8C8D", linewidth=0.8) +
  annotate("segment", x=c(0,175), xend=c(0,175),
           y=Y_ND_CT-0.04, yend=Y_ND_CT+0.04,
           color="#7F8C8D", linewidth=0.8) +

  # ── CT MTS ───────────────────────────────────────────────────────────────────
  geom_rect(data=tibble(x=1,xe=39,y=Y_ND_CT-DOM_H/2,ye=Y_ND_CT+DOM_H/2),
            aes(xmin=x,xmax=xe,ymin=y,ymax=ye), fill="#E74C3C", color="white",
            linewidth=0.3, alpha=0.92) +
  annotate("text", x=20, y=Y_ND_CT, label="MTS", size=2.5,
           color="white", fontface="bold") +

  # ── CT LYR motif (small tick) ────────────────────────────────────────────────
  geom_rect(data=tibble(x=40,xe=56,y=Y_ND_CT-DOM_H/2,ye=Y_ND_CT+DOM_H/2),
            aes(xmin=x,xmax=xe,ymin=y,ymax=ye), fill="#F39C12", color="white",
            linewidth=0.3, alpha=0.92) +
  annotate("text", x=48, y=Y_ND_CT, label="LYR", size=2.0,
           color="white", fontface="bold") +

  # ── CT Complex-I subunit ─────────────────────────────────────────────────────
  geom_rect(data=tibble(x=57,xe=175,y=Y_ND_CT-DOM_H/2,ye=Y_ND_CT+DOM_H/2),
            aes(xmin=x,xmax=xe,ymin=y,ymax=ye), fill="#1A9652", color="white",
            linewidth=0.3, alpha=0.92) +
  annotate("text", x=116, y=Y_ND_CT, label="Complex I\nNDUFS4 subunit",
           size=2.5, color="white", fontface="bold", lineheight=0.85) +

  # ── CT MTS import arrow ───────────────────────────────────────────────────────
  annotate("text", x=20, y=Y_ND_CT+DOM_H/2+0.12,
           label="→ mitochondria", size=2.0, color="#E74C3C", fontface="italic") +

  # ── CT length + label ────────────────────────────────────────────────────────
  annotate("text", x=185, y=Y_ND_CT, label="175 aa", size=2.5, color="#555555", hjust=0) +
  annotate("text", x=-8,  y=Y_ND_CT, label="CT\n(canonical)", hjust=1,
           size=2.8, color=COL_CT, fontface="bold", lineheight=0.85) +
  annotate("text", x=-8,  y=Y_ND_CT-0.26, label="NDUFS4-201",
           hjust=1, size=2.0, color="#888888") +

  # ── AD backbone ─────────────────────────────────────────────────────────────
  annotate("segment", x=0, xend=379, y=Y_ND_AD, yend=Y_ND_AD,
           color="#7F8C8D", linewidth=0.8) +
  annotate("segment", x=c(0,379), xend=c(0,379),
           y=Y_ND_AD-0.04, yend=Y_ND_AD+0.04,
           color="#7F8C8D", linewidth=0.8) +

  # ── AD novel N-terminal (no MTS) ────────────────────────────────────────────
  geom_rect(data=tibble(x=1,xe=49,y=Y_ND_AD-DOM_H/2,ye=Y_ND_AD+DOM_H/2),
            aes(xmin=x,xmax=xe,ymin=y,ymax=ye), fill="#95A5A6", color="white",
            linewidth=0.3, alpha=0.92) +
  annotate("text", x=25, y=Y_ND_AD, label="Novel\nexon", size=2.0,
           color="white", fontface="bold", lineheight=0.85) +
  annotate("text", x=25, y=Y_ND_AD+DOM_H/2+0.12,
           label="MTS absent\n(D+E=4)", size=1.9, color="#E74C3C",
           fontface="italic", lineheight=0.8) +

  # ── AD RVT_1 (LINE-1 RT) ─────────────────────────────────────────────────────
  geom_rect(data=tibble(x=50,xe=350,y=Y_ND_AD-DOM_H/2,ye=Y_ND_AD+DOM_H/2),
            aes(xmin=x,xmax=xe,ymin=y,ymax=ye), fill="#7D3C98", color="white",
            linewidth=0.3, alpha=0.92) +
  annotate("text", x=200, y=Y_ND_AD,
           label="RVT_1  (LINE-1 ORF2p reverse transcriptase, E=4.6×10⁻⁴⁸)",
           size=2.5, color="white", fontface="bold") +

  # ── AD C-terminal ────────────────────────────────────────────────────────────
  geom_rect(data=tibble(x=351,xe=379,y=Y_ND_AD-DOM_H/2,ye=Y_ND_AD+DOM_H/2),
            aes(xmin=x,xmax=xe,ymin=y,ymax=ye), fill="#95A5A6", color="white",
            linewidth=0.3, alpha=0.92) +

  # ── TSS offset annotation ────────────────────────────────────────────────────
  annotate("segment", x=0, xend=0, y=Y_ND_AD-DOM_H/2, yend=Y_ND_AD-DOM_H/2-0.14,
           color="#7D3C98", linewidth=0.7) +
  annotate("text", x=0, y=Y_ND_AD-DOM_H/2-0.22,
           label="TSS +7 bp\nchr5:53,686,672", size=1.9, color="#7D3C98",
           hjust=0.5, lineheight=0.85) +

  # ── AD length + label ────────────────────────────────────────────────────────
  annotate("text", x=389, y=Y_ND_AD, label="379 aa", size=2.5, color="#555555", hjust=0) +
  annotate("text", x=-8,  y=Y_ND_AD, label="AD\n(NNIC)", hjust=1,
           size=2.8, color=COL_AD, fontface="bold", lineheight=0.85) +
  annotate("text", x=-8,  y=Y_ND_AD-0.26, label="tr73243",
           hjust=1, size=2.0, color="#888888") +

  # ── DIFFUSE Δ annotation ──────────────────────────────────────────────────────
  annotate("text", x=190, y=0.28,
           label="DIFFUSE Δ = −0.563  (CT mitochondrial function high → AD locus hijacking)",
           size=2.5, color="#6C3483", fontface="bold.italic", hjust=0.5) +

  scale_x_continuous(limits=c(-70, 460), expand=c(0,0),
                     breaks=seq(0,400,100), labels=seq(0,400,100)) +
  scale_y_continuous(limits=c(0.1, 2.15), expand=c(0,0)) +
  coord_cartesian(clip="off") +
  labs(x="Residue (aa)", y=NULL,
       title="c  NDUFS4: mitochondrial subunit → LINE-1 locus hijacking (Excitatory, AD)") +
  theme_classic(base_size=9) +
  theme(
    plot.background  = element_rect(fill="white", color=NA),
    panel.background = element_rect(fill="white"),
    panel.grid       = element_blank(),
    axis.line.y      = element_blank(),
    axis.ticks.y     = element_blank(),
    axis.text.y      = element_blank(),
    axis.title.x     = element_text(size=8, color="#444444"),
    axis.text.x      = element_text(size=7.5, color="#555555"),
    axis.line.x      = element_line(color="#888888", linewidth=0.4),
    axis.ticks.x     = element_line(color="#888888", linewidth=0.3),
    plot.title       = element_text(size=8.5, face="bold", color="#1A1A2E",
                                    margin=margin(b=3, unit="mm")),
    plot.margin      = margin(4, 5, 2, 8, "mm")
  )

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL d — NDUFS4 multi-evidence bar chart
# ═══════════════════════════════════════════════════════════════════════════════

evidence_df <- tribble(
  ~metric,                    ~isoform, ~value,  ~fill,
  "DIFFUSE\nscore",           "CT",      0.587,  COL_CT,
  "DIFFUSE\nscore",           "AD",      0.024,  COL_AD,
  "STRING score\n(÷10³)", "CT",  0.999, "#27AE60",
  "STRING score\n(÷10³)", "AD",  0.999, "#27AE60",
  "phyloP\n(AD exon)",        "AD",      2.263,  "#1F618D",
  "MitoFates\nscore",         "CT",      0.92,   "#E74C3C",
  "MitoFates\nscore",         "AD",      0.03,   "#BDC3C7"
) %>%
  mutate(metric = factor(metric, levels=unique(metric)),
         isoform= factor(isoform, levels=c("CT","AD")))

p_ndufs4_ev <- ggplot(evidence_df, aes(x=isoform, y=value, fill=I(fill))) +
  geom_col(width=0.62, color="white", linewidth=0.3) +
  geom_text(aes(label=sprintf("%.2f", value), color=I(ifelse(value>0.5,"white","#333333"))),
            position=position_stack(vjust=0.5), size=2.6, fontface="bold") +
  facet_wrap(~metric, nrow=1, scales="free_y") +
  scale_y_continuous(expand=c(0.02, 0)) +
  labs(x=NULL, y="Score", title="d  NDUFS4 multi-evidence: CT functional, AD locus-hijacked") +
  theme_classic(base_size=9) +
  theme(
    plot.background   = element_rect(fill="white", color=NA),
    panel.background  = element_rect(fill="white"),
    panel.grid.major.y= element_line(color="#F0F0F0", linewidth=0.3),
    strip.background  = element_rect(fill="#F5F5F5", color=NA),
    strip.text        = element_text(size=7.0, face="bold", color="#333333",
                                     lineheight=0.88),
    axis.title.y      = element_text(size=8, color="#444444"),
    axis.text         = element_text(size=7.5, color="#555555"),
    axis.line.x       = element_line(color="#888888", linewidth=0.4),
    axis.line.y       = element_line(color="#888888", linewidth=0.4),
    axis.ticks        = element_line(color="#888888", linewidth=0.3),
    panel.spacing     = unit(0.3, "lines"),
    plot.title        = element_text(size=8.5, face="bold", color="#1A1A2E",
                                     margin=margin(b=3, unit="mm")),
    plot.margin       = margin(2, 5, 4, 8, "mm")
  )

# ═══════════════════════════════════════════════════════════════════════════════
# Combine 4 panels: 2 rows × 2 columns
# ═══════════════════════════════════════════════════════════════════════════════

fig_w <- 7.09   # 180 mm
fig_h <- 8.27   # 210 mm

pdf_out <- file.path(OUT_DIR, "figB_cases.pdf")
png_out <- file.path(OUT_DIR, "figB_cases.png")

save_fig <- function(device_fn, ...) {
  device_fn(...)
  grid.newpage()
  # Row 1 (top half): KIF21B arch (left) + pLDDT trace (right)
  print(p_kif_arch,    vp = viewport(x=0.00, y=0.52, width=0.50, height=0.48,
                                     just=c("left","bottom")))
  print(p_kif_plddt,   vp = viewport(x=0.50, y=0.52, width=0.50, height=0.48,
                                     just=c("left","bottom")))
  # Row 2 (bottom half): NDUFS4 arch (left) + evidence panel (right)
  print(p_ndufs4_arch, vp = viewport(x=0.00, y=0.00, width=0.50, height=0.50,
                                     just=c("left","bottom")))
  print(p_ndufs4_ev,   vp = viewport(x=0.50, y=0.00, width=0.50, height=0.50,
                                     just=c("left","bottom")))
  dev.off()
}

tryCatch(
  save_fig(cairo_pdf, pdf_out, width=fig_w, height=fig_h),
  error=function(e) save_fig(pdf, pdf_out, width=fig_w, height=fig_h, useDingbats=FALSE)
)
cat(sprintf("Saved PDF: %s\n", pdf_out))

save_fig(png, png_out,
         width=round(fig_w*300), height=round(fig_h*300), res=300, bg="white")
cat(sprintf("Saved PNG: %s\n", png_out))
