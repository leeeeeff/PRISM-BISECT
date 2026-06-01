## ----style, eval=TRUE, echo=FALSE, results='asis'--------------------------
BiocStyle::latex()

## ----setup_knitr, include=FALSE, cache=FALSE-------------------------------
library(knitr)
opts_chunk$set(cache = FALSE, warning = FALSE, out.width = "7cm", fig.width = 7, out.height = "7cm", fig.height = 7)

## ----news, eval = FALSE----------------------------------------------------
#  news(package = "DRIMSeq")

## ----DSpasilla1------------------------------------------------------------
library(PasillaTranscriptExpr)

data_dir  <- system.file("extdata", package = "PasillaTranscriptExpr")

## Load metadata
pasilla_metadata <- read.table(file.path(data_dir, "metadata.txt"), 
  header = TRUE, as.is = TRUE)

## Load counts
pasilla_counts <- read.table(file.path(data_dir, "counts.txt"), 
  header = TRUE, as.is = TRUE)


## ----DSlibrary, message=FALSE----------------------------------------------
library(DRIMSeq)

## ----DSdmDSdata_create-----------------------------------------------------
pasilla_samples <- data.frame(sample_id = pasilla_metadata$SampleName, 
  group = pasilla_metadata$condition)
levels(pasilla_samples$group)

d <- dmDSdata(counts = pasilla_counts, samples = pasilla_samples)
d
head(counts(d), 3)
head(samples(d), 3)

## ----DSdmDSdata_plot-------------------------------------------------------
plotData(d)

## ----DSdmDSdata_subset-----------------------------------------------------
gene_id_subset <- readLines(file.path(data_dir, "gene_id_subset.txt"))
d <- d[names(d) %in% gene_id_subset, ]
d

## ----DSdmFilter------------------------------------------------------------
# Check what is the minimal number of replicates per condition
table(samples(d)$group)

d <- dmFilter(d, min_samps_gene_expr = 7, min_samps_feature_expr = 3,
  min_gene_expr = 10, min_feature_expr = 10)

## ----DSdmPrecision_design--------------------------------------------------
## Create the design matrix
design_full <- model.matrix(~ group, data = samples(d))
design_full

## ----DSdmPrecision---------------------------------------------------------
## To make the analysis reproducible
set.seed(123)
## Calculate precision
d <- dmPrecision(d, design = design_full)
d
head(mean_expression(d), 3)
common_precision(d)
head(genewise_precision(d))

## ----DSdmPrecision_plot1---------------------------------------------------
plotPrecision(d)

## ----DSdmPrecision_plot2---------------------------------------------------
library(ggplot2)
ggp <- plotPrecision(d)
ggp + geom_point(size = 4)

## ----DSdmFit---------------------------------------------------------------
d <- dmFit(d, design = design_full, verbose = 1)
d

## Get fitted proportions
head(proportions(d))
## Get the DM regression coefficients (gene-level) 
head(coefficients(d))
## Get the BB regression coefficients (feature-level) 
head(coefficients(d), level = "feature")

## ----DSdmTest1-------------------------------------------------------------
d <- dmTest(d, coef = "groupKD", verbose = 1)
design(d)
head(results(d), 3)

## ----DSdmTest2-------------------------------------------------------------
design_null <- model.matrix(~ 1, data = samples(d))
design_null
d <- dmTest(d, design = design_null)
head(results(d), 3)

## ----DSdmTest3-------------------------------------------------------------
contrast <- c(0, 1)
d <- dmTest(d, contrast = contrast)
design(d)
head(results(d), 3)

## ----DSdmTest_results------------------------------------------------------
head(results(d, level = "feature"), 3)

## ----DSdmTest_plot---------------------------------------------------------
plotPValues(d)
plotPValues(d, level = "feature")

## ----DSdmLRT_plotProportions, out.width = "14cm", fig.width = 14-----------
res <- results(d)
res <- res[order(res$pvalue, decreasing = FALSE), ]
top_gene_id <- res$gene_id[1]

plotProportions(d, gene_id = top_gene_id, group_variable = "group")
plotProportions(d, gene_id = top_gene_id, group_variable = "group", 
  plot_type = "lineplot")
plotProportions(d, gene_id = top_gene_id, group_variable = "group", 
  plot_type = "ribbonplot")


## ----stageR, eval = FALSE--------------------------------------------------
#  library(stageR)
#  
#  ## Assign gene-level pvalues to the screening stage
#  pScreen <- results(d)$pvalue
#  names(pScreen) <- results(d)$gene_id
#  
#  ## Assign transcript-level pvalues to the confirmation stage
#  pConfirmation <- matrix(results(d, level = "feature")$pvalue, ncol = 1)
#  rownames(pConfirmation) <- results(d, level = "feature")$feature_id
#  
#  ## Create the gene-transcript mapping
#  tx2gene <- results(d, level = "feature")[, c("feature_id", "gene_id")]
#  
#  ## Create the stageRTx object and perform the stage-wise analysis
#  stageRObj <- stageRTx(pScreen = pScreen, pConfirmation = pConfirmation,
#    pScreenAdjusted = FALSE, tx2gene = tx2gene)
#  
#  stageRObj <- stageWiseAdjustment(object = stageRObj, method = "dtu",
#    alpha = 0.05)
#  
#  getSignificantGenes(stageRObj)
#  
#  getSignificantTx(stageRObj)
#  
#  padj <- getAdjustedPValues(stageRObj, order = TRUE,
#    onlySignificantGenes = FALSE)
#  
#  head(padj)
#  

## ----DRIMSeq_batch---------------------------------------------------------
pasilla_samples2 <- data.frame(sample_id = pasilla_metadata$SampleName, 
  group = pasilla_metadata$condition, 
  library_layout = pasilla_metadata$LibraryLayout)

d2 <- dmDSdata(counts = pasilla_counts, samples = pasilla_samples2)

## Subsetting to a vignette runnable size
d2 <- d2[names(d2) %in% gene_id_subset, ]

## Filtering
d2 <- dmFilter(d2, min_samps_gene_expr = 7, min_samps_feature_expr = 3,
  min_gene_expr = 10, min_feature_expr = 10)

## Create the design matrix
design_full2 <- model.matrix(~ group + library_layout, data = samples(d2))
design_full2

## To make the analysis reproducible
set.seed(123)

## Calculate precision
d2 <- dmPrecision(d2, design = design_full2)

common_precision(d2)
head(genewise_precision(d2))

plotPrecision(d2)

## Fit proportions
d2 <- dmFit(d2, design = design_full2, verbose = 1)

## Test for DTU
d2 <- dmTest(d2, coef = "groupKD", verbose = 1)
design(d2)
head(results(d2), 3)

## Plot p-value distribution
plotPValues(d2)

## ----DRIMSeq_batch_plotProportions, out.width = "14cm", fig.width = 14-----
## Plot the top significant gene
res2 <- results(d2)
res2 <- res2[order(res2$pvalue, decreasing = FALSE), ]
top_gene_id2 <- res2$gene_id[1]
ggp <- plotProportions(d2, gene_id = top_gene_id2, group_variable = "group")
ggp + facet_wrap(~ library_layout)

## ----SQTLgeuvadis, message=FALSE-------------------------------------------
library(GeuvadisTranscriptExpr)

geuv_counts <- GeuvadisTranscriptExpr::counts
geuv_genotypes <- GeuvadisTranscriptExpr::genotypes
geuv_gene_ranges <- GeuvadisTranscriptExpr::gene_ranges
geuv_snp_ranges <- GeuvadisTranscriptExpr::snp_ranges


## ----SQTLlibrary, message=FALSE--------------------------------------------
library(DRIMSeq)

## ----SQTLdmSQTLdata_create, message=FALSE----------------------------------
colnames(geuv_counts)[c(1,2)] <- c("feature_id", "gene_id")
colnames(geuv_genotypes)[4] <- "snp_id"
geuv_samples <- data.frame(sample_id = colnames(geuv_counts)[-c(1,2)])

d <- dmSQTLdata(counts = geuv_counts, gene_ranges = geuv_gene_ranges,
  genotypes = geuv_genotypes, snp_ranges = geuv_snp_ranges, 
  samples = geuv_samples, window = 5e3)
d

## ----SQTLdmSQTLdata_plot---------------------------------------------------
plotData(d, plot_type = "features")
plotData(d, plot_type = "snps")
plotData(d, plot_type = "blocks")

## ----SQTLdmFilter----------------------------------------------------------
d <- dmFilter(d, min_samps_gene_expr = 70, min_samps_feature_expr = 5,
  minor_allele_freq = 5, min_gene_expr = 10, min_feature_expr = 10)

## ----SQTLdmPrecision-------------------------------------------------------
## To make the analysis reproducible
set.seed(123)
## Calculate precision
d <- dmPrecision(d)

plotPrecision(d)

## ----SQTLdmFit-------------------------------------------------------------
d <- dmFit(d)

## ----SQTLdmTest------------------------------------------------------------
d <- dmTest(d)
plotPValues(d)
head(results(d))

## ----SQTLplotProportions, out.width = "14cm", fig.width = 14---------------
res <- results(d)
res <- res[order(res$pvalue, decreasing = FALSE), ]

top_gene_id <- res$gene_id[1]
top_snp_id <- res$snp_id[1]

plotProportions(d, gene_id = top_gene_id, snp_id = top_snp_id)
plotProportions(d, gene_id = top_gene_id, snp_id = top_snp_id,
  plot_type = "boxplot2")

## ----sessionInfo-----------------------------------------------------------
sessionInfo()

