#Processing---variable selection
Select_Loci_by_vargenes <- function(Vargenes, all_loci, width = 2000, species = "mouse"){
  # find all loci within the genebody + width upstream Vargenes
  
  if (species == "mouse"){
    mouse = biomaRt::useMart("ensembl", dataset = "mmusculus_gene_ensembl", host = "useast.ensembl.org")
    loci  = biomaRt::getBM(attributes = c( "mgi_symbol","chromosome_name",'start_position','end_position'), 
                           filters = "mgi_symbol", values = Vargenes, mart = mouse)
  }else{
    human = biomaRt::useMart("ensembl", dataset = "hsapiens_gene_ensembl", host = "useast.ensembl.org")
    loci  = biomaRt::getBM(attributes = c( "hgnc_symbol","chromosome_name",'start_position','end_position'), 
                           filters = "hgnc_symbol", values = Vargenes, mart = human)
  }
  loci     = loci[loci$chromosome_name %in% c(1:20), 'X']
  temp_s   = NULL
  for(zz in 1:dim(loci)[1])
  {
    if(loci[[3]][zz] > width)
    {
      temp_s = c(temp_s, (loci[[3]][zz]-width) )
    }else{
      temp_s = c(temp_s, 0 )
    }
  }
  loci$start_position = temp_s
  loci$end_position   = loci$end_position
  loci_bed            = paste0("chr",loci$chromosome_name, ":", loci$start_position, "-",loci$end_position)
  loci_bed_sort       = bedr::bedr.sort.region(loci_bed)
  
  all_loci            = all_loci[grepl(paste(paste0("chr", c(1:20, 'X'), ":"), collapse="|"), all_loci)]
  all_loci_sort       = bedr::bedr.sort.region(all_loci)
  is_region           = bedr::in.region(all_loci_sort, loci_bed_sort)
  nearby_loci         = loci_bed_sort[is_region]
  
  #find corresponding relationship
  a.int1        = bedr::bedr(input = list(a = loci_bed_sort , b = all_loci_sort ), method = "intersect", params ="-loj -header")
  df_genes_loci = data.frame(Gene   = as.character(loci[[1]])[match(as.character(a.int1[[1]]), loci_bed)], 
                             Region = as.character(a.int1[[1]]), 
                             Loci   = paste(a.int1$V4, ":", a.int1$V5, "-", a.int1$V6, sep = ""))
  
  return(list(nearby_loci, df_genes_loci))
}


#1. plot cell embeddidngs based on latent features for each omics data
plot_umap_embeddings <- function(File_latent_rna, File_latent_atac, File_cellMeta, Out_pdf_name ){
  #File_latent_rna indicates  "./Example_test/scRNA-latent.csv"
  #File_latent_atac indicates "./Example_test/scATAC-latent.csv"
  #File_cellMeta indicates    "./Example_test/cell_metadata.txt"
  #Out_pdf_name indicaets     "./Example_test/scRNA_umap.pdf"
  
  colos = c("BJ" = "chartreuse3", "GM" = "coral2", "H1" = "dodgerblue", "K562"= "cyan3")
  
  library("uwot")
  library("ggplot2")
  
  latent_rna  = as.matrix(read.csv( File_latent_rna, header = T, row.names = 1 ))
  latent_atac = as.matrix(read.csv( File_latent_atac, header = T, row.names = 1 ))
  cell_meta   = as.matrix(read.table(File_cellMeta, header = T, row.names = 1))
  umap_rna    = umap( latent_rna  )
  umap_atac   = umap( latent_atac )
  df_data     = data.frame(UMAP1_rna   = umap_rna[,1], UMAP2_rna  = umap_rna[,2],
                           UMAP1_atac  = umap_rna[,1], UMAP2_atac = umap_rna[,2],
                           cell_type   = cell_meta[,1])
  
  pdf(Out_pdf_name, width = 10, height = 10)
  p1 = ggplot(data = df_data, aes(x = UMAP1_rna, y =UMAP2_rna , color = as.factor(cell_type) )) +
    geom_point()+
    scale_color_manual(values = colos)+
    theme_classic(base_size = 15)+
    theme(axis.text.x  = element_text(face ="bold", size = 10, color = "black"),
          axis.text.y  = element_text(face ="bold", size = 10, color = "black"),
          axis.title.x = element_text(face ="bold", size = 12, color = "black"),
          axis.title.y = element_text(face ="bold", size = 12, color = "black"),
          legend.text  = element_text(face ="bold", size = 10, color = "black"),
          legend.title = element_text(face ="bold", size = 12, color = "black"),
          legend.key   = element_blank(),
          legend.position = "none")+
    labs(title = "", x = "UMAP_1", y = "UMAP_2")
  plot(p1)
  
  p2 = ggplot(data = df_data, aes(x = UMAP1_atac, y =UMAP2_atac , color = as.factor(cell_type) )) +
    geom_point()+
    scale_color_manual(values = colos)+
    theme_classic(base_size = 15)+
    theme(axis.text.x  = element_text(face ="bold", size = 10, color = "black"),
          axis.text.y  = element_text(face ="bold", size = 10, color = "black"),
          axis.title.x = element_text(face ="bold", size = 12, color = "black"),
          axis.title.y = element_text(face ="bold", size = 12, color = "black"),
          legend.text  = element_text(face ="bold", size = 10, color = "black"),
          legend.title = element_text(face ="bold", size = 12, color = "black"),
          legend.key   = element_blank(),
          legend.position = "none")+
    labs(title = "", x = "UMAP_1", y = "UMAP_2")
  plot(p2)
  dev.off()
  
}

#2. calculate TF score for scATAC-seq data
Calculate_TF_score <- function( input_file, input_cellMeta, out_file, cell_meta = NULL, species = "mouse"){
  #input_file indicates "./Example_test/scATAC-norm.csv"
  #input_cellMeta indicates "./Example_test/cell_metadata.txt"
  #out_file indicates "./Example_test/atac_denoise_deviation.txt"
  
  ##load required packages
  library(chromVAR)
  library(motifmatchr)
  library(Matrix)
  library(SummarizedExperiment)
  library(BiocParallel)
  set.seed(2020)
  library("BSgenome")
  library("JASPAR2016")
  
  data1     = t(as.matrix(read.csv( input_file, header = T, row.names = 1 )))
  peaks     = reChromName(row.names(data1))
  cell_meta = as.matrix(read.table(input_cellMeta, header = T, row.names = 1))
  
  ##processing steps
  data_li   = split_chr_loc(peaks)
  rowRanges = GRanges(seqnames   = data_li[[1]],
                      ranges     = data_li[[2]],
                      feature_id = peaks)
  
  colData   = DataFrame(Cell_type = cell_meta[,2],
                        row.names = colnames(data1))
  
  fragment_counts = SummarizedExperiment(assays=SimpleList(counts=data1),
                                         rowRanges=rowRanges,
                                         colData=colData)
  if(species=="mouse")
  {
    library("BSgenome.Mmusculus.UCSC.mm10")
    fragment_counts = addGCBias(fragment_counts, genome = BSgenome.Mmusculus.UCSC.mm10)
    counts_filtered = filterPeaks(fragment_counts, non_overlapping = TRUE)
    motifs          = getJasparMotifs(species = "Mus musculus")
    
    motif_ix <- matchMotifs(motifs, counts_filtered,
                            genome = BSgenome.Mmusculus.UCSC.mm10)
    kmer_ix  <- matchKmers(6, counts_filtered, 
                           genome = BSgenome.Mmusculus.UCSC.mm10)
  }else{
    library("BSgenome.Hsapiens.UCSC.hg38")
    fragment_counts = addGCBias(fragment_counts, genome = BSgenome.Hsapiens.UCSC.hg38)
    counts_filtered = filterPeaks(fragment_counts, non_overlapping = TRUE)
    motifs          = getJasparMotifs(species = "Homo sapiens")
    
    motif_ix        = matchMotifs(motifs, counts_filtered,
                                  genome = BSgenome.Hsapiens.UCSC.hg38)
    kmer_ix         = matchKmers(6, counts_filtered, 
                                 genome = BSgenome.Hsapiens.UCSC.hg38)
  }
  bg  = getBackgroundPeaks(object = counts_filtered)
  dev = computeDeviations(object = counts_filtered, annotations = motif_ix,
                          background_peaks = bg)
 
  scale_devi = Scale_data(dev@assays@data$deviations, ymin = -1, ymax = 1)
  write.table(scale_devi, file = out_file, sep = "\t", quote = F)
}


split_chr_loc_three <- function(Names){
  temp_ch     = NULL
  temp_start  = NULL
  temp_end    = NULL
  for(kk in 1:length(Names))
  {
    temp_s   = unlist(strsplit(Names[kk], ":"))
    temp_s_1 = unlist(strsplit(temp_s[2], "-"))
    
    temp_ch    = c(temp_ch,    temp_s[1] )
    temp_start = c(temp_start, temp_s_1[1] )
    temp_end   = c(temp_end,   temp_s_1[2] )
  }
  
  return(list(temp_ch, temp_start, temp_end))
}

reChromName <- function(Names){
  temp_names = NULL
  for(ii in 1:length(Names))
  {
    tems = unlist(strsplit(Names[ii],"[.]"))
    temp_names = c(temp_names, paste(tems[1], ":", tems[2], "-", tems[3], sep = ""))
  }
  return(temp_names)
}

Scale_data <- function(devi_data, ymin = -1, ymax = 1){
  scale_devi = NULL
  for(i in 1:dim(devi_data)[1])
  {
    min_va   = min(devi_data[i,], na.rm = T)
    max_va   = max(devi_data[i,], na.rm = T)
    temp_das = (ymax-ymin)*(devi_data[i,] - min_va)/(max_va-min_va) + ymin
    scale_devi = rbind(scale_devi, temp_das )
  }
  return(scale_devi)
}


