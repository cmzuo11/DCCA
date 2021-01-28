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
Calculate_TF_score <- function( input_file, input_cellMeta, out_file, species = "mouse"){
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

#3. infer TF-TG relationship
Infer_network <- function(rna_data, atac_data, gene_loci_data, cluster, match_data){

  library("rsq")
  library("MASS")
  library("glmnetUtils")

  temp_rna_data_re = rna_data
  rela_atac_data   = atac_data

  inter_TFs  = intersect(row.names(rna_data), colnames(match_data))
  used_locus = match_data[,match(inter_TFs, colnames(match_data))]
  cor_loci   = NULL

  for(ii in 1:dim(used_locus)[2])
  {
    cor_loci = unique(c(cor_loci, intersect(row.names(atac_data), row.names(match_data)[which(used_locus[,ii])]) ) )
  }

  cor_genes        = intersect(unique(gene_loci_data[,1][match(cor_loci, gene_loci_data[,3])]), row.names(rna_data))
  unique_cl        = unique(cluster)
  temp_rna_data    = temp_rna_data_re[match(cor_genes, row.names(temp_rna_data_re)),]
  TF_loci_lists1   = TF_loci_lists2 = list()
  TF_list1 = loci_list = list()
  adjusted_r2      = list()
  all_gene_tgs     = NULL

  for(i in 1:length(cor_genes))
  {
    inter_map_loci   = intersect(gene_loci_data[,3][which(gene_loci_data[,1]==cor_genes[i])], cor_loci)
    match_locis_data = match_data[match(inter_map_loci, row.names(match_data)),]
    region_tfs       = list()
    used_losuc       = unique_used_tfs = NULL
    if(length(inter_map_loci)>1)
    {
      for(zz in 1:dim(match_locis_data)[1])
      {
        temp_comms = intersect(colnames(match_locis_data)[which(match_locis_data[zz,])], row.names(rna_data))
        if(length(temp_comms)>0)
        {
          region_tfs[[zz]] = temp_comms
          used_losuc       = c(used_losuc, zz)
          unique_used_tfs  = unique(c(unique_used_tfs, temp_comms))
        }
      }
    }else{
      temp_comms       = intersect(colnames(match_locis_data)[which(match_locis_data)], row.names(rna_data))
      if(length(temp_comms)>0)
      {
        region_tfs[[1]] = unique_used_tfs = temp_comms
        used_losuc      = 1
      }
    }
    temp_TF_loci_lists1 = temp_TF_loci_lists2 = list()
    temp_adjusted_r2    = list()
    temp_TF_list1       = list()
    temp_loci_list      = list()
    if(length(region_tfs)>0)
    {
      for(j in 1:length(unique_cl))
      {
        temp_atac_loci_re = rela_atac_data[match(inter_map_loci[used_losuc], row.names(rela_atac_data)), which(unique_cl[j]==cluster)]
        bg_random         = random_correlation_test_bg(temp_rna_data_re[i,which(unique_cl[j]==cluster)], 
                                                       rela_atac_data[,which(unique_cl[j]==cluster)], 
                                                       temp_atac_loci_re, 1000)
        signin_ints       = which( ((bg_random[,2]<=0.05) ) )
        if(length(signin_ints)>0)
        {
          all_unique_tfss  = NULL
          region_tfs_temp  = list()
          for(zz in 1:length(signin_ints))
          {
            region_tfs_temp[[zz]] = region_tfs[[signin_ints[zz]]]
            all_unique_tfss       = unique(c(all_unique_tfss, region_tfs[[signin_ints[zz]]]))
          }
          if(length(all_unique_tfss)>1)
          {
            # temp_atac_loci = rela_atac_data[match(inter_map_loci[used_losuc][signin_ints], row.names(rela_atac_data)), which(unique_cl[j]==cluster)]
            temp_rna_tfss  = temp_rna_data_re[match(all_unique_tfss, row.names(temp_rna_data_re)), which(unique_cl[j]==cluster)]
            df_regression  = data.frame(TG = temp_rna_data_re[i, which(unique_cl[j]==cluster)], as.data.frame(t(temp_rna_tfss)))
            
            temp_form = paste("TG", "~", all_unique_tfss[1])
            for( kk in 2:length(all_unique_tfss) )
            {
              temp_form = paste(temp_form, "+", all_unique_tfss[kk] )
            }
            
            res <- try(cv.glmnet( formula(temp_form), data = df_regression, lower.limits = 0, 
                                  alpha=0, type.measure = "mse"), silent = T)
            if(inherits(res, "try-error"))
            {
              next
            }
            apply_nos = apply(df_regression, 2, function(x){length(which(x>0))})
            if(length(which(apply_nos>3)) > 3 )
            {
              temp_gene_reg  = cv.glmnet( formula(temp_form), data = df_regression, type.measure = "mse",
                                          lower.limits = 0, alpha=0, grouped=TRUE)
              tmp_coeffs     = coef(temp_gene_reg, s = "lambda.min")
              infer_tf_tgs   = tmp_coeffs@Dimnames[[1]][tmp_coeffs@i + 1][which(tmp_coeffs@x>0)]
              adjuste_r2     = temp_gene_reg$glmnet.fit$dev.ratio[which(temp_gene_reg$glmnet.fit$lambda == temp_gene_reg$lambda.min)]
              temp_TF_loci_lists_info1 = temp_TF_loci_lists_info2 = NULL
              temp_TF_lists_info = temp_loci_lists_info      = NULL
              if(length(infer_tf_tgs)>0)
              {
                for(mmm in 1:length(infer_tf_tgs))
                {
                  if( regexpr(":",infer_tf_tgs[mmm])[1] > -1 )
                  {
                    temp_s = unlist(strsplit(infer_tf_tgs[mmm], ":"))
                    if(length(intersect(temp_s[1], unique_used_tfs))>0)
                    {
                      temp_TF_loci_lists_info1 = c(temp_TF_loci_lists_info1, temp_s[1])
                      temp_s1 = unlist(strsplit(temp_s[2], "[.]"))
                      temp_TF_loci_lists_info2 = c(temp_TF_loci_lists_info2, paste(temp_s1[1], ":", temp_s1[2],"-", temp_s1[3], sep = ""))
                    }else{
                      temp_TF_loci_lists_info1 = c(temp_TF_loci_lists_info1, temp_s[2])
                      temp_s1 = unlist(strsplit(temp_s[1], "[.]"))
                      temp_TF_loci_lists_info2 = c(temp_TF_loci_lists_info2, paste(temp_s1[1], ":", temp_s1[2],"-", temp_s1[3], sep = ""))
                    }
                  }else{
                    if(!(regexpr("Intercept", infer_tf_tgs[mmm])[1]>-1))
                    {
                      if(length(intersect(infer_tf_tgs[mmm], row.names(rna_data)))>0)
                      {
                        temp_TF_lists_info = c(temp_TF_lists_info, infer_tf_tgs[mmm])
                      }else{
                        temp_s1 = unlist(strsplit(infer_tf_tgs[mmm], "[.]"))
                        temp_loci_lists_info = c(temp_loci_lists_info, paste(temp_s1[1], ":", temp_s1[2],"-", temp_s1[3], sep = ""))
                      }
                    }
                  }
                }
                temp_TF_loci_lists1[[j]] = temp_TF_loci_lists_info1
                temp_TF_loci_lists2[[j]] = temp_TF_loci_lists_info2
                temp_TF_list1[[j]]       = temp_TF_lists_info
                temp_loci_list[[j]]      = temp_loci_lists_info
                temp_adjusted_r2[[j]]    = adjuste_r2
              }else{
                temp_TF_loci_lists1[[j]] = temp_TF_loci_lists2[[j]] = NA
                temp_adjusted_r2[[j]]    = NA
                temp_TF_list1[[j]]       = NA
                temp_loci_list[[j]]      = NA
              }
            }
          }
        }
      }
    }
    TF_loci_lists1[[i]] = temp_TF_loci_lists1
    TF_loci_lists2[[i]] = temp_TF_loci_lists2
    TF_list1[[i]]       = temp_TF_list1
    loci_list[[i]]      = temp_loci_list
    adjusted_r2[[i]]    = temp_adjusted_r2
    all_gene_tgs        = c(all_gene_tgs, cor_genes[i])
  }
  return(list(TF_loci_lists1, TF_loci_lists2, TF_list1, loci_list, adjusted_r2, all_gene_tgs))
}


random_correlation_test_bg <- function(data1, data2, temp_data2, time2 = 1000)
{
  sample_l2  = sample(1:dim(data2)[1], time2)
  cor_temp = NULL
  temp_da1 = data1
  for(kkk in sample_l2)
  {
    temp_da2 = data2[kkk,]
    if((length(which(temp_da1>0))>2) && ( length(which(temp_da2>0))>2 ) )
    {
      temp_cor = cor.test(temp_da1, temp_da2, method = "pearson")
      if(!is.na(temp_cor$p.value))
      {
        if(temp_cor$p.value<0.05)
        {
          cor_temp = c(cor_temp, temp_cor$estimate)
        }else{
          cor_temp = c(cor_temp, 0)
        }
      }else{
        cor_temp = c(cor_temp, 0)
      }
      
    }else{
      cor_temp = c(cor_temp, 0)
    }
  }
  
  mean_cor_cell = mean(cor_temp)
  sd_cor_cell   = sd(cor_temp)
  
  temp_corss   = NULL
  temp_pvalues = NULL
  
  for(kk in 1:dim(temp_data2)[1])
  {
    if((length(which(data1>0))>2) && ( length(which(temp_data2[kk,]>0))>2 ) )
    {
      temp_cor = cor.test(data1, temp_data2[kk,], method = "pearson")
      if(!is.na(temp_cor$p.value))
      {
        if(temp_cor$p.value<0.05)
        {
          temp_corss    = c(temp_corss, temp_cor$estimate)
          temp_pvalues  = c(temp_pvalues, 1-(pnorm(temp_cor$estimate, mean_cor_cell, sd_cor_cell )))
        }else{
          temp_corss = c(temp_corss, 0)
          temp_pvalues  = c(temp_pvalues, 1)
        }
      }else{
        temp_corss = c(temp_corss, 0)
        temp_pvalues  = c(temp_pvalues, 1)
      }
    }else{
      temp_corss = c(temp_corss, 0)
      temp_pvalues  = c(temp_pvalues, 1)
    }
  }
  
  return(cbind(temp_corss, temp_pvalues))
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


