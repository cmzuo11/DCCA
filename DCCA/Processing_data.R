#Processing---variable selection for two-omics data
Create_Seurat_from_scRNA <- function(scRNA_data, nDim = 10, cell_meta_file,remove_HK = F){
  # scRNA_data: features * cells
  library("Seurat")

  if(remove_HK)
  {
    hk_genes = read.table("./Utilities/Housekeeping_gene.txt", header=F)[[1]]
  }

  Seurat_obj = CreateSeuratObject(scRNA_data[-match(hk_genes, row.names(scRNA_data)),], min.cells = 3, min.features = 10)
  Seurat_obj = NormalizeData(Seurat_obj, display.progress = F)
  Seurat_obj = FindVariableFeatures(Seurat_obj, selection.method = 'vst', nfeatures = 3000)
  Seurat_obj = ScaleData(Seurat_obj, verbose = FALSE)
  Seurat_obj = RunPCA(Seurat_obj, assay = "RNA",reduction.name = "pca_scRNA", 
                      reduction.key = "pca_scRNA_", verbose = FALSE)
  Seurat_obj = FindNeighbors( Seurat_obj, dims = 1:nDim, reduction = "pca_scRNA",graph.name = "RNA_neigh" )
  Seurat_obj = FindClusters( Seurat_obj, resolution = 0.4, graph.name = "RNA_neigh", verbose = FALSE )
  Seurat_obj = RunUMAP( Seurat_obj, dims = 1:nDim, reduction = "pca_scRNA", reduction.name = "umap_rna1")

  cell_meta = read.table(cell_meta_file, header=T) # i.e., cell_meta_file for './Example_test/cell_metadata.txt'

  Seurat_obj[["Cell_type"]] = cell_meta[,2]

  return(Seurat_obj)
}

Normalized_data_by_seurat <- function(scRNA_data)
{
  Seu_obj  = CreateSeuratObject(scRNA_data, min.cells = 1, min.features = 1)
  Seu_obj  = NormalizeData(Seu_obj, display.progress = F)
  Seu_obj  = ScaleData(Seu_obj, display.progress = F)
  
  norm_rna = Seu_obj[["RNA"]]@scale.data

  return(norm_rna)
}

Select_HVGs_from_scRNA <- function(Seurat_obj, hvg_method = 'vst', nFeature = 1000){

  Seurat_obj = FindVariableFeatures(Seurat_obj, selection.method = hvg_method, nfeatures = nFeature )
  HVG_genes  = VariableFeatures(Seurat_obj)
  return( HVG_genes )
}

Select_Loci_by_vargenes <- function(Vargenes, scATAC_data, width = 2000, species = "mouse"){
  # Vargenes: highly variable genes by scRNA-seq data
  # scATAC_data: features * cells
  # find all loci within the genebody + width upstream Vargenes

  all_loci   = row.names(scATAC_data)
  
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


#Plot_umap_embeddings('./scRNA-latent.csv', './scATAC-latent.csv', './cell_metadata.txt','./Latent_umap.pdf' )

#1. plot cell embeddidngs based on latent features for each omics data
Plot_umap_embeddings <- function(latent_rna, latent_atac, cellMeta, pdf_name, color_define){
  #File_latent_rna and File_latent_atac indicate latent files for scRNA-seq and scATAC 
  #File_cellMeta indicates    "./Example_test/cell_metadata.txt"
  #Out_pdf_name indicaets     "./Example_test/scRNA_umap.pdf"
  
  library("uwot")
  library("ggplot2")
  library("ggpubr")
  
  latent_rna  = as.matrix(read.csv( latent_rna, header = T, row.names = 1 ))
  latent_atac = as.matrix(read.csv( latent_atac, header = T, row.names = 1 ))
  cell_meta   = as.matrix(read.table(cellMeta, header = T, row.names = 1))
  umap_rna    = umap( latent_rna  )
  umap_atac   = umap( latent_atac )
  df_data     = data.frame(UMAP1_rna   = umap_rna[,1], UMAP2_rna   = umap_rna[,2],
                           UMAP1_atac  = umap_atac[,1], UMAP2_atac = umap_atac[,2],
                           cell_type   = cell_meta[,1])
  
  pdf(pdf_name, width = 15, height = 8)

  p1 = ggplot(data = df_data, aes(x = UMAP1_rna, y =UMAP2_rna , color = as.factor(cell_type) )) +
       geom_point()+
       scale_color_manual(values = color_define)+
       labs(title = "Zx (scRNA-seq)", x = "UMAP_1", y = "UMAP_2")+
       theme_format
    
  p2 = ggplot(data = df_data, aes(x = UMAP1_atac, y =UMAP2_atac , color = as.factor(cell_type) )) +
       geom_point()+
       scale_color_manual(values = color_define)+
       labs(title = "Zy (scATAC-seq)", x = "UMAP_1", y = "UMAP_2")+
       theme_format

  figure  = ggarrange(p1, p2, ncol = 2, nrow = 1)

  plot(figure)
  dev.off()
}

### Construct association between TF and loci
TF_loci_mapping <- function(atac_data, species = "human"){
  library(chromVAR)
  library(motifmatchr)
  library(Matrix)
  library(SummarizedExperiment)
  library(BiocParallel)
  set.seed(2020)
  library("BSgenome")
  library("JASPAR2016")

  peaks     = reChromName(row.names(atac_data))
  
  ##processing steps
  data_li   = split_chr_loc(peaks)
  rowRanges = GRanges(seqnames   = data_li[[1]],
                      ranges     = data_li[[2]],
                      feature_id = peaks)
  
  fragment_counts = SummarizedExperiment(assays=SimpleList(counts=atac_data),
                                         rowRanges=rowRanges)
  if(species=="mouse")
  {
    library("BSgenome.Mmusculus.UCSC.mm10")
    fragment_counts = addGCBias(fragment_counts, genome = BSgenome.Mmusculus.UCSC.mm10)
    counts_filtered = filterPeaks(fragment_counts, non_overlapping = TRUE)
    motifs          = getJasparMotifs(species = "Mus musculus")
    
    motif_ix <- matchMotifs(motifs, counts_filtered,
                            genome = BSgenome.Mmusculus.UCSC.mm10)
  }else{
    library("BSgenome.Hsapiens.UCSC.hg38")
    fragment_counts = addGCBias(fragment_counts, genome = BSgenome.Hsapiens.UCSC.hg38)
    counts_filtered = filterPeaks(fragment_counts, non_overlapping = TRUE)
    motifs          = getJasparMotifs(species = "Homo sapiens")
    
    motif_ix        = matchMotifs(motifs, counts_filtered,
                                  genome = BSgenome.Hsapiens.UCSC.hg38)
  }

  match_data            = as.matrix(motif_ix@assays@data[[1]])
  row.names(match_data) = names(motif_ix@rowRanges)
  colnames(match_data)  = motif_ix@colData[[1]]

  return(match_data)

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
  # rna_data:       gene * cell
  # rna_data:       loci * cell
  # gene_loci_data: gene, genomics_region, loci
  # match_data:     loci * TF

  library("rsq")
  library("MASS")
  library("glmnetUtils")

  inter_tfs  = intersect(row.names(rna_data), colnames(match_data))
  used_locus = match_data[ , match(inter_tfs, colnames(match_data))]

  cor_loci   = NULL
  for(ii in 1:dim(used_locus)[2])
  {
    cor_loci = unique(c(cor_loci, intersect(row.names(atac_data), row.names(match_data)[which(used_locus[,ii])]) ) )
  }

  cor_genes        = intersect(unique(gene_loci_data[,1][match(cor_loci, gene_loci_data[,3])]), row.names(rna_data))
  temp_rna_data    = rna_data[match(cor_genes, row.names(rna_data)),]
  unique_cl        = unique(cluster)

  TF_loci_lists1   = TF_loci_lists2 = list()
  TF_list1         = loci_list      = list()
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

    temp_adjusted_r2    = list()
    temp_TF_list1       = list()
    temp_loci_list      = list()

    if(length(region_tfs)>0)
    {

      for(j in 1:length(unique_cl))
      {
        cluster_ints      = which(unique_cl[j]==cluster)
        temp_atac_data    = atac_data[match(inter_map_loci[used_losuc], row.names(atac_data)), cluster_ints]

        bg_random         = random_correlation_test_bg(temp_rna_data[i,cluster_ints], atac_data[,cluster_ints], temp_atac_data)
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
            temp_rna_tfss  = rna_data[match(all_unique_tfss, row.names(rna_data)), cluster_ints]
            df_regression  = data.frame(TG = temp_rna_data[i, cluster_ints], as.data.frame(t(temp_rna_tfss)))
            temp_form      = paste("TG", "~", all_unique_tfss[1])

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
              
              temp_TF_lists_info   = NULL
              temp_loci_lists_info = NULL

              if(length(infer_tf_tgs)>0)
              {
                for(mmm in 1:length(infer_tf_tgs))
                {
                  if(!(regexpr("Intercept", infer_tf_tgs[mmm])[1]>-1))
                  {
                    if(length(intersect(infer_tf_tgs[mmm], row.names(rna_data)))>0)
                    {
                      temp_TF_lists_info   = c(temp_TF_lists_info, infer_tf_tgs[mmm])

                    }else{
                      temp_s1              = unlist(strsplit(infer_tf_tgs[mmm], "[.]"))
                      temp_loci_lists_info = c(temp_loci_lists_info, paste(temp_s1[1], ":", temp_s1[2],"-", temp_s1[3], sep = ""))

                    }
                  }
                }

                temp_TF_list1[[j]]       = temp_TF_lists_info
                temp_loci_list[[j]]      = temp_loci_lists_info
                temp_adjusted_r2[[j]]    = adjuste_r2

              }else{

                temp_TF_list1[[j]]       = NA
                temp_loci_list[[j]]      = NA
                temp_adjusted_r2[[j]]    = NA

              }
            }
          }
        }
      }
    }

    TF_list1[[i]]    = temp_TF_list1
    loci_list[[i]]   = temp_loci_list
    adjusted_r2[[i]] = temp_adjusted_r2
    all_gene_tgs     = c(all_gene_tgs, cor_genes[i])

  }

  names(TF_list1)    = unique_cl
  names(loci_list)   = unique_cl
  names(adjusted_r2) = unique_cl
  return(list(TF_list1, loci_list, adjusted_r2, all_gene_tgs))
}

Generate_cell_type_regulon <- function(infer_net)
{
  net_nos       = NULL
  cell_type_net = list()
  
  for(i in 1:length(infer_net[[1]]))
  {
    if(length(infer_net[[1]][[i]])>0)
    {
      for(j in 1:length(infer_net[[1]][[i]]))
      {
        if(length(infer_net[[1]][[i]][[j]])>0)
        {
          cell_type_net[[j]] = c(cell_type_net[[j]], paste( infer_net[[1]][[i]][[j]], "-", infer_net[[4]][[i]], sep=""))
        }
      }
    }
  }
  
  return(cell_type_net)
}


Activity_function <- function(TF_data, TG_data)
{
  temp_sums   = rep(0, dim(TG_data)[2])

  for(i in 1:dim(TG_data)[1])
  {
    temp_sums = temp_sums + TG_data[i,]*(apply(TG_data, 2, sum))
  }

  problity   = NULL
  for( i in 1:dim(TG_data)[1])
  {
    problity = rbind(problity, (TG_data[i,]*TF_data)/temp_sums)
  }
  
  temp_prob = NULL
  for(i in 1:dim(problity)[2])
  {
    temp_prob = cbind(temp_prob,  sum(problity[,i]*log(problity[,i])))
  }

  return( -(temp_prob/log(dim(problity)[1])) )

}

Regulon_activity <- function(Cell_specific_net, rna_data, cell_cluster)
{
  unique_cl      = unique(cell_cluster)
  cell_type_acti = list()
  
  tf_s           = NULL

  for(i in 1:length(Cell_specific_net))
  {
    if(length(Cell_specific_net[[i]])>0)
    {
      for(j in 1:length(Cell_specific_net[[i]]))
      {
        temps_s = unlist(strsplit(Cell_specific_net[[i]][j], "-"))
        tf_s    = c(tf_s, temps_s[1])
      }
    }
    unique_tfs = unique(tf_s)
  }
  
  for(i in 1:length(Cell_specific_net))
  {
    tf_s        = NULL
    tg_s        = NULL
    acti_matrix = NULL
    
    if(length(Cell_specific_net[[i]])>0)
    {
      for(j in 1:length(Cell_specific_net[[i]]))
      {
        temps_s = unlist(strsplit(Cell_specific_net[[i]][j], "-"))
        tf_s    = c(tf_s, temps_s[1])
        tg_s    = c(tg_s, temps_s[2])
      }
      
      for(k in 1:length(unique_tfs))
      {
        temp_tgs  = unique(tg_s[which(tf_s==unique_tfs[k])])

        if(length(temp_tgs)>1)
        {
          temp_rnas      = rna_data[match(temp_tgs, row.names(rna_data) ),which(cell_cluster==unique_cl[i])]
          temp_tfs       = rna_data[which(row.names(rna_data)==unique_tfs[k]), which(cell_cluster==unique_cl[i])]
          Activity_cells = Activity_function(temp_tfs, temp_rnas)

        }else{
          Activity_cells = rep(0, length(which(cell_cluster==unique_cl[i])))

        }
        acti_matrix    = rbind(acti_matrix, Activity_cells)

      }
      
      colnames(acti_matrix)  = colnames(rna_data)[which(cell_cluster==unique_cl[i])]
      cell_type_acti[[i]]    =  acti_matrix
    }else{
      acti_matrix            = matrix(0, length(unique_tfs), length(which(cell_cluster==unique_cl[i])))
      colnames(acti_matrix)  = colnames(rna_data)[which(cell_cluster==unique_cl[i])]
      cell_type_acti[[i]]    = acti_matrix
    }
  }
  
  tfs_acti_matrix = NULL
  cell_ss         = NULL
  cellNamess      = NULL

  for(zz in 1:length(cell_type_acti))
  {

    tfs_acti_matrix          = cbind(tfs_acti_matrix, cell_type_acti[[zz]])
    cell_ss                  = c(cell_ss, rep(unique_cl[zz], dim(cell_type_acti[[zz]])[2]))
    cellNamess               = c(cellNamess, colnames(cell_type_acti[[zz]]))

  }

  row.names(tfs_acti_matrix) = unique_tfs
  colnames(tfs_acti_matrix)  = cellNamess
  
  return(tfs_acti_matrix)
}

Plot_regulon_activity <- function(regulon_activity_cells, cell_type)
{
  library('pheatmap')

  Norma_acti_data  = Scale_data(regulon_activity_cells) # scale data into (0,1)
  unique_cls       = unique(cell_type)

  sum_activity     = NULL
  for(zz in 1:length(unique_cls))
  {
    sum_activity   = cbind(sum_activity, apply(Norma_acti_data[,which(cell_type==unique_cls[zz])], 1, mean))
  }

  row.names(sum_activity) = unique_tfs
  colnames(sum_activity)  = unique_cls

  my_sample_col            = data.frame(  Label = unique_cls)
  row.names(my_sample_col) = unique_cls

  bk         = c(seq(0,0.5,by=0.01),seq(0.51, 1, by=0.01))
  ann_colors = list( Label = color_PBMC )

  pdf('regulon_activity_cells.pdf')

  pheatmap(sum_activity, cluster_rows = T, cluster_cols =F, scale="row", 
           annotation_col = my_sample_col, annotation_colors = ann_colors,
           show_rownames=T, treeheight_row = F, fontsize_row = 7, clustering_distance_rows = "euclidean",
           fontsize_col = 10, show_colnames = F,angle_col = 90, main="", cex.main=2,
           color=c(colorRampPalette(colors = c("dodgerblue","white"))(length(bk)/2),
                   colorRampPalette(colors = c("white","tomato2"))(length(bk)/2)), breaks=bk )
  dev.off()
}


random_correlation_test_bg <- function(data1, data2, temp_data2, times = 1000)
{
  sample_seq  = sample(1:dim(data2)[1], times)
  cor_temp    = NULL
  temp_da1    = data1

  for(kkk in sample_seq)
  {
    temp_da2  = data2[kkk,]
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

Scale_data <- function(devi_data, ymin = 0, ymax = 1){
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


Similarity_coregulated_gene <- function(Net, species = "mouse")
{
  library(GOSemSim)

  if(species=="mouse")
  {
    hsGO2     = godata('org.Mm.eg.db', keytype = "SYMBOL", ont="BP", computeIC=FALSE)

  }else{

    hsGO2     = godata('org.Hs.eg.db', keytype = "SYMBOL", ont="BP", computeIC=FALSE)
  }
  
  unique_TFs       = unique(Net[[2]])
  Similarity_score = list()
  geneLists        = NULL
  
  for(zz in 1:length(unique_TFs))
  {
    geneLists = Net[[3]][which(Net[[2]]==unique_TFs[zz])]

    if(length(geneLists)>2)
    {
      res <- try(mgeneSim(geneLists, semData=hsGO2, measure="Wang", combine="max", verbose=FALSE), silent = T)
      if(inherits(res, "try-error"))
      {
        Similarity_score[[zz]] = NA
        next
      }

      Eval_wang = mgeneSim(geneLists, semData=hsGO2, measure="Wang", combine="max", verbose=FALSE)
      used_sim  = NULL
      for(i in 2:dim(Eval_wang)[1])
      {
        for(j in 1:(i-1))
        {
          used_sim = c(used_sim, Eval_wang[i,j])
        }
      }

      Similarity_score[[zz]] = used_sim

    }else{

      Similarity_score[[zz]] = NA
    }
  }

  names(Similarity_score) = unique_TFs
  return(Similarity_score)
}


colo_cellLine = c("BJ" = "chartreuse3", "GM" = "coral2", "H1" = "dodgerblue", "K562"= "cyan3")
color_PBMC    = c("CD14 Mono"="#B97E4C", "CD16 Mono"="#A061E2", "B intermediate"="#E5E02F", "B memory"="#5DB8EA", 
                  "B naive"= "#33A02C" ,"CD4 Naive"="#D0021B", "CD4 TCM"="#F5A623", "CD4 TEM"="#517E1E", "CD8 Naive"="#686BEC",
                  "CD8 TEM" = "dodgerblue", "cDC2" = "tomato2", "gdT" = "#BF43D9", "MAIT" = "palegreen", "NK"="cyan2", 
                  "pDC" = "chocolate1", "Treg" ="plum2" )

theme_format = theme_classic(base_size = 15)+
               theme(axis.text.x  = element_text(face ="bold", size = 10, color = "black"),
                     axis.text.y  = element_text(face ="bold", size = 10, color = "black"),
                     axis.title.x = element_text(face ="bold", size = 12, color = "black"),
                     axis.title.y = element_text(face ="bold", size = 12, color = "black"),
                     legend.text  = element_text(face ="bold", size = 10, color = "black"),
                     legend.title = element_text(face ="bold", size = 12, color = "black"),
                     legend.key   = element_blank(),
                     plot.title   = element_text(hjust = 0.5),
                     legend.position = "none")
