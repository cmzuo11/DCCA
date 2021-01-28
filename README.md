# Deep cross-omics cycle attention (DCCA) model for joint analysis of single-cell multi-omics data.

We proposed DCCA for accurately dissecting the cellular heterogeneity on joint profiling multi-omics data from the same individual cell by transferring representation between each other. 

# Installation

DCCA is implemented in Pytorch framework. Please run DCCA on CUDA if possible. DCCA requires python 3.6.12 or latter, and torch 1.6.0 or latter. 

* git clone git://github.com/cmzuo11/DCCA.git

* cd DCCA

* python setup.py install


# Quick start

## Input: 

* the raw count data of scRNA-seq and scMethylation data (i.e., binary scATAC-seq data). 

* Row indicates variable (genes and loci), and column indicates sample (cell).

* the example files for both omics data are included in the Example_test folder.

## Run: 

* python Main_SNARE_seq.py 

## Useful paramters:

* modify the initial learning rate paramters for each omics data: i.e., lr1 for one omics (i.e., scRNA-seq, default value is 0.01), lr2 for another omics (i.e., scATAC-seq, default value is 0.002);

* modify the neural network structure based on your selected variables;

* modify the trade-off paramters between the latent feature representing information of each omics data and supervision signal from other omcis data. i.e., sf1    indicates the the weights of signal from scRNA-seq data, args.sf2 indicates the the weights of signal from scATAC-seq data. the default value for two parameter is 2. you can adjust them from 1 to 10 by 1.

## Output:

the output file will be saved for further analysis:

* model_DCCA.pt: saved model for reproducing results.

* scRNA-latent.csv: latent features (joint-learning space) for scRNA-seq data for clustering and visulization.

* scATAC-latent.csv: latent features for scATAC-seq (or other omics) data for clustering and visulization.

* scRNA-norm.csv: normalized data for the scRNA-seq data.

* scATAC-norm.csv: imputated or denoised data for the scATAC-seq (other omics) data.

## Further analysis:

The detailed function (at ./DCCA/Processing_data.R) for how to anlayze single-cell multi-omics data as follows:

* Select_Loci_by_vargenes: select the genomics loci based on predefined genes;
* Plot_umap_embeddings: plot cell embeddings based on each latent feature for each omics data;
* Calculate_TF_score: calcucate the TF score for each cell based on your input data;
* ...

# Reference:

Chunman Zuo, Hao Dai, Luonan Chen. Deep cross-omics cycle attention model for joint analysis of single-cell multi-omics data. 2021. (submitted).
