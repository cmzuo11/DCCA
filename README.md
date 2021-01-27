# DCCA
Deep cross-omics cycle attention (DCCA) model for joint analysis of single-cell multi-omics data.

We proposed DCCA for accurately dissecting the cellular heterogeneity on joint profiling multi-omics data from the same individual cell by transferring representation between each other. 

# Installation

DCCA network is implemented in Pytorch framework. Please run DCCA on CUDA if possible. 

DCCA requires python 3.6.12 or latter, and torch 1.6.0 or latter. 

git clone git://github.com/cmzuo11/DCCA.git

cd DCCA

python setup.py install


# Quick start

## Input: 

> the raw count data of scRNA-seq and scMethylation data (i.e., binary scATAC-seq data). 
> Row indicates variable (genes and loci), and column indicates sample (cell).
> the example files for both omics data are included in the Example_test folder.

## Run: 

> python Main_SNARE_seq.py 

## Output:

the output file will be saved for further analysis:

model_DCCA.pt: saved model for reproducing results.
scRNA-latent.csv: latent features (joint-learning space) for scRNA-seq data for clustering and visulization.
scATAC-latent.csv: latent features for scATAC-seq (or other omics) data for clustering and visulization.
scRNA-norm.csv: normalized data for the scRNA-seq data.
scATAC-norm.csv: imputated or denoised data for the scATAC-seq (other omics) data.

# useful paramters:
modify the initial learning rate paramters for each omics data: i.e., lr1 (the default value of flr1 is lr1/10), lr2 (the default value of flr2 is lr2/10);
use the neural network structure based on your selected variables.
modify the trade-off paramters between the latent feature representing information of each omics data and supervision signal from other omcis data. i.e., sf2 indicates the the weights of signal from scRNA-seq data, args.sf3 indicates the the weights of signal from scATAC-seq data.
