# Deep cross-omics cycle attention (DCCA) model for joint analysis of single-cell multi-omics data.

![image](https://github.com/cmzuo11/DCCA/blob/main/Utilities/Figure_1.png)

Overview of DCCA model. (A) Given the raw scRNA-seq data (x_i with M variables) or scEpigenomics data (y_i with N variables) as input, the DCCA model learned a coordinated but separate representation for each omics data (z_x and z_y), by mutually supervising each other based on semantic similarity between embeddings, and then reconstructed back to the original dimension as output through a decoder for each omics data. Note: the same cell order for multi-omics data when using attention-transfer as an additional loss is used to ensure the accuracy of the borrowed knowledge from each other. (B) Each low-dimensional embedding (z_x and z_y) for each omics data learned by DCCA can be used for cell visualization and clustering. (C) Aggregated scEpigenomics data (i.e., scATAC-seq) learned by DCCA was used to characterize the TF motif activity of each cell. (D) Transcription regulation was inferred by using correlation analysis and GLR model on both reconstructed data. A detailed description of how to construct a regulation network is given in Methods.

# Installation

DCCA is implemented in the Pytorch framework. Please run DCCA on CUDA if possible. DCCA requires python 3.6.12 or later, and torch 1.6.0 or later. The used packages (described by "used_package.txt") for DCCA can be automatically installed.

* git clone git://github.com/cmzuo11/DCCA.git

* cd DCCA

* python setup.py install


# Quick start

## Input: 

* the raw count data of scRNA-seq and scEpigenomics data (i.e., binary scATAC-seq data). 

* Row indicates variable (genes and loci), and column indicates sample (cell).

* the example files for both omics data are included in the Example_test folder.

## Run: 

* python Main_SNARE_seq.py 

## Useful paramters:

* modify the initial learning rate paramters for each omics data: i.e., lr1 for one omics (i.e., scRNA-seq, default value is 0.01), lr2 for another omics (i.e., scATAC-seq, default value is 0.005);

* modify the neural network structure based on the number of selected variables;

* modify the trade-off paramters between the latent feature representing information of each omics data and supervision signal from other omcis data. i.e., sf1    indicates the the weight of signal from scRNA-seq to scEpigenomics, args.sf2 indicates the the weight of signal from scEpigenomics to scRNA-seq. the default value for sf1 and sf2 is 5 and 1, respectively. You can adjust them from 1 to 10 by 1.

## Output:

the output file will be saved for further analysis:

* model_DCCA.pth.tar: saved model for reproducing results.

* scRNA-latent.csv: latent features (Zx, joint-learning space) for scRNA-seq data for clustering and visulization.

* scATAC-latent.csv: latent features (Zy) for scEpigenomics data (including scATAC-seq and scMethylation data) for clustering and visulization.

* scRNA-norm.csv: normalized data for the scRNA-seq data.

* scATAC-norm.csv: imputated or denoised data for the scATAC-seq (other omics) data.

## Further analysis:

The detailed functions (at ./DCCA/Processing_data.R) regarding how to anlayze single-cell multi-omics data as follows:

* Select_Loci_by_vargenes: select the genomics loci based on predefined genes;
* Plot_umap_embeddings: plot cell embeddings based on each latent feature for each omics data;
* Calculate_TF_score: calcucate the TF score for each cell based on input scATAC-seq data;
* Infer_network: infer TF-TG relationship based on two-omics data;
* ...

## Tutorial
[Cell line mixture dataset of SNARE-seq technology by DCCA model](https://github.com/cmzuo11/DCCA/wiki/cellMix-dataset-from-SNARE-seq-technology-by-DCCA-model)

# Reference:

Chunman Zuo, Hao Dai, Luonan Chen. Deep cross-omics cycle attention model for joint analysis of single-cell multi-omics data. 2021. (submitted).
