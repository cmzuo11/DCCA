# DCCA
A computational method for joint analysis of single-cell multi-omics data by combining VAEs and attention-transfer.

DCCA runs as follows: 
1. The input of DCCA is the raw data of scRNA-seq and scMethylation data (binary data, i.e., scATAC-seq). for example, ‘example.csv’ in the above 
The input data is SNV data, either binary or ternary data. If it is binary data, 0 represents non-mutation site, 1 represents mutation site, 3 represents missing; if ternary data, 0 represents non-mutation site, 1 represents mutation heterozygous site, 2 represents mutation homozygous site and 3 represents missing，for example, ‘example.csv’ in the above list. 
2. Run the matlab script file, named "carryout_RPCA.m" to recover the genotype matrix (please note that the path of the folder RobustClone/matlab_and_R_scripts in the second line of code "addpath(genpath('./RobustClone/matlab_and_R_scripts'));" needs to be completed).
3. Run the R language script file, named "carryout_clonal_tree.R" to cluster cells and reconstruct the subclonal evolutionary tree (please note that the path of the R file RobustClone/matlab_and_R_scripts/Clustering_EvolutionaryTree_function.R in the first and second lines of code, i.e., "source('./RobustClone/matlab_and_R_scripts/Clustering_EvolutionaryTree_function.R')" and "setwd('./RobustClone/example')" need to be completed)
