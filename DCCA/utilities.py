# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 19:30:50 2020

@author: chunmanzuo
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import scanpy as sc
import os
import time
import argparse
import scipy as sp
import torch
from torch.autograd import Variable
import torch.utils.data as data_utils
from torch import optim

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import mixture
from scipy.io import mmread
import operator

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, scale
from sklearn.model_selection import train_test_split

def parameter_setting():
    
    parser = argparse.ArgumentParser(description='Single cell Multi-omics data analysis')
    
    outPath = '/sibcb1/chenluonanlab6/zuochunman/workPath/Multimodal/MVAE/Datasets/Simulated/Simulated_2/new_test/'
    
    parser.add_argument('--File1', '-F1', type=str, default = 'scRNA_seq_SNARE.tsv',    help='input file name1')
    parser.add_argument('--File2', '-F2', type=str, default = 'scATAC_seq_SNARE.txt', help='input file name2')
    parser.add_argument('--File2_1', '-F2_1', type=str, default = 'scATAC_seq_SNARE.txt', help='input file name2_1')

    parser.add_argument('--File3', '-F3', type=str, default = '5-cellinfo-RNA.tsv',  help='input meta file')
    parser.add_argument('--File_combine', '-F_com', type=str, default = 'Gene_chromatin_order_combine.tsv',    help='input combine file name')
    
    parser.add_argument('--workdir', '-wk', type=str, default = outPath, help='work path')
    parser.add_argument('--outdir', '-od', type=str, default = outPath, help='Output path')
    
    parser.add_argument('--lr1', type=float, default = 0.01, help='Learning rate1')
    parser.add_argument('--flr1', type=float, default = 0.001, help='Final learning rate1')
    parser.add_argument('--lr2', type=float, default = 0.002, help='Learning rate2')
    parser.add_argument('--flr2', type=float, default = 0.0002, help='Final learning rate2')
    parser.add_argument('--weight_decay', type=float, default = 1e-6, help='weight decay')
    parser.add_argument('--eps', type=float, default = 0.01, help='eps')

    parser.add_argument('--sf1', type=float, default = 2.0, help='scale_factor_1 for supervision signal from scRNA-seq')
    parser.add_argument('--sf2', type=float, default = 2.0, help='scale_factor_2 for supervision signal from scEpigenomics')
    parser.add_argument('--cluster1', '-clu1', type=int, default=2, help='predefined cluster for scRNA')
    parser.add_argument('--cluster2', '-clu2', type=int, default=2, help='predefined cluster for other epigenomics')
    parser.add_argument('--geneClu', '-gClu', type=list, default = None, help='predefined gene cluster for scRNA')
    
    parser.add_argument('--batch_size', '-b', type=int, default=64, help='Batch size')
    parser.add_argument('--use_cuda', dest='use_cuda', default=True, action='store_true', help=" whether use cuda(default: True)")
    
    parser.add_argument('--seed', type=int, default=200, help='Random seed for repeat results')
    parser.add_argument('--latent', '-l',type=int, default=10, help='latent layer dim')
    parser.add_argument('--max_epoch', '-me', type=int, default=500, help='Max epoches')
    parser.add_argument('--max_iteration', '-mi', type=int, default=3000, help='Max iteration')
    parser.add_argument('--anneal_epoch', '-ae', type=int, default=200, help='Anneal epoch')
    parser.add_argument('--epoch_per_test', '-ept', type=int, default=10, help='Epoch per test')
    parser.add_argument('--max_ARI', '-ma', type=int, default=-200, help='initial ARI')
    
    return parser

def read_dataset( File1 = None, File2 = None, File3 = None, File4 = None, transpose = True, test_size_prop = 0.15, state = 0, 
                  format_rna="table", formar_epi = "table" ):
    # read single-cell multi-omics data together

    ### raw reads count of scRNA-seq data
    adata = adata1 = None

    if File1 is not None:
        if format_rna == "table":
            adata  = sc.read(File1)
        else: # 10X format
            adata  = sc.read_mtx(File1)
           
        if transpose:
            adata  = adata.transpose()

    ##$ the binarization data for scEpigenomics file
    if File2 is not None:
        if formar_epi == "table":
            adata1  = sc.read( File2 )
        else:# 10X format
            adata1  = sc.read_mtx(File2)

        if transpose: 
            adata1  = adata1.transpose()
    
    ### File3 and File4 for cell group information of scRNA-seq and scEpigenomics data
    label_ground_truth = []
    label_ground_truth1 = []

    if state == 0 :
        if File3 is not None:
            Data2 = pd.read_csv( File3, header=0, index_col=0 )
            group = Data2['Group'].values
            for g in group:
                g = int(g.split('Group')[1])
                label_ground_truth.append(g)
        else:
            label_ground_truth =  np.ones( len( adata.obs_names ) )

        if File4 is not None:
            Data2 = pd.read_csv( File4, header=0, index_col=0 )
            group = Data2['Group'].values
            for g in group:
                g = int(g.split('Group')[1])
                label_ground_truth1.append(g)
        else:
            label_ground_truth1 =  np.ones( len( adata.obs_names ) )

    elif state == 1:
        if File3 is not None:
            Data2 = pd.read_table( File3, header=0, index_col=0 )
            label_ground_truth = Data2['cell_line'].values
        else:
            label_ground_truth =  np.ones( len( adata.obs_names ) )

        if File4 is not None:
            Data2 = pd.read_table( File4, header=0, index_col=0 )
            label_ground_truth1 = Data2['cell_line'].values
        else:
            label_ground_truth1 =  np.ones( len( adata.obs_names ) )
    else:
        if File3 is not None:
            Data2 = pd.read_table( File3, header=0, index_col=0 )
            label_ground_truth = Data2['Cluster'].values
        else:
            label_ground_truth =  np.ones( len( adata.obs_names ) )

        if File4 is not None:
            Data2 = pd.read_table( File4, header=0, index_col=0 )
            label_ground_truth1 = Data2['Cluster'].values
        else:
            label_ground_truth1 =  np.ones( len( adata.obs_names ) )

    #split datasets into training and testing sets
    if test_size_prop > 0 :
        train_idx, test_idx = train_test_split(np.arange(adata.n_obs), 
                                               test_size = test_size_prop, 
                                               random_state = 200)
        spl = pd.Series(['train'] * adata.n_obs)
        spl.iloc[test_idx]  = 'test'
        adata.obs['split']  = spl.values
        
        if File2 is not None:
            adata1.obs['split'] = spl.values
    else:
        train_idx, test_idx = list(range( adata.n_obs )), list(range( adata.n_obs ))
        spl = pd.Series(['train'] * adata.n_obs)
        adata.obs['split']       = spl.values

        if File2 is not None:
            adata1.obs['split']  = spl.values
        
    adata.obs['split'] = adata.obs['split'].astype('category')
    adata.obs['Group'] = label_ground_truth
    adata.obs['Group'] = adata.obs['Group'].astype('category')
    
    if File2 is not None:
        adata1.obs['split'] = adata1.obs['split'].astype('category')
        adata1.obs['Group'] = label_ground_truth
        adata1.obs['Group'] = adata1.obs['Group'].astype('category')

    print('Successfully preprocessed {} genes and {} cells.'.format(adata.n_vars, adata.n_obs))
    
    ### here, adata with cells * features
    return adata, adata1, train_idx, test_idx, label_ground_truth, label_ground_truth1

def normalize( adata, filter_min_counts=True, size_factors=True, 
               normalize_input=False, logtrans_input=True):

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)

    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if logtrans_input:
        sc.pp.log1p(adata)

    if size_factors:
        #adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
        adata.obs['size_factors'] = np.log( np.sum( adata.X, axis = 1 ) )
    else:
        adata.obs['size_factors'] = 1.0

    if normalize_input:
        sc.pp.scale(adata)

    return adata

def calculate_log_library_size( Dataset ):
    ### Dataset is raw read counts, and should be cells * features
    Nsamples     =  np.shape(Dataset)[0]
    library_sum  =  np.log( np.sum( Dataset, axis = 1 ) )
    
    lib_mean     =  np.full( (Nsamples, 1), np.mean(library_sum) )
    lib_var      =  np.full( (Nsamples, 1), np.var(library_sum) ) 
    
    return lib_mean, lib_var

def adjust_learning_rate(init_lr, optimizer, iteration, max_lr, adjust_epoch):

    lr = max(init_lr * (0.9 ** (iteration//adjust_epoch)), max_lr)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr   

def save_checkpoint(model, filename='model_best.pth.tar', folder='./saved_model/'):

    if not os.path.isdir(folder):
        os.mkdir(folder)

    torch.save(model.state_dict(), os.path.join(folder, filename))


def load_checkpoint(file_path, model, use_cuda=False):

    if use_cuda:
        device = torch.device( "cuda" )
        model.load_state_dict( torch.load(file_path) )
        model.to(device)
        
    else:
        device = torch.device('cpu')
        model.load_state_dict( torch.load(file_path, map_location=device) )
    model.eval()

    return model

def Normalized_0_1 (Data):
    ## here ,Data is cell * genes
    adata = sc.AnnData( Data )
    sc.pp.normalize_per_cell( adata, counts_per_cell_after=1,
                              key_n_counts='n_counts2' ) 
    return adata

def estimate_cluster_numbers(data):
    """
    Estimate number of groups k:
        based on random matrix theory (RTM), borrowed from SC3
        input data is (n,p) matrix, n is feature, p is sample
    """
    n, p = data.shape
    if type(data) is not np.ndarray:
        data = data.toarray()

    x             = scale(data) # normalization for each sample
    muTW          = (np.sqrt(n-1) + np.sqrt(p)) ** 2
    sigmaTW       = (np.sqrt(n-1) + np.sqrt(p)) * (1/np.sqrt(n-1) + 1/np.sqrt(p)) ** (1/3)
    sigmaHatNaive = x.T.dot(x)

    bd    = np.sqrt(p) * sigmaTW + muTW
    evals = np.linalg.eigvalsh(sigmaHatNaive)

    k = 0
    for i in range(len(evals)):
        if evals[i] > bd:
            k += 1
    return k
