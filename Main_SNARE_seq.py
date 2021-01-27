# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 18:18:54 2020

@author: chunmanzuo
"""

import numpy as np
import pandas as pd
import os
import torch
import torch.utils.data as data_utils

#from MVAE_model import VAE, AE, MVAE_POE
from utilities import read_dataset, normalize, parameter_setting
from MVAE_cycleVAE import DCCA

def train_with_argas( args ):

    args.batch_size     = 64
    args.epoch_per_test = 10
    args.use_cuda       =  args.use_cuda and torch.cuda.is_available()

    args.sf2        =  args.sf3 = 2
    args.cluster1   =  args.cluster2   =  4
    args.lr1        =  0.01
    args.flr1       =  0.001
    args.lr2        =  0.005
    args.flr2       =  0.0005

    args.workdir  =  '/sibcb2/chenluonanlab7/cmzuo/workPath/sc_dl/Multi-model/DCCA-main/'
    args.outdir   =  '/sibcb2/chenluonanlab7/cmzuo/workPath/sc_dl/Multi-model/DCCA-main/DCCA_out/'

    args.File1    =  os.path.join( args.workdir, 'scRNA_seq_SNARE.tsv' ) 
    args.File3    =  os.path.join( args.workdir, 'cell_metadata.txt' ) 
    args.File2_1  =  os.path.join( args.workdir, 'scATAC_seq_SNARE.txt' ) 

    adata, _, adata2, train_index, test_index, label_ground_truth, _ = read_dataset( File1 = args.File1,
                                                                                     File3 = args.File3,
                                                                                     File2 = None,
                                                                                     File4 = args.File2_1,
                                                                                     File5 = None,
                                                                                     transpose = True, 
                                                                                     test_size_prop = 0.1,
                                                                                     state = 1 )
    adata  = normalize( adata, filter_min_counts=True, 
                        size_factors=True, normalize_input=False, 
                        logtrans_input=True ) 
    adata2 = normalize( adata2, filter_min_counts=True, 
                        size_factors=False, normalize_input=False, 
                        logtrans_input=False ) 
    
    Nsample1, Nfeature1   =  np.shape( adata.X )
    Nsample2, Nfeature2   =  np.shape( adata2.X )

    train           = data_utils.TensorDataset( torch.from_numpy( adata[train_index].X ),
                                                torch.from_numpy( adata.raw[train_index].X ), 
                                                torch.from_numpy( adata.obs['size_factors'][train_index].values ),
                                                torch.from_numpy( adata2[train_index].X ),
                                                torch.from_numpy( adata2.raw[train_index].X ), 
                                                torch.from_numpy( adata2.obs['size_factors'][train_index].values ))

    train_loader    = data_utils.DataLoader( train, batch_size = args.batch_size, shuffle = True )

    test            = data_utils.TensorDataset( torch.from_numpy( adata[test_index].X ),
                                                torch.from_numpy( adata.raw[test_index].X ), 
                                                torch.from_numpy( adata.obs['size_factors'][test_index].values ),
                                                torch.from_numpy( adata2[test_index].X ),
                                                torch.from_numpy( adata2.raw[test_index].X ), 
                                                torch.from_numpy( adata2.obs['size_factors'][test_index].values ) )

    test_loader     = data_utils.DataLoader( test, batch_size = len(test_index), shuffle = False )
    
    total           = data_utils.TensorDataset( torch.from_numpy( adata.X  ),
                                                torch.from_numpy( adata.raw.X ),
                                                torch.from_numpy( adata.obs['size_factors'].values ),
                                                torch.from_numpy( adata2.X  ),
                                                torch.from_numpy( adata2.raw.X ),
                                                torch.from_numpy( adata2.obs['size_factors'].values ) )
    total_loader    = data_utils.DataLoader( total, batch_size = (len(train_index)+ len(test_index)) , shuffle = False )

    model =  DCCA( layer_e_1 = [Nfeature1, 128], hidden1_1 = 128, Zdim_1 = 4, layer_d_1 = [4, 128],
                   hidden2_1 = 128, layer_e_2 = [Nfeature2, 3000, 1500, 128], hidden1_2 = 128, Zdim_2 = 4,
                   layer_d_2 = [4], hidden2_2 = 4, args = args, ground_truth = label_ground_truth,
                   ground_truth1 = label_ground_truth, Type_1 = "NB", Type_2 = "Bernoulli", cycle = 1, 
                   attention_loss = "Eucli" )

    if args.use_cuda:
        model.cuda()

    NMI_score1, ARI_score1, NMI_score2, ARI_score2  =  model.fit_model(train_loader, test_loader, total_loader, "RNA" )

    torch.save(model, os.path.join( args.outdir, 'model_DCCA.pt' ) )

    result1, result2 = model( total_loader )
    latent_z1 = result1["latent_z1"].data.cpu().numpy()
    norm_x1   = result1["norm_x"].data.cpu().numpy()
    recon_x1  = result1["recon_x"].data.cpu().numpy()

    latent_z2 = result2["latent_z1"].data.cpu().numpy()
    norm_x2   = result2["norm_x"].data.cpu().numpy()

    if latent_z1 is not None:
        imputed_val  = pd.DataFrame( latent_z1, index= adata.obs_names ).to_csv( os.path.join( args.outdir, 'scRNA-latent.csv' ) ) 

    if norm_x1 is not None:
        norm_x1_1    = pd.DataFrame( norm_x1, columns =  adata.var_names, 
                                     index= adata.obs_names ).to_csv( os.path.join( args.outdir, 'scRNA-norm.csv' ) )

    if latent_z2 is not None:
        imputed_val  = pd.DataFrame( latent_z2, index= adata2.obs_names ).to_csv( os.path.join( args.outdir, 'scATAC-latent.csv' ) ) 

    if norm_x2 is not None:
        norm_x1_1    = pd.DataFrame( norm_x2, columns =  adata2.var_names, 
                                     index= adata2.obs_names ).to_csv( os.path.join( args.outdir, 'scATAC-norm.csv' ) )


if __name__ == "__main__":

    parser        =  parameter_setting()
    args          =  parser.parse_args()

    train_with_argas(args)


