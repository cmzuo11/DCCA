# -*- coding: utf-8 -*-
"""
@author: chunmanzuo
"""

import numpy as np
import pandas as pd
import os
import torch
import torch.utils.data as data_utils

from DCCA.utilities import read_dataset, normalize, parameter_setting, save_checkpoint, load_checkpoint
from DCCA.Cycle_model_missing import DCCA_missing

def train_with_argas( args ):

    args.workdir  =  '/sibcb1/chenluonanlab6/zuochunman/workPath/sc_dl/Multi-model/datasets/real/DNA_methylation/'
    args.outdir   =  '/sibcb2/chenluonanlab7/zuochunman/Software/'
    save_model_p  =  '/sibcb2/chenluonanlab7/zuochunman/Software/best_model/'

    if not os.path.exists( save_model_p ) :
        os.makedirs( save_model_p )

    args.batch_size     = 64
    args.epoch_per_test = 10

    args.use_cuda       =  args.use_cuda and torch.cuda.is_available()
    cycle_used          =  [ 1 ]

    atten_los_list      =  [ "Elu"]
    args.cluster1       =  5
    args.cluster2       =  5

    first_group         = [ "RNA" ]
    penality_list       = [ 'Gaussian']

    adata_r, _, train_idx1, test_idx1, _, _ = read_dataset( File1     = os.path.join( args.workdir,'scRNA_1000_1940_new.txt' ) ,
                                                            transpose = True, test_size_prop = 0.0 )

    adata_r         = normalize( adata_r, filter_min_counts=False, size_factors=True, 
                                 normalize_input=False, logtrans_input=True ) 

    #scRNA
    total           = data_utils.TensorDataset( torch.from_numpy( adata_r.X ),
                                                torch.from_numpy( adata_r.raw.X ), 
                                                torch.from_numpy( adata_r.obs['size_factors'].values ) )
    total_loader_r  = data_utils.DataLoader( total, batch_size = (len(test_idx1)+len(train_idx1)), shuffle = False )
        
    adata, adata1, train_index, test_index, label_ground_truth, _ = read_dataset( File1          = os.path.join( args.workdir,'scRNA_methy_1000_559cells_new.txt' ) ,
                                                                                  File3          = os.path.join( args.workdir,'scRNA_methy_559cells_annotation.txt'),
                                                                                  File2          = os.path.join( args.workdir,'scRNA_methy_1000_599cells_modification_new.txt' ),
                                                                                  transpose      = True, 
                                                                                  test_size_prop = 0.1,
                                                                                  state          = 3 )
    adata           = normalize( adata, 
                                 filter_min_counts = False, 
                                 size_factors      = True, 
                                 normalize_input   = False, 
                                 logtrans_input    = True ) 
    adata1          = normalize( adata1, 
                                 filter_min_counts = False, 
                                 size_factors      = False, 
                                 normalize_input   = False, 
                                 logtrans_input    = False ) 

    train           = data_utils.TensorDataset( torch.from_numpy( adata[train_index].X ),
                                                                  torch.from_numpy( adata.raw[train_index].X ), 
                                                                  torch.from_numpy( adata.obs['size_factors'][train_index].values ),
                                                                  torch.from_numpy( adata1[train_index].X ),
                                                                  torch.from_numpy( adata1.raw[train_index].X ), 
                                                                  torch.from_numpy( adata1.obs['size_factors'][train_index].values ))

    train_loader    = data_utils.DataLoader( train, batch_size = args.batch_size, shuffle = True )

    test            = data_utils.TensorDataset( torch.from_numpy( adata[test_index].X ),
                                                                  torch.from_numpy( adata.raw[test_index].X ), 
                                                                  torch.from_numpy( adata.obs['size_factors'][test_index].values ),
                                                                  torch.from_numpy( adata1[test_index].X ),
                                                                  torch.from_numpy( adata1.raw[test_index].X ), 
                                                                  torch.from_numpy( adata1.obs['size_factors'][test_index].values ) )
    test_loader     = data_utils.DataLoader( test, batch_size = len(test_index), shuffle = False )

    total           = data_utils.TensorDataset( torch.from_numpy( adata.X  ),
                                                torch.from_numpy( adata.raw.X ),
                                                torch.from_numpy( adata.obs['size_factors'].values ),
                                                torch.from_numpy( adata1.X  ),
                                                torch.from_numpy( adata1.raw.X ),
                                                torch.from_numpy( adata1.obs['size_factors'].values ) )
    total_loader    = data_utils.DataLoader( total, batch_size = args.batch_size , shuffle = False )

    
    Nsample1, Nfeature1   =  np.shape( adata.X )
    Nsample2, Nfeature2   =  np.shape( adata1.X )

    args.sf1  = 10
    args.sf2  = 1

    args.lr1  = 0.001
    args.flr1 = 0.0001
    args.lr2  = 0.001
    args.flr2 = 0.0001
 
    model     =  DCCA_missing( layer_e_1      = [Nfeature1, 128],       hidden1_1 = 128, Zdim_1 = 10,       layer_d_1  = [10, 128],
                               hidden2_1      = 128,                    layer_e_2 = [Nfeature2, 128 ],       hidden1_2 = 128, 
                               Zdim_2         = 10,                     layer_d_2 = [10, 128],              hidden2_2  = 128,  
                               args           = args,              ground_truth   = label_ground_truth,  ground_truth1 = label_ground_truth,                   
                               Type_1         = 'NB',                      Type_2 = 'Gaussian',                  cycle = 1, 
                               attention_loss = atten_los_list[0],       patttern = "Both" )

    if args.use_cuda:
        model.cuda()

    NMI_score1, ARI_score1, NMI_score2, ARI_score2  = model.fit_model( train_loader, test_loader, total_loader, label_ground_truth[test_index] )

    recon_ys                     = model.inference_other_from_rna( total_loader_r )
    latent_z1, norm_x1, recon_x1 = model.encoder_batch_single(total_loader_r, type_modal = "RNA")

    latent_z1_sub, latent_z2_sub, norm_x1_sub, recon_x1_sub, norm_x2_sub, _ = model.encodeBatch( total_loader )

    if latent_z1 is not None:
        imputed_val  = pd.DataFrame( latent_z1, index= adata_r.obs_names ).to_csv( os.path.join( args.outdir, 
                                     'scRNA-latent_RNA_1940.csv' ) ) 

    if norm_x1 is not None:
        norm_x1_1    = pd.DataFrame( norm_x1, columns =  adata_r.var_names, 
                                     index= adata_r.obs_names ).to_csv( os.path.join( args.outdir,
                                     'scRNA-norm_RNA_1940.csv' ) )

    if recon_x1 is not None:
        recon_x1_1   = pd.DataFrame( recon_x1, columns =  adata_r.var_names, 
                                     index= adata_r.obs_names ).to_csv( os.path.join( args.outdir,
                                     'scRNA-recon_RNA_1940.csv' ) )

    if latent_z1_sub is not None:
        imputed_val  = pd.DataFrame( latent_z1_sub, index= adata.obs_names ).to_csv( os.path.join( args.outdir, 
                                     'scRNA-latent_RNA_599.csv' ) ) 

    if norm_x1_sub is not None:
        norm_x1_1    = pd.DataFrame( norm_x1_sub, columns =  adata.var_names, 
                                     index= adata.obs_names ).to_csv( os.path.join( args.outdir,
                                     'scRNA-norm_RNA_599.csv' ) )

    if recon_x1_sub is not None:
        recon_x1_1   = pd.DataFrame( recon_x1_sub, columns =  adata.var_names, 
                                     index= adata.obs_names ).to_csv( os.path.join( args.outdir,
                                     'scRNA-recon_RNA_599.csv' ) )

    if latent_z2_sub is not None:
        recon_x1_1   = pd.DataFrame( latent_z2_sub, index= adata1.obs_names ).to_csv( os.path.join( args.outdir,
                                     'scMethy-latent_RNA_599.csv' ) )

    if norm_x2_sub is not None:
        recon_x1_1   = pd.DataFrame( norm_x2_sub, columns =  adata1.var_names, 
                                     index= adata1.obs_names ).to_csv( os.path.join( args.outdir,
                                     'scMethy-recon_RNA_599.csv' ) )

    if recon_ys is not None:
        recon_methy1 = pd.DataFrame( recon_ys, index= adata_r.obs_names, 
                                     columns =  adata1.var_names).to_csv( os.path.join( args.outdir, 
                                     'scMethy_by_RNA-test_1940.csv' ) ) 

if __name__ == "__main__":

    parser        =  parameter_setting()
    args          =  parser.parse_args()

    train_with_argas(args)
