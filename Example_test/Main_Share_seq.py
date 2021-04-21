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
from DCCA.MVAE_cycleVAE import DCCA

def train_with_argas( args ):

  args.workdir  =  '/sibcb2/chenluonanlab7/cmzuo/workPath/sc_dl/Multi-model/Datasets/Real/SHARE-seq'
  args.outdir   =  '/sibcb2/chenluonanlab7/cmzuo/workPath/sc_dl/Multi-model/Datasets/Real/SHARE-seq/DCCA/'
  save_model_p  =  '/sibcb2/chenluonanlab7/cmzuo/workPath/sc_dl/Multi-model/Datasets/Real/SHARE-seq/DCCA/best_model/'

  if not os.path.exists( save_model_p ) :
    os.makedirs( save_model_p )

  args.File1    =  os.path.join( args.workdir, 'Skin_2000_rna_34774cells.txt' )
  args.File2    =  os.path.join( args.workdir, 'Skin_35943_atac_34774cells_binary.txt' ) 
  args.File3    =  os.path.join( args.workdir, 'GSM4156597_skin_cluster.txt' ) 
  model_file    =  os.path.join( save_model_p, 'model_DCCA.pth.tar' )

  args.batch_size     = 256
  args.epoch_per_test = 10
  args.use_cuda       =  args.use_cuda and torch.cuda.is_available()

  adata, adata1, train_index, test_index, label_ground_truth, _ = read_dataset( File1          = args.File1,
                                                                                File2          = args.File2,
                                                                                File3          = args.File3,
                                                                                File4          = None,
                                                                                transpose      = True, 
                                                                                test_size_prop = 0.1,
                                                                                state          = 3 )

  adata  = normalize( adata, filter_min_counts=False, 
                      size_factors=True, normalize_input=False, 
                      logtrans_input=True ) 

  adata1 = normalize( adata1, filter_min_counts=False, 
                      size_factors=False, normalize_input=False, 
                      logtrans_input=False ) 
    
  Nsample1, Nfeature1   =  np.shape( adata.X )
  Nsample2, Nfeature2   =  np.shape( adata1.X )

  train  = data_utils.TensorDataset( torch.from_numpy( adata[train_index].X ),
                                     torch.from_numpy( adata.raw[train_index].X ), 
                                     torch.from_numpy( adata.obs['size_factors'][train_index].values ),
                                     torch.from_numpy( adata1[train_index].X ),
                                     torch.from_numpy( adata1.raw[train_index].X ), 
                                     torch.from_numpy( adata1.obs['size_factors'][train_index].values ))

  train_loader   = data_utils.DataLoader( train, batch_size = args.batch_size, shuffle = True )

  test           = data_utils.TensorDataset( torch.from_numpy( adata[test_index].X ),
                                             torch.from_numpy( adata.raw[test_index].X ), 
                                             torch.from_numpy( adata.obs['size_factors'][test_index].values ),
                                             torch.from_numpy( adata1[test_index].X ),
                                             torch.from_numpy( adata1.raw[test_index].X ), 
                                             torch.from_numpy( adata1.obs['size_factors'][test_index].values ) )

  test_loader    = data_utils.DataLoader( test, batch_size = len(test_index), shuffle = False )

  total          = data_utils.TensorDataset( torch.from_numpy( adata.X ),
                                             torch.from_numpy( adata.raw.X ), 
                                             torch.from_numpy( adata.obs['size_factors'].values ),
                                             torch.from_numpy( adata1.X ),
                                             torch.from_numpy( adata1.raw.X ), 
                                             torch.from_numpy( adata1.obs['size_factors'].values ))
  total_loader   = data_utils.DataLoader( total, batch_size = args.batch_size, shuffle=False, drop_last=False )

  args.lr1       = 0.005 
  args.flr1      = 0.0005
  args.lr2       = 0.01
  args.flr2      = 0.001
  args.cluster1  = 23
  args.cluster2  = 23

  args.sf1       = 10
  args.sf2       = 1

  model =  DCCA( layer_e_1 = [Nfeature1, 1000, 128], hidden1_1 = 128, Zdim_1 = 20, layer_d_1  = [20, 128, 1000],
                 hidden2_1 = 1000,        layer_e_2 = [Nfeature2, 10000, 5000, 1000, 128], hidden1_2 = 128, Zdim_2 = 20,
                 layer_d_2 = [20],        hidden2_2 = 20,     args = args, ground_truth = None,
                 ground_truth1  = None,      Type_1 = 'NB', Type_2 = "Bernoulli", cycle = 3, 
                 attention_loss = "Eucli", patttern = "Both" )

  if args.use_cuda:
    model.cuda()

  NMI_score1, ARI_score1, NMI_score2, ARI_score2  =  model.fit_model(train_loader, test_loader, None, 'RNA' )

  save_checkpoint(model, model_file ) 

  #cluster_rna, cluster_epi = model.predict_cluster_by_kmeans(total_loader)

  #model_new = load_checkpoint( model_file , model, args.use_cuda )
  #latent_z1, latent_z2, norm_x1, _, norm_x2, _ = model_new( total_loader )

  latent_z1, latent_z2, norm_x1, recon_x1, norm_x2, _ = model.encodeBatch( total_loader )

  if latent_z1 is not None:
    imputed_val  = pd.DataFrame( latent_z1, index= adata.obs_names ).to_csv( os.path.join( args.outdir, 
                                 'scRNA-latent_RNA.csv' ) ) 
  if norm_x1 is not None:
    norm_x1_1    = pd.DataFrame( norm_x1, columns =  adata.var_names, 
                                 index= adata.obs_names ).to_csv( os.path.join( args.outdir,
                                 'scRNA-norm_RNA.csv' ) )
  if recon_x1 is not None:
    recon_x1_1   = pd.DataFrame( recon_x1, columns =  adata.var_names, 
                                 index= adata.obs_names ).to_csv( os.path.join( args.outdir,
                                 'scRNA-recon_RNA.csv' ) )
  if latent_z2 is not None:
    imputed_val  = pd.DataFrame( latent_z2, index= adata1.obs_names ).to_csv( os.path.join( args.outdir, 
                                 'scATAC-latent_RNA.csv' ) ) 

  if norm_x2 is not None:
    norm_x1_1    = pd.DataFrame( norm_x2, columns =  adata1.var_names, 
                                 index= adata1.obs_names ).to_csv( os.path.join( args.outdir,
                                 str(file_fla) + '_scATAC-norm_RNA.csv' ) )
   
if __name__ == "__main__":

    parser        =  parameter_setting()
    args          =  parser.parse_args()

    train_with_argas(args)
