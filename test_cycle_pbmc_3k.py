#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 21:17:57 2020

@author: chunmanzuo
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 18:18:54 2020

@author: chunmanzuo
"""

import numpy as np
import pandas as pd
import os
import time
import torch
import math
import torch.utils.data as data_utils
from torch.autograd import Variable
from torch import optim
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics.cluster import normalized_mutual_info_score

from sklearn.metrics.pairwise import pairwise_distances

import shap
from random import randint, sample

#from MVAE_model import VAE, AE, MVAE_POE
from utilities import read_dataset, normalize, parameter_setting, save_checkpoint, adjust_learning_rate, getFinalResult
from MVAE_cycleVAE import scCycle_VAE_old

def train_with_argas( args ):

    args.workdir  =  '/sibcb2/chenluonanlab7/cmzuo/workPath/sc_dl/Multi-model/Datasets/Real/10X/pbmc_unsorted_3k/filtered_feature_bc_matrix/corrput_data/'
    temp_outdir   =  '/sibcb2/chenluonanlab7/cmzuo/workPath/sc_dl/Multi-model/Datasets/Real/10X/pbmc_unsorted_3k/filtered_feature_bc_matrix/corrput_data/scCycleVAE_AT/'

    file1_group   = [  'scRNA-0.9.tsv', 'scRNA-0.8.tsv', 'scRNA-0.8.tsv', 'scRNA-0.8.tsv', 
                       'scRNA-0.7.tsv', 'scRNA-0.7.tsv', 'scRNA-0.7.tsv' ]
    file2_group   = [   'scATAC-0.7.tsv', 'scATAC-0.9.tsv', 'scATAC-0.8.tsv', 'scATAC-0.7.tsv', 
                       'scATAC-0.9.tsv', 'scATAC-0.8.tsv', 'scATAC-0.7.tsv' ]
    file3_group   = [  'pbmc_3k_cell_cluster_2874.txt', 'pbmc_3k_cell_cluster_2874.txt', 'pbmc_3k_cell_cluster_2874.txt', 
                       'pbmc_3k_cell_cluster_2874.txt', 'pbmc_3k_cell_cluster_2874.txt', 'pbmc_3k_cell_cluster_2874.txt', 
                       'pbmc_3k_cell_cluster_2874.txt', 'pbmc_3k_cell_cluster_2874.txt', 'pbmc_3k_cell_cluster_2874.txt' ]
    file5_group   = [  'pbmc_3k_cell_cluster_2874.txt', 'pbmc_3k_cell_cluster_2874.txt', 'pbmc_3k_cell_cluster_2874.txt', 
                       'pbmc_3k_cell_cluster_2874.txt', 'pbmc_3k_cell_cluster_2874.txt', 'pbmc_3k_cell_cluster_2874.txt', 
                       'pbmc_3k_cell_cluster_2874.txt', 'pbmc_3k_cell_cluster_2874.txt', 'pbmc_3k_cell_cluster_2874.txt' ]
    File2_1_group = [  'scATAC-0.7_binary.tsv', 
                       'scATAC-0.9_binary.tsv', 'scATAC-0.8_binary.tsv', 'scATAC-0.7_binary.tsv', 
                       'scATAC-0.9_binary.tsv', 'scATAC-0.8_binary.tsv', 'scATAC-0.7_binary.tsv' ]

    cor_groups    = [ '0.9_0.7', '0.8_0.9', '0.8_0.8', '0.8_0.7', '0.7_0.9', '0.7_0.8', '0.7_0.7' ]

    args.batch_size     = 64
    args.epoch_per_test = 10
    use_cuda            =  args.use_cuda and torch.cuda.is_available()

    args.use_cuda   =  args.use_cuda and torch.cuda.is_available()
    cycle_used      =  [ 3 ]

    atten_los_list  =  [ "AT" ]
    args.cluster1   =  4
    args.cluster2   =  4

    first_group     = [ "RNA" ]
    penality_list   = [ 'Gaussian']

    for zz in range( len(file1_group) ): 

        args.File1    =  os.path.join( args.workdir, file1_group[zz] ) 
        args.File2    =  os.path.join( args.workdir, file2_group[zz] ) 
        args.File3    =  os.path.join( args.workdir, file3_group[zz] ) 
        args.File2_1  =  os.path.join( args.workdir, File2_1_group[zz] ) 
        args.File5    =  os.path.join( args.workdir, file5_group[zz] )

        args.outdir   = os.path.join( temp_outdir, cor_groups[zz] )

        if not os.path.exists( args.outdir ) :
            os.makedirs( args.outdir )
        
        ### read two datasets together
        adata, _, adata2, train_index, test_index, label_ground_truth, label_ground_truth1 = read_dataset( File1 = args.File1,
                                                                                                           File3 = None,
                                                                                                           File2 = args.File2,
                                                                                                           File4 = args.File2_1,
                                                                                                           File5 = None,
                                                                                                           transpose = True, 
                                                                                                           test_size_prop = 0.1,
                                                                                                           state = 1 )

        adata  = normalize( adata, filter_min_counts=False, 
                            size_factors=True, normalize_input=False, 
                            logtrans_input=True ) 

        adata2 = normalize( adata2, filter_min_counts=False, 
                            size_factors=False, normalize_input=False, 
                            logtrans_input=False ) 
    
        Nsample1, Nfeature1   =  np.shape( adata.X )
        Nsample2, Nfeature2   =  np.shape( adata2.X )

        train       = data_utils.TensorDataset( torch.from_numpy( adata[train_index].X ),
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

        layer_e_1   =  [  [Nfeature1, 128] ]
        hidden1_1   =  [  128 ]
        Zdim_1      =  [  10 ]
        layer_d_1   =  [ [10, 128] ]
        hidden2_1   =  [  128 ]

        Type_1      =  [ 'NB' ]

        l_rate_1        =  [  0.005 ]
        final_rate_1    =  [  0.0005 ]
        drop_rate_1     =  [ 0.0 ]
        # for other omics data
        layer_e_2   =  [  [Nfeature2, 5000, 2500, 1000, 128 ] ]
        hidden1_2   =  [  128 ]
        Zdim_2      =  [  10 ]
        layer_d_2   =  [  [10] ]
        hidden2_2   =  [  10 ]
        Type_2      =  [ 'Bernoulli' ]

        l_rate_2        =  [ 0.005, 0.01,  ]
        final_rate_2    =  [ 0.0005, 0.001 ]
        drop_rate_2     =  [ 0.0 ]
        Pattern_group   =  ['Both' ]


        scale_factor_1  =  [ 1 ]
        scale_factor_3  =  [ 5 ]
        scale_factor_2  =  [ 1 ]
    
        test_like_max_list   = []
        reco_epoch_test_list = []

        file_fla_save   = []
        layer_e1_s, hidden1_1_s, Zdim_1_s  = [], [], []
        layer_d_1_s, hidden2_1_s, Type_1_s = [], [], []

        layer_e2_s, hidden1_2_s, Zdim_2_s  = [], [], []
        layer_d_2_s, hidden2_2_s, Type_2_s = [], [], []

        lr_save, drop_save, final_rate_save = [], [], []
        scale_factor_save, cycle_save, atten_los_save = [], [], []
        scale_atten_save, attenti_type_save = [], []
        lr_save2, final_rate_save2          = [], []
   
        NMI_1_list  = []
        ARI_1_list  = []
        NMI_2_list  = []
        ARI_2_list  = []

        ARI_score1, NMI_score1 = -100, -100
        ARI_score2, NMI_score2 = -100, -100

        file_fla   = 0
        first_save = []

        for gg in range( len(Pattern_group) ):

            for cc in range( len(cycle_used) ):

                for ss in range( len(Type_1) ):

                    for att in range( len(atten_los_list) ):

                        for sc in range( len(scale_factor_3) ):

                            for en in range( len(layer_e_1) ):

                                for lr1 in range( len(l_rate_1) ):

                                    for lr2 in range( len(l_rate_2) ):

                                        args.lr1   = l_rate_1[lr1]
                                        args.flr1  = final_rate_1[lr1]
                                        args.lr2   = l_rate_2[lr2]
                                        args.flr2  = final_rate_2[lr2]

                                        file_fla = file_fla + 1
                                        print( str(file_fla) + "   "+ str(cycle_used[cc]) + "   "+ str(args.lr1) + "  "+ str(args.lr2) + "   "+ 
                                               str(scale_factor_3[sc]) + "   "+ str(atten_los_list[att]) + "  "+ str(Zdim_1[en])  )

                                        file_fla_save.append(file_fla)
                                        args.sf1  = scale_factor_1[0]
                                        args.sf2  = scale_factor_2[ss]
                                        args.sf3  = scale_factor_3[sc]

                                        first_save.append( Pattern_group[gg] )
                                        layer_e1_s.append( '-'.join( map(str, layer_e_1[en] )) )
                                        hidden1_1_s.append( hidden1_1[en] )
                                        Zdim_1_s.append( Zdim_1[en] )
                                        layer_d_1_s.append( '-'.join( map(str, layer_d_1[en] )) )
                                        hidden2_1_s.append( hidden2_1[en] )
                                        Type_1_s.append( Type_1[ss] )

                                        layer_e2_s.append( '-'.join( map(str, layer_e_2[en] )) )
                                        hidden1_2_s.append( hidden1_2[en] )

                                        Zdim_2_s.append( Zdim_2[en] )
                                        layer_d_2_s.append( '-'.join( map(str, layer_d_2[en] )) )
                                        hidden2_2_s.append( hidden2_2[en] )
                                        Type_2_s.append( Type_2[0] )

                                        cycle_save.append( cycle_used[cc] )
                                        lr_save.append( l_rate_1[ lr1 ] )
                                        final_rate_save.append( final_rate_1[ lr1 ] )

                                        lr_save2.append( l_rate_2[ lr2 ] )
                                        final_rate_save2.append( final_rate_2[ lr2 ] )

                                        drop_save.append( drop_rate_1[0] )
                                        scale_factor_save.append( scale_factor_1[0] )
                                        scale_atten_save.append(  scale_factor_3[sc] )
                                        attenti_type_save.append( atten_los_list[att] )

                                        model =  scCycle_VAE_old( layer_e_1 = layer_e_1[en], hidden1_1 = hidden1_1[en], Zdim_1    = Zdim_1[en], layer_d_1 = layer_d_1[en],
                                                                  hidden2_1 = hidden2_1[en], layer_e_2 = layer_e_2[en], hidden1_2 = hidden1_2[en], Zdim_2 = Zdim_2[en],
                                                                  layer_d_2 = layer_d_2[en], hidden2_2 = hidden2_2[en], args = args, ground_truth = None,
                                                                  ground_truth1 = None, Type_1 = Type_1[ss], Type_2 = "Bernoulli", cycle = cycle_used[cc], 
                                                                  attention_loss = atten_los_list[att], patttern = "Both" )

                                        if args.use_cuda:
                                            model.cuda()

                                        NMI_score1, ARI_score1, NMI_score2, ARI_score2  =  model.fit_model(train_loader, test_loader, total_loader, 'RNA' )

                                        NMI_1_list.append( NMI_score1 )
                                        ARI_1_list.append( ARI_score1 )
                                        NMI_2_list.append( NMI_score2 )
                                        ARI_2_list.append( ARI_score2 )

                                        result1, result2 = model( total_loader )
                                        latent_z1 = result1["latent_z1"].data.cpu().numpy()
                                        norm_x1   = result1["norm_x"].data.cpu().numpy()
                                        recon_x1  = result1["recon_x"].data.cpu().numpy()

                                        latent_z2 = result2["latent_z1"].data.cpu().numpy()
                                        norm_x2   = result2["norm_x"].data.cpu().numpy()
                                        recon_x2  = result2["recon_x"].data.cpu().numpy()

                                        if latent_z1 is not None:
                                            imputed_val  = pd.DataFrame( latent_z1, index= adata.obs_names ).to_csv( os.path.join( args.outdir, 
                                                                         str(file_fla) + '_scRNA-latent_RNA.csv' ) ) 

                                        if norm_x1 is not None:
                                            norm_x1_1    = pd.DataFrame( norm_x1, columns =  adata.var_names, 
                                                                         index= adata.obs_names ).to_csv( os.path.join( args.outdir,
                                                                         str(file_fla) + '_scRNA-norm_RNA.csv' ) )

                                        if recon_x1 is not None:
                                            recon_x1_1   = pd.DataFrame( recon_x1, columns =  adata.var_names, 
                                                                         index= adata.obs_names ).to_csv( os.path.join( args.outdir,
                                                                         str(file_fla) + '_scRNA-recon_RNA.csv' ) )

                                        if latent_z2 is not None:
                                            imputed_val  = pd.DataFrame( latent_z2, index= adata2.obs_names ).to_csv( os.path.join( args.outdir, 
                                                                         str(file_fla) + '_scATAC-latent_RNA.csv' ) ) 

                                        if norm_x2 is not None:
                                            norm_x1_1    = pd.DataFrame( norm_x2, columns =  adata2.var_names, 
                                                                         index= adata2.obs_names ).to_csv( os.path.join( args.outdir,
                                                                         str(file_fla) + '_scATAC-norm_RNA.csv' ) )

                                        #if recon_x2 is not None:
                                        #    recon_x1_1   = pd.DataFrame( recon_x2, columns =  adata2.var_names, 
                                        #                                 index= adata2.obs_names ).to_csv( os.path.join( args.outdir,
                                        #                                 str(file_fla) + '_scATAC-recon_RNA.csv' ) )
   
        data_three_save  = { "file_labbel": file_fla_save, "layer_e_1": layer_e1_s, "hidden1_1": hidden1_1_s, "Zdim_1": Zdim_1_s, "layer_d_1": layer_d_1_s, 
                             "hidden2_1": hidden2_1_s, "Type_1": Type_1_s, "layer_e_2": layer_e2_s, "hidden1_2": hidden1_2_s, "Zdim_2": Zdim_2_s , 
                             "layer_d_2": layer_d_2_s, "hidden2_2": hidden2_2_s, "Type_2": Type_2_s, "cycle_used": cycle_save, "l_rate_1": lr_save,
                             "final_rate_1": final_rate_save, "l_rate_2": lr_save2,"final_rate_2": final_rate_save2, "drop_rate_1": drop_save, 
                             "scale_factor_1": scale_factor_save, "attention_type": attenti_type_save, "scale_attention": scale_atten_save, 
                             "ARI_1": ARI_1_list, "NMI_1": NMI_1_list, "ARI_2": ARI_2_list, "NMI_2": NMI_2_list, "Pattern_group": first_save, 
                           }
   
        data_three_save1 = pd.DataFrame(data_three_save).to_csv(os.path.join( args.outdir, 'Simulated_data_cycleAE_RNA.csv' ))


if __name__ == "__main__":

    parser        =  parameter_setting()
    args          =  parser.parse_args()

    train_with_argas(args)
