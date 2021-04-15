# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 17:00:41 2020
@author: chunmanzuo
"""

import numpy as np
import pandas as pd
import os
import time
import math
import torch.utils.data as data_utils

from torch.autograd import Variable
from torch import optim
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal, kl_divergence as kl

from DCCA.layers import Encoder, Decoder_logNorm_ZINB, Decoder_logNorm_NB, Decoder
from DCCA.loss_function import log_zinb_positive, log_nb_positive, binary_cross_entropy, mse_loss, KL_diver
from DCCA.loss_function import NSTLoss, FactorTransfer, Similarity, Correlation, Attention, Eucli_dis, L1_dis
from DCCA.utilities import adjust_learning_rate

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class VAE(nn.Module):
    #def __init__( self, layer_e, hidden1, hidden2, layer_l, layer_d, hidden ):
    def __init__( self, layer_e, hidden1, Zdim, layer_d, hidden2, 
                  n_centroids, Type = 'NB', penality = 'Gaussian', 
                  droprate = 0.1 ):

        super(VAE, self).__init__()
        
        ###  encoder
        self.encoder     = Encoder( layer_e, hidden1, Zdim, droprate = droprate )
        self.activation  = nn.Softmax(dim=-1)

        ### the decoder
        if Type == 'ZINB':
            self.decoder = Decoder_logNorm_ZINB( layer_d, hidden2, layer_e[0], droprate = droprate )

        elif Type == 'NB':
            self.decoder = Decoder_logNorm_NB( layer_d, hidden2, layer_e[0], droprate = droprate )

        else: ## Bernoulli, or Gaussian
            self.decoder = Decoder( layer_d, hidden2, layer_e[0], Type, droprate = droprate)

        ### parameters
        self.Type            = Type
        self.penality        = penality
    
    def inference(self, X = None, scale_factor = 1.0):
        # encoder
        mean_1, logvar_1, latent_1, hidden = self.encoder.return_all_params( X )
        
        ### decoder
        if self.Type == 'ZINB' :
            output        =  self.decoder( latent_1, scale_factor )
            norm_x        =  output["normalized"]
            disper_x      =  output["disperation"]
            recon_x       =  output["scale_x"]
            dropout_rate  =  output["dropoutrate"]

        elif self.Type == 'NB' :
            output        =  self.decoder( latent_1, scale_factor )
            norm_x        =  output["normalized"]
            disper_x      =  output["disperation"]
            recon_x       =  output["scale_x"]
            dropout_rate  =  None

        else:
            recons_x      =  self.decoder( latent_1 )
            recon_x       =  recons_x
            norm_x        =  recons_x
            disper_x      =  None
            dropout_rate  =  None

        return dict( norm_x   = norm_x,  disper_x   = disper_x,  dropout_rate  = dropout_rate,
                     recon_x  = recon_x, latent_z1  = latent_1,  mean_1        = mean_1,  
                     logvar_1 = logvar_1,  hidden   = hidden
                    )

    def return_loss(self, X      = None, X_raw       = None, latent_pre        = None, 
                    mean_pre     = None, logvar_pre  = None, latent_pre_hidden = None, 
                    scale_factor = 1.0, cretion_loss = None, attention_loss    = None ):

        output       = self.inference( X, scale_factor )
        recon_x      = output["recon_x"]
        disper_x     = output["disper_x"]
        dropout_rate = output["dropout_rate"]

        mean_1       = output["mean_1"]
        logvar_1     = output["logvar_1"]
        latent_z1    = output["latent_z1"]

        hidden       = output["hidden"]
      
        if self.Type == 'ZINB':
            loss = log_zinb_positive( X_raw, recon_x, disper_x, dropout_rate )

        elif self.Type == 'NB':
            loss = log_nb_positive( X_raw, recon_x, disper_x )
            
        elif self.Type == 'Bernoulli': # here X and X_raw are same
            loss = binary_cross_entropy( recon_x, X_raw )
            
        else:
            loss = mse_loss( X, recon_x )

        ##calculate KL loss for Gaussian distribution
        mean             =  torch.zeros_like(mean_1)
        scale            =  torch.ones_like(logvar_1)
        kl_divergence_z  =  kl( Normal(mean_1, logvar_1), 
                                Normal(mean, scale)).sum(dim=1)

        atten_loss1 = torch.tensor(0.0)
        if latent_pre is not None and latent_pre_hidden is not None:

            if attention_loss == "KL_div":
                atten_loss1 = cretion_loss( mean_1, logvar_1, mean_pre, logvar_pre )

            else:
                atten_loss1 = cretion_loss(latent_z1, latent_pre)

        return loss, kl_divergence_z, atten_loss1
        
    def forward( self, X = None, scale_factor = 1.0 ):

        output =  self.inference( X, scale_factor )

        return output

    def fit( self, train_loader, test_loader, total_loader, model_pre, args, criterion, cycle, state, first = "RNA", attention_loss = "Eucli" ):

        params  = filter(lambda p: p.requires_grad, self.parameters())

        if cycle%2 == 0:
            optimizer       = optim.Adam( params, lr = args.lr1, weight_decay = args.weight_decay, eps = args.eps )
        else:
            optimizer       = optim.Adam( params, lr = args.lr2, weight_decay = args.weight_decay, eps = args.eps )

        train_loss_list   = []
        reco_epoch_test   = 0
        test_like_max     = 100000
        flag_break        = 0

        patience_epoch    = 0
        args.anneal_epoch = 10

        model_pre.eval()

        start = time.time()

        for epoch in range( 1, args.max_epoch + 1 ):

            self.train()

            patience_epoch += 1
            kl_weight      =  min( 1, epoch / args.anneal_epoch )

            if cycle%2 == 0:
                epoch_lr       =  adjust_learning_rate( args.lr1, optimizer, epoch, args.flr1, 10 )
            else:
                epoch_lr       =  adjust_learning_rate( args.lr2, optimizer, epoch, args.flr2, 10 )

            for batch_idx, ( X1, X1_raw, size_factor1, X2, X2_raw, size_factor2 ) in enumerate(train_loader):

                if args.use_cuda:
                    X1, X1_raw, size_factor1 = X1.to("cuda:0"), X1_raw.to("cuda:0"), size_factor1.to("cuda:0")
                    X2, X2_raw, size_factor2 = X2.to("cuda:1"), X2_raw.to("cuda:1"), size_factor2.to("cuda:1")
                
                X1, X1_raw, size_factor1 = Variable( X1 ), Variable( X1_raw ), Variable( size_factor1 )
                X2, X2_raw, size_factor2 = Variable( X2 ), Variable( X2_raw ), Variable( size_factor2 )

                optimizer.zero_grad()

                if first == "RNA" :

                    if cycle%2 == 0:

                        if state == 0 :
                            # initialization of scRNA-seq model
                            loss1, kl_divergence_z, atten_loss1 = self.return_loss( X1, X1_raw, None, None, None, None, 
                                                                                    size_factor1, criterion, attention_loss)
                            loss = torch.mean( loss1  + (kl_weight * kl_divergence_z)  )

                        else:
                            #transfer representation from scEpigenomics model to scRNA-seq model
                            result_2  = model_pre( X2, size_factor2)
                            latent_z1 = result_2["latent_z1"].to("cuda:0")
                            hidden_1  = result_2["hidden"].to("cuda:0")
                            mean_1    = result_2["mean_1"].to("cuda:0")
                            logvar_1  = result_2["logvar_1"].to("cuda:0")

                            loss1, kl_divergence_z, atten_loss1 = self.return_loss( X1, X1_raw, latent_z1, mean_1, logvar_1, hidden_1,
                                                                                    size_factor1, criterion, attention_loss)
                            loss = torch.mean( loss1 +  (kl_weight * kl_divergence_z) + (args.sf2 * (atten_loss1) ) ) 
                    
                    else:
                        if state == 0 :
                            # initialization of scEpigenomics model
                            loss1, kl_divergence_z, atten_loss1 = self.return_loss( X2, X2_raw, None, None,  None, None, 
                                                                                    size_factor2, criterion, attention_loss)
                            loss = torch.mean( loss1  + (kl_weight * kl_divergence_z)  ) 

                        else:
                            #transfer representation form scRNA-seq model to scEpigenomics model
                            result_2  = model_pre( X1, size_factor1)
                            latent_z1 = result_2["latent_z1"].to("cuda:1")
                            hidden_1  = result_2["hidden"].to("cuda:1")
                            mean_1    = result_2["mean_1"].to("cuda:1")
                            logvar_1  = result_2["logvar_1"].to("cuda:1")

                            loss1, kl_divergence_z, atten_loss1 = self.return_loss( X2, X2_raw, latent_z1, mean_1, logvar_1, hidden_1,
                                                                                    size_factor2, criterion, attention_loss)
                            loss = torch.mean( loss1 + (kl_weight * kl_divergence_z)  + (args.sf1 * (atten_loss1) ) )
                else:

                    if cycle%2 == 0:

                        if state == 0 :
                            # initialization of scEpigenomics model
                            loss1, kl_divergence_z, atten_loss1 = self.return_loss( X2, X2_raw, None, None, None, None, 
                                                                                    size_factor2, criterion, attention_loss)
                            loss = torch.mean( loss1 + (kl_weight * kl_divergence_z)  )

                        else:
                            #transfer representation from scRNA-seq model to scEpigenomics model
                            result_2  = model_pre( X1, size_factor1)
                            latent_z1 = result_2["latent_z1"].to("cuda:1")
                            hidden_1  = result_2["hidden"].to("cuda:1")
                            mean_1    = result_2["mean_1"].to("cuda:1")
                            logvar_1  = result_2["logvar_1"].to("cuda:1")

                            loss1, kl_divergence_z, atten_loss1 = self.return_loss( X2, X2_raw, latent_z1, mean_1, logvar_1, hidden_1, 
                                                                                    size_factor2, criterion, attention_loss)
                            loss = torch.mean( loss1 + (kl_weight * kl_divergence_z) + (args.sf1 * (atten_loss1) ) ) 
                    
                    else:
                        if state == 0 :
                            # initialization of scRNA-seq model
                            loss1, kl_divergence_z, atten_loss1 = self.return_loss( X1, X1_raw, None, None, None, None, 
                                                                                    size_factor1, criterion, attention_loss)
                            loss = torch.mean( loss1 + (kl_weight * kl_divergence_z)  ) 

                        else:
                            #transfer representation from scEpigenomics model to scRNA-seq model
                            result_2  = model_pre( X2, size_factor2)
                            latent_z1 = result_2["latent_z1"].to("cuda:0")
                            hidden_1  = result_2["hidden"].to("cuda:0")
                            mean_1    = result_2["mean_1"].to("cuda:0")
                            logvar_1  = result_2["logvar_1"].to("cuda:0")

                            loss1, kl_divergence_z, atten_loss1 = self.return_loss( X1, X1_raw, latent_z1, mean_1, logvar_1, hidden_1, 
                                                                                    size_factor1, criterion, attention_loss)
                            loss = torch.mean( loss1 + (kl_weight * kl_divergence_z)  + (args.sf2 * (atten_loss1) ) )

                loss.backward()
                optimizer.step()

            if epoch % args.epoch_per_test == 0 and epoch > 0: 
                self.eval()

                with torch.no_grad():

                    for batch_idx, ( X1, X1_raw, size_factor1, X2, X2_raw, size_factor2 ) in enumerate(test_loader): 

                        if args.use_cuda:
                            X1, X1_raw, size_factor1 = X1.to("cuda:0"), X1_raw.to("cuda:0"), size_factor1.to("cuda:0")
                            X2, X2_raw, size_factor2 = X2.to("cuda:1"), X2_raw.to("cuda:1"), size_factor2.to("cuda:1")

                        X1, X1_raw, size_factor1 = Variable( X1 ), Variable( X1_raw ), Variable( size_factor1 )
                        X2, X2_raw, size_factor2 = Variable( X2 ), Variable( X2_raw ), Variable( size_factor2 )

                        if first == "RNA" :

                            if cycle%2 == 0:
                                if state == 0 :
                                    loss1, kl_divergence_z, atten_loss1 = self.return_loss( X1, X1_raw, None, None, None, None, 
                                                                                            size_factor1, criterion, attention_loss)
                                    test_loss = torch.mean( loss1 +  (kl_weight * kl_divergence_z)  )

                                else:
                                    result_2  = model_pre( X2, size_factor2)
                                    latent_z1 = result_2["latent_z1"].to("cuda:0")
                                    hidden_1  = result_2["hidden"].to("cuda:0")
                                    mean_1    = result_2["mean_1"].to("cuda:0")
                                    logvar_1  = result_2["logvar_1"].to("cuda:0")

                                    loss1, kl_divergence_z, atten_loss1 = self.return_loss( X1, X1_raw, latent_z1, mean_1, logvar_1, hidden_1,
                                                                                            size_factor1, criterion, attention_loss)
                                    test_loss = torch.mean( loss1 +  (kl_weight * kl_divergence_z) + (args.sf2 * (atten_loss1) ) ) 

                            else:
                                if state == 0 :
                                    loss1, kl_divergence_z, atten_loss1 = self.return_loss( X2, X2_raw, None, None, None, None, 
                                                                                            size_factor2, criterion, attention_loss)
                                    test_loss = torch.mean( loss1 + (kl_weight * kl_divergence_z)  ) 

                                else:
                                    result_2  = model_pre( X1, size_factor1)
                                    latent_z1 = result_2["latent_z1"].to("cuda:1")
                                    hidden_1  = result_2["hidden"].to("cuda:1")
                                    mean_1    = result_2["mean_1"].to("cuda:1")
                                    logvar_1  = result_2["logvar_1"].to("cuda:1")

                                    loss1, kl_divergence_z, atten_loss1 = self.return_loss( X2, X2_raw, latent_z1, mean_1, logvar_1, hidden_1,
                                                                                            size_factor2, criterion, attention_loss)
                                    test_loss = torch.mean( loss1 + (kl_weight * kl_divergence_z)  + (args.sf1 * (atten_loss1) ) )

                        else:
                            if cycle%2 == 0:

                                if state == 0 :
                                    loss1, kl_divergence_z, atten_loss1 = self.return_loss( X2, X2_raw, None, None, None, None, 
                                                                                            size_factor2, criterion, attention_loss)
                                    test_loss = torch.mean( loss1 + (kl_weight * kl_divergence_z)  )

                                else:
                                    result_2  = model_pre( X1, size_factor1)
                                    latent_z1 = result_2["latent_z1"].to("cuda:1")
                                    hidden_1  = result_2["hidden"].to("cuda:1")
                                    mean_1    = result_2["mean_1"].to("cuda:1")
                                    logvar_1  = result_2["logvar_1"].to("cuda:1")

                                    loss1, kl_divergence_z, atten_loss1 = self.return_loss( X2, X2_raw, latent_z1, mean_1, logvar_1, hidden_1, 
                                                                                            size_factor2, criterion, attention_loss)
                                    test_loss = torch.mean( loss1 + (kl_weight * kl_divergence_z) + (args.sf1 * (atten_loss1) ) ) 

                            else:
                                if state == 0 :
                                    loss1, kl_divergence_z, atten_loss1 = self.return_loss( X1, X1_raw, None, None, None, None, 
                                                                                            size_factor1, criterion, attention_loss)
                                    test_loss = torch.mean( loss1 + (kl_weight * kl_divergence_z)  ) 

                                else:
                                    result_2  = model_pre( X2, size_factor2)
                                    latent_z1 = result_2["latent_z1"].to("cuda:0")
                                    hidden_1  = result_2["hidden"].to("cuda:0")
                                    mean_1    = result_2["mean_1"].to("cuda:0")
                                    logvar_1  = result_2["logvar_1"].to("cuda:0")

                                    loss1, kl_divergence_z, atten_loss1 = self.return_loss( X1, X1_raw, latent_z1, mean_1, logvar_1, hidden_1, 
                                                                                            size_factor1, criterion, attention_loss)
                                    test_loss = torch.mean( loss1 +  (kl_weight * kl_divergence_z)  + (args.sf2 * (atten_loss1) ) )

                        train_loss_list.append( test_loss.item() )

                        print( str(epoch)+ "   " + str(test_loss.item()) + "   " + str(torch.mean(loss1).item()) + "   " + 
                               str(torch.mean(kl_divergence_z).item()) + "   " + str(torch.mean(atten_loss1).item())  )

                        if math.isnan(test_loss.item()):
                            flag_break = 1
                            break

                        if test_like_max >  test_loss.item():
                            test_like_max   = test_loss.item()
                            reco_epoch_test = epoch
                            patience_epoch  = 0        

            if flag_break == 1:
                print("containin NA")
                print(epoch)
                break

            if patience_epoch >= 30 :
                print("patient with 30")
                print(epoch)
                break
            
            if len(train_loss_list) >= 2 :
                if abs(train_loss_list[-1] - train_loss_list[-2]) / train_loss_list[-2] < 1e-4 :

                    print("converged!!!")
                    print(epoch)
                    break

        duration = time.time() - start

        print('Finish training, total time is: ' + str(duration) + 's' )
        self.eval()
        print(self.training)

        print( 'train likelihood is :  '+ str(test_like_max) + ' epoch: ' + str(reco_epoch_test) )

class DCCA(nn.Module):
    def __init__( self, layer_e_1, hidden1_1, Zdim_1, layer_d_1, hidden2_1, 
                  layer_e_2, hidden1_2, Zdim_2, layer_d_2, hidden2_2, args,
                  ground_truth, ground_truth1, Type_1 = 'NB', Type_2 = 'Bernoulli', cycle = 1, 
                  attention_loss = 'Eucli', penality = 'Gaussian', n_centroids = 4,
                  droprate = 0.1, patttern = "Both" ):

        super(DCCA, self).__init__()
        # cycle indicates the mutual learning, 0 for initiation of model1 with scRNA-seq data, 
        # and odd for training other models, even for scRNA-seq

        self.model1 = VAE(layer_e = layer_e_1, hidden1 = hidden1_1, Zdim = Zdim_1, 
                          layer_d = layer_d_1, hidden2 = hidden2_1, n_centroids = n_centroids,
                          Type    = Type_1, penality = penality, droprate = droprate )
        self.model2 = VAE(layer_e = layer_e_2, hidden1 = hidden1_2, Zdim = Zdim_2, 
                          layer_d = layer_d_2, hidden2 = hidden2_2, n_centroids = n_centroids,
                          Type    = Type_2, penality = penality, droprate = droprate)

        if attention_loss == 'NST':
            self.attention = NSTLoss()

        elif attention_loss == 'FT':
            self.attention = FactorTransfer()

        elif attention_loss == 'SL':
            self.attention = Similarity()

        elif attention_loss == 'CC':
            self.attention = Correlation()

        elif attention_loss == 'AT':
            self.attention = Attention()

        elif attention_loss == 'KL_div':
            self.attention = KL_diver()

        elif attention_loss == 'L1':
            self.attention = L1_dis()

        else:
            self.attention = Eucli_dis()

        self.cycle          = cycle
        self.args           = args
        self.ground_truth   = ground_truth
        self.ground_truth1  = ground_truth1
        self.penality       = penality
        self.attention_loss = attention_loss
        self.patttern       = patttern
    
    def fit_model( self, train_loader = None, test_loader = None, total_loader = None, first = "RNA" ):

        if self.args.use_cuda:
            self.model1.to("cuda:0")
            self.model2.to("cuda:1")

        used_cycle = 0

        ARI_score1, NMI_score1 = -100, -100
        ARI_score2, NMI_score2 = -100, -100

        if self.ground_truth is not None:
            self.evluation_metrics( total_loader )

        while used_cycle < (self.cycle+1) :

            if first == "RNA":

                if used_cycle % 2 == 0:

                    self.model2.eval()

                    if used_cycle == 0:

                        self.model1.fit( train_loader, test_loader, total_loader, self.model2, self.args, self.attention, used_cycle, 0, first, self.attention_loss  )

                    else:
                        self.model1.fit( train_loader, test_loader, total_loader, self.model2, self.args, self.attention, used_cycle, 1, first, self.attention_loss )

                else:
                    self.model1.eval()

                    if used_cycle == 1:

                        self.model2.fit( train_loader, test_loader, total_loader, self.model1, self.args, self.attention, used_cycle, 0, first, self.attention_loss )
                        
                        if self.ground_truth is not None:
                            self.evluation_metrics( total_loader )

                        if self.attention_loss is not None:
                            self.model2.fit( train_loader, test_loader, total_loader, self.model1, self.args, self.attention, used_cycle, 1, first, self.attention_loss )

                    else:
                        self.model2.fit( train_loader, test_loader, total_loader, self.model1, self.args, self.attention, used_cycle, 1, first, self.attention_loss  )

            else:
                if used_cycle % 2 == 0:

                    self.model1.eval()

                    if used_cycle == 0:

                        self.model2.fit( train_loader, test_loader, total_loader, self.model1, self.args, self.attention, used_cycle, 0, first, self.attention_loss )

                    else:
                        self.model2.fit( train_loader, test_loader, total_loader, self.model1, self.args, self.attention, used_cycle, 1, first, self.attention_loss )

                else:
                    self.model2.eval()

                    if used_cycle == 1:

                        self.model1.fit( train_loader, test_loader, total_loader, self.model2, self.args, self.attention, used_cycle, 0, first, self.attention_loss  )
                        
                        if self.ground_truth is not None:
                            self.evluation_metrics( total_loader )
                        
                        self.model1.fit( train_loader, test_loader, total_loader, self.model2, self.args, self.attention, used_cycle, 1, first, self.attention_loss  )

                    else:
                        self.model1.fit( train_loader, test_loader, total_loader, self.model2, self.args, self.attention, used_cycle, 1, first, self.attention_loss  )

            if self.ground_truth is not None:
                NMI_score1, ARI_score1, NMI_score2, ARI_score2  = self.evluation_metrics( total_loader )
   
            used_cycle = used_cycle + 1

        return NMI_score1, ARI_score1, NMI_score2, ARI_score2 

    def evluation_metrics(self, dataloader):

        self.model1.eval()
        self.model2.eval()

        kmeans1 = KMeans( n_clusters = self.args.cluster1, n_init = 5, random_state = 200 )
        kmeans2 = KMeans( n_clusters = self.args.cluster2, n_init = 5, random_state = 200 )

        latent_code_rna  = []
        latent_code_atac = []

        ARI_score1, NMI_score1 = -100, -100
        ARI_score2, NMI_score2 = -100, -100
        
        for batch_idx, ( X1, _, size_factor1, X2, _, size_factor2 ) in enumerate(dataloader): 

            if self.args.use_cuda:
                X1, size_factor1 = X1.to("cuda:0"), size_factor1.to("cuda:0")
                X2, size_factor2 = X2.to("cuda:1"), size_factor2.to("cuda:1")

            X1, size_factor1 = Variable( X1 ), Variable( size_factor1 )
            X2, size_factor2 = Variable( X2 ), Variable( size_factor2 )

            result1  = self.model1.inference( X1, size_factor1)
            result2  = self.model2.inference( X2, size_factor2)

            latent_code_rna.append( result1["latent_z1"].data.cpu().numpy() )
            latent_code_atac.append( result2["latent_z1"].data.cpu().numpy() )

        latent_code_rna  = np.concatenate(latent_code_rna)
        latent_code_atac = np.concatenate(latent_code_atac)

        pred_z1    = kmeans1.fit_predict( latent_code_rna )
        NMI_score1 = round( normalized_mutual_info_score( self.ground_truth, pred_z1,  average_method='max' ), 3 )
        ARI_score1 = round( metrics.adjusted_rand_score( self.ground_truth, pred_z1 ), 3 )

        pred_z2    = kmeans1.fit_predict( latent_code_atac )
        NMI_score2 = round( normalized_mutual_info_score( self.ground_truth, pred_z2,  average_method='max' ), 3 )
        ARI_score2 = round( metrics.adjusted_rand_score( self.ground_truth, pred_z2 ), 3 )

        print('scRNA-ARI: ' + str(ARI_score1) + ' NMI: ' + str(NMI_score1) + ' scEpigenomics-ARI: ' + str(ARI_score2) + ' NMI: ' + str(NMI_score2) )
        return NMI_score1, ARI_score1, NMI_score2, ARI_score2

    def encodeBatch(self, total_loader):
        # processing large-scale datasets
        latent_z1 = []
        latent_z2 = []
        norm_x1   = []
        recon_x1  = []
        norm_x2   = []
        recon_x2  = []

        for batch_idx, ( X1, _, size_factor1, X2, _, size_factor2 ) in enumerate(total_loader): 
            if self.args.use_cuda:
                X1, size_factor1 = X1.to("cuda:0"), size_factor1.to("cuda:0")
                X2, size_factor2 = X2.to("cuda:1"), size_factor2.to("cuda:1")

            X1, size_factor1 = Variable( X1 ), Variable( size_factor1 )
            X2, size_factor2 = Variable( X2 ), Variable( size_factor2 )

            result1  = self.model1( X1, size_factor1)
            result2  = self.model2( X2, size_factor2)

            latent_z1.append( result1["latent_z1"].data.cpu().numpy() )
            latent_z2.append( result2["latent_z1"].data.cpu().numpy() )

            norm_x1.append( result1["norm_x"].data.cpu().numpy() )
            recon_x1.append( result1["recon_x"].data.cpu().numpy() )

            norm_x2.append( result2["norm_x"].data.cpu().numpy() )
            recon_x2.append( result2["recon_x"].data.cpu().numpy() )

        latent_z1 = np.concatenate(latent_z1)
        latent_z2 = np.concatenate(latent_z2)
        norm_x1   = np.concatenate(norm_x1)
        recon_x1  = np.concatenate(recon_x1)
        norm_x2   = np.concatenate(norm_x2)
        recon_x2  = np.concatenate(recon_x2)

        return latent_z1, latent_z2, norm_x1, recon_x1, norm_x2, recon_x2

    def forward( self, total_loader = None):

        latent_z1, latent_z2, norm_x1, recon_x1, norm_x2, recon_x2 = self.encodeBatch(total_loader)

        return latent_z1, latent_z2, norm_x1, recon_x1, norm_x2, recon_x2

    def encodeBatch_latent(self, total_loader, train_order):
        # saving latent features of each model at each step
        latent_z1 = []
        latent_z2 = []

        for batch_idx, ( X1, _, size_factor1, X2, _, size_factor2 ) in enumerate(total_loader): 
            if self.args.use_cuda:
                X1, size_factor1 = X1.to("cuda:0"), size_factor1.to("cuda:0")
                X2, size_factor2 = X2.to("cuda:1"), size_factor2.to("cuda:1")

            X1, size_factor1 = Variable( X1 ), Variable( size_factor1 )
            X2, size_factor2 = Variable( X2 ), Variable( size_factor2 )

            result1  = self.model1( X1, size_factor1)
            result2  = self.model2( X2, size_factor2)

            latent_z1.append( result1["latent_z1"].data.cpu().numpy() )
            latent_z2.append( result2["latent_z1"].data.cpu().numpy() )

        latent_z1 = np.concatenate(latent_z1)
        latent_z2 = np.concatenate(latent_z2)

        latent_z11  = pd.DataFrame( latent_z1 ).to_csv( os.path.join( self.save_path, 
                                    str(self.file_label) + '_scRNA-latent-' + str(train_order) + '.csv' ) ) 
        latent_z21  = pd.DataFrame( latent_z2 ).to_csv( os.path.join( self.save_path, 
                                    str(self.file_label) + '_scATAC-latent-' + str(train_order) + '.csv' ) ) 

    def inference_other_from_rna( self, total_loader ):

        self.model1.eval()
        self.model2.eval()

        for batch_idx, ( X1, _, size_factor1, _, _, _ ) in enumerate(total_loader): 

            if self.args.use_cuda:
                X1           = X1.to("cuda:0")
                size_factor1 = size_factor1.to("cuda:0")

            X1, size_factor1 = Variable( X1 ), Variable( size_factor1 )

            result1          = self.model1( X1, size_factor1)
            latent_z1        = result1['latent_z1'].to("cuda:1")
            recons_y         = self.model2.decoder( latent_z1 )

        return recons_y.data.cpu().numpy()

