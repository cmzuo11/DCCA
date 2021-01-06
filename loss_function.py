#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 19:27:03 2019

@author: chunmanzuo
"""

import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
import math
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import seaborn as sns
from torch.distributions import Normal, kl_divergence as kl


def binary_cross_entropy(recon_x, x):
    #mask = torch.sign(x)
    
    return - torch.sum(x * torch.log(recon_x + 1e-8) + (1 - x) * torch.log(1 - recon_x + 1e-8), dim=1)


def log_zinb_positive(x, mu, theta, pi, eps=1e-8):

    x = x.float()

    if theta.ndimension() == 1:

        theta = theta.view( 1, theta.size(0) ) 

    softplus_pi = F.softplus(-pi)

    log_theta_eps = torch.log( theta + eps )

    log_theta_mu_eps = torch.log( theta + mu + eps )

    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

    case_non_zero = ( -softplus_pi + pi_theta_log
        + x * ( torch.log(mu + eps) - log_theta_mu_eps )
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1) )

    mul_case_non_zero = torch.mul( (x > eps).type(torch.float32), case_non_zero )

    res = mul_case_zero + mul_case_non_zero

    return - torch.sum( res, dim = 1 )

def vae_kl_cost_weight(mean, stddev, weight, epsilon=1e-8):

    print(mean.size())
    print(stddev.size())

    aa   = torch.mean(0.5 * ( (mean).pow(2) + (stddev).pow(2) - 2.0 * torch.log(stddev + epsilon) - 1.0 ), axis=0)
    print(aa.size())
    print(aa)

    loss = torch.sum(torch.mul( aa, weight ) )

    print(loss)
    print(loss.size())

    return loss

def log_nb_positive(x, mu, theta, eps=1e-8):
    
    x = x.float()
    
    if theta.ndimension() == 1:
        theta = theta.view(
            1, theta.size(0)
        )  # In this case, we reshape theta for broadcasting

    log_theta_mu_eps = torch.log(theta + mu + eps)

    res = (
        theta * (torch.log(theta + eps) - log_theta_mu_eps)
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )

    return - torch.sum( res, dim = 1 )

def Encoders_loss_latent(Z1, Z2):
    
    Z1 = Z1.float()
    Z2 = Z2.float()

    ret = torch.pow( (Z1 - Z2) , 2)

    return torch.sum( ret, dim = 1 )


def NB_loss( y_true, y_pred, theta , eps = 1e-10 ):

    y_true = y_true.float()
    y_pred = y_pred.float()

    t1 = torch.lgamma( theta + eps ) + torch.lgamma(y_true+1.0) - torch.lgamma(y_true+theta+eps)
    t2 = (theta+y_true) * torch.log(1.0 + (y_pred/(theta+eps))) + (y_true * (torch.log(theta+eps) - torch.log(y_pred+eps)))

    final = t1 + t2
    
    return - torch.sum( final, dim = 1 )

def mse_loss(y_true, y_pred):

    mask = torch.sign(y_true)

    y_pred = y_pred.float()
    y_true = y_true.float()

    ret = torch.pow( (y_pred - y_true) * mask , 2)

    return torch.sum( ret, dim = 1 )

def poisson_loss(y_true, y_pred):

    y_pred = y_pred.float()
    y_true = y_true.float()
    
    ret = y_pred - y_true * torch.log(y_pred+1e-10) + torch.lgamma(y_true+1.0)

    return  torch.sum( ret, dim=1 )

def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) 
    y = y.unsqueeze(0) 
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)

    return torch.exp(-kernel_input) # (x_size, y_size)

def compute_mmd(x, y):
    x = x.float()
    y = y.float()
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)

    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()

    return mmd

def GMM_loss(gamma, c_params, z_params):
    """
    L elbo(x) = Eq(z,c|x)[ log p(x|z) ] - KL(q(z,c|x)||p(z,c))
              = Eq(z,c|x)[ log p(x|z) + log p(z|c) + log p(c) - log q(z|x) - log q(c|x) ]
    """
    mu_c, var_c, pi = c_params; #print(mu_c.size(), var_c.size(), pi.size())
    n_centroids = pi.size(1)
    mu, logvar = z_params
    mu_expand = mu.unsqueeze(2).expand(mu.size(0), mu.size(1), n_centroids)
    logvar_expand = logvar.unsqueeze(2).expand(logvar.size(0), logvar.size(1), n_centroids)


    # log p(z|c)
    logpzc = -0.5*torch.sum(gamma*torch.sum(math.log(2*math.pi) + \
                                           torch.log(var_c) + \
                                           torch.exp(logvar_expand)/var_c + \
                                           (mu_expand-mu_c)**2/var_c, dim=1), dim=1)
    # log p(c)
    logpc = torch.sum(gamma*torch.log(pi), 1)

    # log q(z|x) or q entropy    
    qentropy = -0.5*torch.sum(1+logvar+math.log(2*math.pi), 1)

    # log q(c|x)
    logqcx = torch.sum(gamma*torch.log(gamma), 1)

    kld = -logpzc - logpc + qentropy + logqcx

    return  kld




def plot_embedding(X, labels, classes=None, method='PCA', cmap='tab20', figsize=(4, 4), markersize=4, marker=None,
                   return_emb=False, save=False, save_emb=False, show_legend=True, show_axis_label=True, **legend_params):

    if marker is not None:
        X = np.concatenate([X, marker], axis=0)

    N = len(labels)

    if X.shape[1] != 2:

        if method == 'tSNE':
            from sklearn.manifold import TSNE
            X = TSNE(n_components=2, random_state=200).fit_transform(X)

        if method == 'UMAP':
            from umap import UMAP
            X = UMAP(n_neighbors=30, min_dist=0.3, metric='correlation').fit_transform(X)

        if method == 'PCA':
            from sklearn.decomposition import PCA
            X = PCA(n_components=2, random_state=200).fit_transform(X)
        
    plt.figure(figsize=figsize)
    if classes is None:
        classes = np.unique(labels)

    if cmap is not None:
        cmap = cmap
    elif len(classes) <= 10:
        cmap = 'tab10'
    elif len(classes) <= 20:
        cmap = 'tab20'
    else:
        cmap = 'husl'
    colors = sns.color_palette(cmap, n_colors=len(classes))
        
    for i, c in enumerate(classes):
        plt.scatter(X[:N][labels==c, 0], X[:N][labels==c, 1], s=markersize, color=colors[i], label=c)

    if marker is not None:
        plt.scatter(X[N:, 0], X[N:, 1], s=10*markersize, color='black', marker='*')
    #     plt.axis("off")
    
    legend_params_ = {'loc': 'center left',
                     'bbox_to_anchor':(1.0, 0.45),
                     'fontsize': 10,
                     'ncol': 1,
                     'frameon': False,
                     'markerscale': 1.5
                    }
    legend_params_.update(**legend_params)
    if show_legend:
        plt.legend(**legend_params_)
    sns.despine(offset=10, trim=True)
    if show_axis_label:
        plt.xlabel(method+' dim 1', fontsize=12)
        plt.ylabel(method+' dim 2', fontsize=12)

    if save:
        plt.savefig(save, format='pdf', bbox_inches='tight')
    else:
        plt.show()
        
    if save_emb:
        np.savetxt(save_emb, X)
    if return_emb:
        return X

    plt.clf()

class Eucli_dis(nn.Module):
    """like what you like: knowledge distill via neuron selectivity transfer"""
    def __init__(self):
        super(Eucli_dis, self).__init__()
        pass

    def forward(self, g_s, g_t):

        g_s = g_s.float()
        g_t = g_t.float()

        ret = torch.pow( (g_s - g_t) , 2)

        return torch.sum( ret, dim = 1 )

class NSTLoss(nn.Module):
    """like what you like: knowledge distill via neuron selectivity transfer"""
    def __init__(self):
        super(NSTLoss, self).__init__()
        pass

    def forward(self, g_s, g_t):
        return [self.nst_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]

    def nst_loss(self, f_s, f_t):

        f_s = f_s.view(f_s.shape[0], f_s.shape[1], -1)
        f_s = F.normalize(f_s, dim=2)
        f_t = f_t.view(f_t.shape[0], f_t.shape[1], -1)
        f_t = F.normalize(f_t, dim=2)

        # set full_loss as False to avoid unnecessary computation
        full_loss = False
        if full_loss:
            return (self.poly_kernel(f_t, f_t).mean().detach() + self.poly_kernel(f_s, f_s).mean()
                    - 2 * self.poly_kernel(f_s, f_t).mean())
        else:
            return self.poly_kernel(f_s, f_s).mean() - 2 * self.poly_kernel(f_s, f_t).mean()

    def poly_kernel(self, a, b):
        a = a.unsqueeze(1)
        b = b.unsqueeze(2)
        res = (a * b).sum(-1).pow(2)
        return res

class FactorTransfer(nn.Module):
    """Paraphrasing Complex Network: Network Compression via Factor Transfer, NeurIPS 2018"""
    def __init__(self, p1=2, p2=1):
        super(FactorTransfer, self).__init__()
        self.p1 = p1
        self.p2 = p2

    def forward(self, f_s, f_t):
        return self.factor_loss(f_s, f_t)

    def factor_loss(self, f_s, f_t):
        s_H, t_H = f_s.shape[2], f_t.shape[2]
        if s_H > t_H:
            f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
        elif s_H < t_H:
            f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
        else:
            pass
        if self.p2 == 1:
            return (self.factor(f_s) - self.factor(f_t)).abs().mean()
        else:
            return (self.factor(f_s) - self.factor(f_t)).pow(self.p2).mean()

    def factor(self, f):
        return F.normalize(f.pow(self.p1).mean(1).view(f.size(0), -1))

class Similarity(nn.Module):
    """Similarity-Preserving Knowledge Distillation, ICCV2019, verified by original author"""
    def __init__(self):
        super(Similarity, self).__init__()

    def forward(self, g_s, g_t):
        return [self.similarity_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]

    def similarity_loss(self, f_s, f_t):
        bsz = f_s.shape[0]
        f_s = f_s.view(bsz, -1)
        f_t = f_t.view(bsz, -1)

        G_s = torch.mm(f_s, torch.t(f_s))
        # G_s = G_s / G_s.norm(2)
        G_s = torch.nn.functional.normalize(G_s)
        G_t = torch.mm(f_t, torch.t(f_t))
        # G_t = G_t / G_t.norm(2)
        G_t = torch.nn.functional.normalize(G_t)

        G_diff = G_t - G_s
        loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
        return loss

class Correlation(nn.Module):
    """Correlation Congruence for Knowledge Distillation, ICCV 2019.
    The authors nicely shared the code with me. I restructured their code to be 
    compatible with my running framework. Credits go to the original author"""
    def __init__(self):
        super(Correlation, self).__init__()

    def forward(self, f_s, f_t):
        delta = torch.abs(f_s - f_t)
        loss = torch.mean((delta[:-1] * delta[1:]).sum(1))
        return loss

class KL_diver(nn.Module):
    
    def __init__(self):
        super(KL_diver, self).__init__()

    def forward(self, mean_1, logvar_1, mean_2, logvar_2 ):

        loss = kl( Normal(mean_1, logvar_1), 
                   Normal(mean_2, logvar_2)).sum(dim=1)

        return loss

class Attention(nn.Module):
    """Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks
    via Attention Transfer
    code: https://github.com/szagoruyko/attention-transfer"""
    def __init__(self, p=2):
        super(Attention, self).__init__()
        self.p = p

    def forward(self, g_s, g_t):
        g_s_norm = F.normalize(g_s, p=2, dim=1)
        g_t_norm = F.normalize(g_t, p=2, dim=1)
        diff_g   = g_s_norm - g_t_norm

        result   = (diff_g.norm(p=2, dim=1, keepdim=True)).sum(dim=1)

        return result
