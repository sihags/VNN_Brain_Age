import pandas as pd
import numpy as np
import torch
from numpy import zeros, newaxis

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from scipy.stats import pearsonr, zscore
import re
import os

import torch; torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.datasets import make_spd_matrix
from scipy.sparse import random
import seaborn as sns

from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter

import Modules.VNN_archit as VNNarchit
import Utils.graphML as gml

import statsmodels.api as sm
from statsmodels.stats.multicomp import (pairwise_tukeyhsd,
                                         MultiComparison)

from pingouin import anova,ancova
import nibabel
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import nibabel
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, zscore







import warnings
warnings.filterwarnings("ignore")


    
def reg_profile_oasis(training_set):
    # reg_profile_oasis: Function to evaluate (i) the regional profile corresponding to robsutness of the elevated regional residuals in the AD+ group with respect 
    # to HC group in OASIS-3 and (ii) the correlations between the regional residuals and eigenvectors of the anatomical covariance matrix. 
    
    # Training set: The dataset on which the selected VNNs were trained ('148' for OASIS-3)
    
    df_OASIS = pd.read_excel('Data/CT_OASIS_demo.xlsx')
    #df_OASIS = df_OASIS[df_OASIS['SubjID'].str.contains('Freesurfer53')].reset_index()
    
#     CT_OASIS = df_OASIS.iloc[:,2:150].to_numpy() # Cortical thickness data for OASIS-3
#     X = torch.tensor(zscore(CT_OASIS.T,axis=1))
    
    scaler = StandardScaler().set_output(transform="default")

   
    CT_OASIS_HC = df_OASIS.loc[df_OASIS['dx1'] == 'Cognitively normal'].iloc[:, 1:149].to_numpy()
    scaled_HC = scaler.fit_transform(CT_OASIS_HC)

    CT_OASIS_AD = df_OASIS.loc[df_OASIS['dx1'] == 'AD'].iloc[:, 1:149].to_numpy()
    scaled_AD = scaler.transform(CT_OASIS_AD)

    X = torch.tensor(np.concatenate((scaled_AD,scaled_HC),axis=0).T)

    
    C = torch.load('Data/C_OASIS.pt') # Anatomical covariance matrix from complete OASIS-3 dataset (normalized to have max eigenvalue =1)

    w,v = np.linalg.eig(C) # Eigendecomposition of C

    
    res_matrices_up = dict() # This variable will store the information necessary to determine the regional profile
    corr_eig = dict() # This variable will store the information necessary to determine the correlations between the regional residuals and eigenvectors of the anatomical covariance                              # matrix. 
    
    return_layers = {
    'GFL': 'GFL'
    }
    res = 148 
    disease = 'AD'

    df_y = df_OASIS.reset_index()
    samples_c = df_y.shape[0]

    xAll = torch.tensor(np.expand_dims(X, axis=1))
    perms = 100
    
    VNN = int(training_set)

    res_matrices_up[str(VNN)] = np.zeros((100,res))
    corr_eig[(str(VNN))] = np.zeros((df_y.shape[0],res))

    for i in range(perms):
        with torch.no_grad():
   
            if VNN == 148: #if selected VNNs were trained on OASIS-3
                    GNN1Ly =  VNNarchit.SelectionGNN([1,5,5], [6,10], True, nn.ReLU, [148,148], 
                                gml.NoPool, [1,1], [1], torch.zeros((148,148)) )
                    GNN2Ly =  VNNarchit.SelectionGNN([1,5,5], [6,10], True, nn.ReLU, [res,res], 
                                         gml.NoPool, [1,1], [1],C )
                    fbank = 5


            #load VNN i among 100 nominal models
            if VNN == 148:
                       GNN1Ly.load_state_dict(torch.load('OASIS3_VNN_NeurIPS/test{}.pth'.format(i)))
                    
            GNN2Ly.GFL.load_state_dict(GNN1Ly.GFL.state_dict()) 
            
            # Obtain the outputs at the final layer of VNN i
            mid_getter = MidGetter(GNN2Ly, return_layers=return_layers, keep_output=True)
            mid_outputs, model_output = mid_getter(xAll[:,:,:].T)
            
            # Evaluate regional residuals derived from the final layer outputs of VNN i
            temp_pca = ((mid_outputs['GFL'][:,0,:] - model_output))/fbank

            for j in range(1,fbank):
                 temp_pca = temp_pca + ((mid_outputs['GFL'][:,j,:] - model_output))/fbank

            # Store regional residuals for all subjects in OASIS-3 in a dataframe        
            df_pca0 = pd.DataFrame(temp_pca.detach().numpy(), columns=[str(j) for j in range(res)])
            
            # For every subject, corr_eig stores the mean of the correlations between regional residuals and the eigenvectors of C obtained from 100 VNNs
            for k in range(df_pca0.shape[0]):
                for l in range(res):
                    # Evaluate  correlation between regional residuals and $l$-the eigenvector of C for subject $k$ 
                    corr_eig[(str(VNN))][k][l] =  corr_eig[(str(VNN))][k][l] + np.abs(np.matmul(v[:,l].T, temp_pca[k]/np.linalg.norm(temp_pca[k])))                                                    /perms
 
            # 'dx1' stores the clinical label: 'Cognitively normal' for HC and 'AD' for Alzheimer's disease
            df_pca0['dx1'] = df_y['dx1']
            
            # If 'age, gender, CDR sum of boxes data is available in df_OASIS, uncomment below
            #df_pca0['Age'] = df_y['Age']
            #df_pca0['GENDER'] = df_y['GENDER']
            #df_pca0['CDRSUM'] = df_y['CDRSUM_x']

            
            F_vals1 = np.zeros(shape=(res,1))
            p_vals1 = np.zeros(shape=(res,1))
#             F_vals2 = np.zeros(shape=(res,1))
#             p_vals2 = np.zeros(shape=(res,1))
            # For every feature/brain region, evaluate group difference between AD+ and HC group using ANOVA on regional residuals for VNN i
            for j in range(res):
                key = str(j)
                r1 = anova(data=df_pca0,between = 'dx1', dv = key)
                
                # If age and gender data is available, perform ANCOVA to check if significance of the group difference is retained
                #r2 = ancova(data=df_pca0,covar=['Age','GENDER'],between = 'dx1', dv = key)
                
                # If CDR data is available, evaluate correlation between regional residuals derived from VNN i and CDR sum of boxes
                #corr_cdr[(str(VNN))][i][j] = pearsonr(df_pca0.loc[df_pca0['CDRSUM']>0,'CDRSUM'], df_pca0.loc[df_pca0['CDRSUM']>0,key])[0]
    


                F_vals1[j] = r1['F'][0]
                p_vals1[j] = r1['p-unc'][0]
                
#                 F_vals2[j] = r2['F'][0]
#                 p_vals2[j] = r2['p-unc'][0]

                # If group difference between the regional residuals for AD+ and HC is significant
                if (r1['p-unc'][0] < 0.05/res):  #and  (r2['p-unc'][0] < 0.05): 
                    # If mean of the regional residuals over HC group is smaller than the mean of regional residuals over AD+ group
                    if df_pca0.loc[df_pca0['dx1']=='Cognitively normal', str(j)].mean() < df_pca0.loc[df_pca0['dx1']==disease, 
                                                                                                 str(j)].mean():
                        # Set 1 if regional residuals at region j from VNN i are significantly lower in HC than AD+ group, otherwise 0
                        res_matrices_up[str(VNN)][i][j] = 1 
    
    
    
    c_mat = corr_eig[str(VNN)] # mean of the correlations between regional residuals and the eigenvectors of C obtained from 100 VNNs for all individuals in OASIS-3
    c_Normal = np.zeros(res)
    c_AD = np.zeros(res)
    
    # Evaluate coefficient of variation in c_mat across HC and AD+ groups
    for i in range(res):
             c_Normal[i] = np.std(c_mat[df_y['dx1'] == 'Cognitively normal',i])/(np.mean(c_mat[df_y['dx1'] == 'Cognitively normal',i])+0.000001)
             c_AD[i] = np.std(c_mat[(df_y['dx1'] == 'AD'),i])/(np.mean(c_mat[(df_y['dx1'] == 'AD') ,i]+0.000001))
    a_AD = np.zeros((res,2))
    a_controls = np.zeros((res,2))
    # Evaluate mean and standard deviation of the entries in c_mat across AD+ group
    for i in np.argwhere(c_AD<0.3).T[0]:
        a_AD[i,0]=   (np.mean(c_mat[(df_y['dx1'] == 'AD') ,i]))
        a_AD[i,1] = (np.std(c_mat[(df_y['dx1'] == 'AD'),i]))
    
    # # Evaluate mean and standard deviation of the entries in c_mat across HC group
#     for i in np.argwhere(c_Normal<0.3).T[0]:
#         a_controls[i,0]=   (np.mean(c_mat[(df_y['dx1'] == 'Cognitively normal') ,i]))
#         a_controls[i,1] = (np.std(c_mat[(df_y['dx1'] == 'AD'),i]))
        
    return res_matrices_up[str(VNN)], a_AD



def visualize_reg_profile_OASIS(reg_profile): # generate annot files for visualization of the regional profile in reg_profile (DKT atlas)

    VNN = 148
    res  =148
    res_matrix =  reg_profile.sum(axis=0)
    #res_matrix[res_matrix<=50] = 0

    a = nibabel.freesurfer.io.read_annot('Parcellations/fsaverage5/label/lh.aparc.a2009s.annot')

    viridis = cm.get_cmap('PuRd', 101) #cm.get_cmap('PuRd', 101)
    norm = plt.Normalize(0, 101)

    j_atlas = 0
    j_oasis = 0

    for i in range(int(res/2)):


        if j_atlas == 0:

            j_atlas = j_atlas + 1


        if (j_atlas == 42):
            col = viridis(norm(0))
            col = 255*np.array(col[0:3])
            a[1][j_atlas] = np.concatenate(([25,       5,      25], 
                                     np.array([a[1][i][3]]),
                                     np.array([res_matrix[0]])))
            j_atlas = j_atlas + 1


        col = viridis(norm(res_matrix[i]))
        col = 255*np.array(col[0:3])
        col = col.astype(int)

        a[1][j_atlas] = np.concatenate((np.array(col), 
                                     np.array([a[1][i][3]]),
                                     np.array([res_matrix[i]])))
        j_atlas = j_atlas +1

    nibabel.freesurfer.io.write_annot('Parcellations/fsaverage5/label/lh.OASIS_{}VNN.annot'.format(VNN),labels=a[0],ctab=a[1],names=a[2])



    a = nibabel.freesurfer.io.read_annot('Parcellations/fsaverage5/label/rh.aparc.a2009s.annot')
    j_atlas = 0
    for i in range(int(res/2)):

        if j_atlas == 0:
            j_atlas = j_atlas + 1

        if (j_atlas == 42):
            col = viridis(norm(0))
            col = 255*np.array(col[0:3])
            a[1][j_atlas] = np.concatenate(([25,       5,      25], 
                                     np.array([a[1][i][3]]),
                                     np.array([res_matrix[0]])))
            j_atlas = j_atlas + 1


        col = viridis(norm(res_matrix[i+int(res/2)]))
        col = 255*np.array(col[0:3])
        col = col.astype(int)

        a[1][j_atlas] = np.concatenate((np.array(col), 
                                     np.array([a[1][i][3]]),
                                     np.array([res_matrix[i+int(res/2)]])))

        j_atlas = j_atlas +1



    nibabel.freesurfer.io.write_annot('Parcellations/fsaverage5/label/rh.OASIS_{}VNN.annot'.format(VNN),labels=a[0],ctab=a[1],names=a[2])