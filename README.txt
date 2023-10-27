Readme:

Content:

The following trained VNN models are included.
1. OASIS3_VNN_NeurIPS: This folder includes the 100 VNNs that were trained on HC group in OASIS-3 dataset.


2. Data: This folder contains the following:
     a) C_OASIS.pt: normalized anatomical covariance matrix (max eigenvalue = 1) derived from cortical thickness organized according to DKT atlas for HC and AD+ groups in OASIS-3 dataset.
     b) CT_OASIS_demo.xlsx: this file contains cortical thickness data for 20 individuals (10 from HC and 10 from AD+) and diagnosis information. 
     
     
3. Demonstrations:
     a) regional_profile_demo.ipynb: This jupyter notebook demonstrates (i) the evaluation of regional profiles associated with the robustness of observing significantly elevated regional               residuals in AD+ group relative to HC group in OASIS-3, where the robustness is evaluated across 100 VNN models trained on the HC group of OASIS-3, and (ii) the association between regional residuals and eigenvectors/principal components of the anatomical covariance matrix from OASIS-3. 
     Packages used in the demonstrations:
     pandas==1.1.3
     numpy==1.19.2
     torch==1.11.0+cu113
     utils==3.6.3
     base==3.6.3
     fsbrain==0.5.3 (R package)
     matplotlib==3.3.2
     re==2.2.1
     seaborn==0.11.0
     statsmodels.api==0.13.3
     nibabel==3.2.1
     
4. delta_age_code.py: This file contains the code for brain age evaluation from VNN outputs after age-bias correction. 

5. neurips_residuals_oasis.zip: Due to space limitations, this file provides the regional residuals for the OASIS-3 dataset evaluated from 50 VNN models among the 100 that were trained on HC group. The regional residuals for all 100 VNN models are available online at https://github.com/sihags/VNN_Brain_Age.
