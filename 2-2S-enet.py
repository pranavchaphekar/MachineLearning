# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 09:42:08 2015

@author: elliott
"""

from sklearn.linear_model import ElasticNetCV
import numpy as np
import pandas as pd
import statsmodels.api as sm

# load first-stage-predicted Xhat matrix       
fullXhat = pd.read_pickle('Xhat.pkl')

# filter out predictors with weak first-stage F-statistics
Fstats = pd.read_pickle('Fstats.pkl')
minFstat = 12
Fkeep = (Fstats > minFstat)  
XhatF = fullXhat[:,Fkeep]

# filter out predictors with weak second-stage effect
tstats = pd.read_pickle('2S-tstats.pkl')
minTstat = 3
tkeep = np.where(np.abs(tstats[np.isfinite(tstats)])>minTstat)[0]   

Xhat = XhatF[:,tkeep]
Xhat = Xhat - np.mean(Xhat,axis=0)

# outcome variable                                      
Y = pd.read_pickle('Yvar.pkl')

enetcv = ElasticNetCV(l1_ratio=[.01, .1,.5,.7,.9, .99, 1], n_alphas=20, n_jobs=4,
              selection='random', max_iter=3000, tol=1e-6)
                            
enetcv.fit(Xhat,Y)

ypred_enet = enetcv.predict(Y)
enetX = np.where(enetcv.coef_ != 0)[0]

if len(enetX) == 0:
    ypred_postenet = None
    numSelected = 0
    print('No predictors selected.')
else:
    numSelected = len(enetX)
    print(len(enetX),'predictors selected.')
                        
    postenet = sm.OLS(Y,Xhat[:,enetX]).fit()
    ypred_postenet = postenet.predict()
                                                                                    
pd.to_pickle(enetX,'enetX.pkl')
pd.to_pickle(numSelected,'numSelected-2S.pkl')
pd.to_pickle(ypred_enet,'ypred-enet.pkl')
pd.to_pickle(ypred_postenet,'ypred-post-enet.pkl')
pd.to_pickle(enetcv.coef_ ,'enet-coefs.pkl')
pd.to_pickle(postenet.coef_ ,'post-enet-coefs.pkl')