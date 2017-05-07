import pickle
import os
import pandas as pd
import numpy as np
from statsmodels.sandbox.regression import gmm
import data_processing as dp
import sys
import dashboard as db
from sklearn import preprocessing
import statsmodels.api as sm
import data_processing_2S as dp2

run_level = db.level.circuityear

text_feature_lag = 1
env_leads = 1

y_cols = ['CO','NH3','NOX','PM10','PM25','SO2','VOC']
features_selected = ['x_dem','x_republican','x_instate_ba X x_aba']
cy_cols = ['Circuit', 'year']


Z = dp.read_circuityear_level_data()
pre_X = dp.read_X(text_feature_lag)

# Merging X and Z
merged_X_Z = dp.level_wise_merge(pre_X, Z, run_level)

pollution = pd.read_csv('data/pollutants_final.csv')
# dp2.use_lags_and_leads(pollution, y_cols, n_leads=env_leads)


expectations = dp.get_expectation_col_names_for_features(features_selected)
dummies = [col for col in list(merged_X_Z) if col.startswith('dummy_')]
instruments_1s = features_selected + expectations + dummies + dp.get_lags_features(pre_X)


Xvars = [col for col in list(pre_X) if col not in cy_cols + instruments_1s]

combined_df = pd.merge(merged_X_Z, pollution, on=['Circuit', 'year'])
combined_df.to_csv('data/combined_2s.csv')

Z = combined_df[instruments_1s]

# Drops columns with the same value for all rows
Z = Z.loc[:, (Z != 0).any(axis=0)]

Xvars.remove(db.lawvar)
X = combined_df[Xvars]
X = sm.add_constant(X)
Y = combined_df[y_cols[0]]

# vals = Y.values #returns a numpy array
# min_max_scaler = preprocessing.MinMaxScaler()
# y_scaled = min_max_scaler.fit_transform(vals)
# Y = pd.DataFrame(y_scaled)

# Y = (Y - Y.min()) / (Y.max() - Y.min())

print(Z.shape, X.shape, Y.shape)

# Z.to_csv('data/Z.csv', index=False)
# Y.to_csv('data/Y.csv', index=False)
# X.to_csv('data/X.csv', index=False)

model = gmm.IV2SLS(Y, X, Z)

model.fit()
results = model._results
results_ols2nd = model._results_ols2nd
print(results.summary())
print(results_ols2nd.summary())