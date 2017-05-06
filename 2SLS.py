import pickle
import os
import pandas as pd
import numpy as np
from statsmodels.sandbox.regression import gmm
import data_processing as dp
import sys
import dashboard as db
from dashboard import Level

run_level = Level.circuityear

y_cols = ['CO','NH3','NOX','PM10','PM25','SO2','VOC']

df = pd.read_csv('data/circuityear_level_agg.csv')

pollution = pd.read_csv('data/pollutants_final.csv')

features_selected = db.ols_filter_col

expectations = set()
for col in features_selected:
    if col.find("X") is -1:                     #if not an interaction, interaction format a 'X' b
        expec_col = "e_" + col
        expectations.add(expec_col)
    elif col.find("X") >= 0:
        expectations.add('e_'+col.split('X')[0].strip())
        expectations.add('e_'+col.split('X')[1].strip())

cy_cols = ['Circuit', 'year']
cols_for_2s = features_selected + list(expectations) + [col for col in list(df) if col.startswith('dummy_')]
cols_for_merge = cy_cols + cols_for_2s

def _generate_X_():
    sys.stdout.write("\nGenerating X".ljust(50))
    X = dp.level_wise_lawvar(run_level)
    sys.stdout.write("--loaded lawvar\n")
    if db.run_high_dimensional:
        sys.stdout.write("\nGenerating Text Features".ljust(50))
        df = dp.generate_text_features_for_lawvar_cases()
        df = dp.generate_pca_of_text_features(run_level)
        X = dp.level_wise_merge(X,df,run_level)
        sys.stdout.write("--complete\n")
    sys.stdout.write("\nGenerating X".ljust(50))
    sys.stdout.write("--complete\n")
    return X

df = df[cols_for_merge]
pre_X = _generate_X_()

combined_df = pd.merge(df, pre_X, on=['Circuit', 'year'])
combined_df = pd.merge(combined_df, pollution, on=['Circuit', 'year'])
combined_df.to_csv('data/combined_2s.csv')

Z = combined_df[cols_for_2s]

# Drops columns with the same value for all rows
Z = Z.loc[:, (Z != 0).any(axis=0)]

X = combined_df[db.lawvar]
Y = combined_df[y_cols[0]]

print(Z.shape, X.shape, Y.shape)

model = gmm.IV2SLS(Y, X, Z)

model.fit()
results = model._results
results_ols2nd = model._results_ols2nd
print(results.summary())