import pandas as pd
import statsmodels.api as sm
from statsmodels.sandbox.regression import gmm
import dashboard as db
import data_processing as dp
import data_processing_2S as dp2
from linearmodels.iv import IV2SLS

run_level = db.level.circuityear

text_feature_lag = 1
env_leads = 0

y_cols = ['CO', 'NH3', 'NOX', 'PM10', 'PM25', 'SO2', 'VOC']
features_selected = ['x_dem', 'x_republican']
cy_cols = ['Circuit', 'year']

Z = dp.read_circuityear_level_data()
pca_lags = dp.generate_lags_and_leads(n_lags=1)
Z = dp.level_wise_merge(Z, pca_lags, run_level)

pre_X = dp.read_X(text_feature_lag)

# Merging X and Z
merged_X_Z = dp.level_wise_merge(pre_X, Z, run_level)

pollution = pd.read_csv('data/pollutants_final.csv')
dp2.use_lag_or_lead(pollution, y_cols, n_leads=env_leads)
pollution.dropna(subset=y_cols, inplace=True)

expectations = dp.get_expectation_col_names_for_features(features_selected)
# dummies = [col for col in list(merged_X_Z) if col.startswith('dummy_')]
dummies = []
instruments_1s = features_selected + expectations + dummies + dp.get_lags_features(merged_X_Z)

Xvars = [col for col in list(pre_X) if col not in cy_cols + instruments_1s]

combined_df = pd.merge(merged_X_Z, pollution, on=['Circuit', 'year'])
combined_df[y_cols] = (combined_df[y_cols] - combined_df[y_cols].min()) / (combined_df[y_cols].max() - combined_df[y_cols].min())
combined_df.to_csv('data/combined_2s.csv')

Z = combined_df[instruments_1s]
print(instruments_1s)

# Drops columns with the same value for all rows
Z = Z.loc[:, (Z != 0).any(axis=0)]

# Xvars.remove(db.lawvar)
X = combined_df[db.lawvar]
X = sm.add_constant(X)
Y = combined_df[y_cols[0]]

# vals = Y.values #returns a numpy array
# min_max_scaler = preprocessing.MinMaxScaler()
# y_scaled = min_max_scaler.fit_transform(vals)
# Y = pd.DataFrame(y_scaled)

# Y = (Y - Y.min()) / (Y.max() - Y.min())

instruments_1s = list(set(instruments_1s) - set(expectations))

model = gmm.IV2SLS(Y, X, Z)

yvar = y_cols[0]
instr = '+'.join(instruments_1s)
exog = '+'.join(expectations) + '+C(Circuit)+C(year)'
endog = '+'.join(Xvars)

eqn = yvar + ' ~ 1 + ' + exog + ' [' + endog + ' ~ ' + instr + ']'

lm_model = IV2SLS.from_formula(formula=eqn, data=combined_df)
result = lm_model.fit(cov_type='clustered', clusters=combined_df[['Circuit']])

model.fit()
results = model._results
results_ols2nd = model._results_ols2nd
# print(results.summary())
# print(results_ols2nd.summary())

print(result)
