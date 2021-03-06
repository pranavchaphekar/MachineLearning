import multiprocessing
from time import time

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import MultiTaskElasticNetCV
from sklearn.linear_model import MultiTaskLasso
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from statsmodels.iolib import SimpleTable
import data_processing as dp

from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from dashboard import *


def ols_sklearn(train, test):
    '''
    Caries out Ordinary Least Square regression
    and prints the result
    :param train: Training dataframe
    :param test:  Test dataframe
    '''
    # df = pd.read_csv('data/result_panel.csv', low_memory=False)  # load into the data frame
    # filter_col = [col for col in list(df) if col.startswith('x_')]
    target = lawvar
    linear_reg = LinearRegression(normalize=True)
    linear_reg.fit(train[ols_filter_col], train[target])
    result = pd.DataFrame(list(zip(ols_filter_col, linear_reg.coef_)), columns=['features', 'coefficients'])
    expected_insample = train[target]
    expected_outsample = test[target]
    predicted_insample = linear_reg.predict(train[ols_filter_col])
    predicted_outsample = linear_reg.predict(test[ols_filter_col])
    print(result)
    print()
    print('Intercept: ' + str(linear_reg.intercept_))
    print('R-sq: ' + str(linear_reg.score(train[ols_filter_col], train[target])))
    print('in sample mse: ' + str(np.mean(predicted_insample - expected_insample) ** 2))
    print('out sample mse: ' + str(np.mean(predicted_outsample - expected_outsample) ** 2))


def fit_stat_model(df, filter_col, yvars=[lawvar], normalize=False, add_controls=True, includetext_feature_lags=False,
                   text_feature_lag=1,use_expectations=True,use_dummies=True):
    '''
    Train the model using the training data
    :param df:
    :param filter_col:
    :param target:
    :return: Linear Regression with least OLS
    '''
    final_cols = list(filter_col)
    include_list = list(filter_col)
    # Adding the expectations for the selected features
    if add_controls:
        expectations = set()
        for col in filter_col:
            if col.find("X") <= 0 and not col.startswith('pca_'):  # if not an interaction, interaction format a 'X' b
                expec_col = "e_" + col
                expectations.add(expec_col)
            elif col.find("X") >= 0:
                expectations.add('e_' + col.split('X')[0].strip())
                expectations.add('e_' + col.split('X')[1].strip())

    # Include in X the text feature lags if 'use_text_features_lag' is True
    if includetext_feature_lags:
        l = [col for col in list(df) if col.startswith('pca_') and '_lag' + str(text_feature_lag) in col]
        final_cols += l
        include_list += l
    if use_expectations:
        final_cols.extend(list(expectations))
    # if use_dummies:
    #     final_cols += [col for col in list(df) if col.startswith('dummy_')]
    X = df[final_cols]
    i = 0
    models = {}

    # Removing the Circuits where all the values are zero
    X = X.loc[:, (X != 0).any(axis=0)]
    # X = sm.add_constant(X)

    RHS = '+'.join(list(X)) + '+ C(Circuit) + C(year)'

    for yvar in yvars:
        print("Running OLS for : " + yvar)
        y = df[yvar]
        eqn = yvar + ' ~ ' + RHS
        print(eqn)
        model = smf.ols(formula=eqn, data=df)
        # model = sm.GLM(y, X,family=sm.families.Gamma())
        models[i] = model.fit(cov_type='cluster',cov_kwds={'groups':df['Circuit'], 'use_correction': True})  # cov_type='hc0'
        print(models[i].summary(xname=include_list))
        i += 1

    if len(yvars) > 1:
        compare_and_print_statsmodels(models)
    return model


def convert_textfile(model):
    '''

    :param model: Takes in the model generated by the training data
    :return: saves the .txt file to output folder
    '''
    f = open('outputfiles/ols_output.txt', "w+")
    f.write(str(model.summary()))
    f.close()


def test_stat_model(model, insample, outsample):
    '''
    Test the stat model on testing data
    :return: Accuracy summary
    '''
    filter_col = ols_filter_col
    target = lawvar

    # In sample prediction
    ypred = model.predict(insample[filter_col])
    y_actual = insample[target]
    print('MSE: (insample) ' + str(np.mean((ypred - y_actual)) ** 2))

    # Out of sample prediction
    ypred = model.predict(outsample[filter_col])
    y_actual = outsample[target]
    print('MSE: (outsample) ' + str(np.mean((ypred - y_actual)) ** 2))


# We can use class sklearn.pipeline.Pipeline(steps)
def feature_selection(df, target=lawvar, model=LassoCV()):
    characteristics_cols = [col for col in list(df) if col.startswith('x_')]

    # characteristics_cols += [col for col in list(df) if col.startswith('e_x_')]
    # characteristics_cols += [col for col in list(df) if col.startswith('dummy_')]

    #characteristics_cols.extend(dp.get_lags_features(df, 1))
    X, y = df[characteristics_cols].fillna(0), df[target]

    # model = ElasticNetCV(normalize=True,selection='random',max_iter=10000,tol=0.001)
    # model = RandomForestRegressor(max_features='sqrt')
    sfm = SelectFromModel(model, threshold=-3)
    sfm.fit(X, y)

    n_features = sfm.transform(X).shape[1]

    # Reset the threshold till the number of features equals 5.
    # Note that the attribute can be set directly instead of repeatedly
    # fitting the metatransformer.
    while n_features > 5:
        sfm.threshold += 0.0001
        X_transform = sfm.transform(X)
        n_features = X_transform.shape[1]

    features_selected = [x for (x, y) in zip(characteristics_cols, sfm.get_support()) if y == True]
    print("Features Selected: " + str(features_selected))
    return features_selected


def compare_and_print_statsmodels(estimators, indice=0):
    '''
    Prints & saves comparitive results for different
    StatsModels
    :param estimators: Different statsmodel OLS models
    '''
    print("\n")
    if indice in [0, 2]:
        data_dict = {}
        coeff = {}
        i = 1
        keys = []
        if len(estimators) > 0:
            for k, est in estimators.items():
                data_dict["(" + str(i) + ")"] = est.summary2().tables[indice].iloc[:, 1::2].stack().values
                coeff_with_err = []
                keys = []
                # for attr in dir(est):
                #    print("obj.%s = %s" % (attr, getattr(est, attr)))
                # print(type(est.params.values))
                for i in range(len(est.params.values)):
                    if not est.params.keys()[i].lower().startswith("dummy") and \
                            not est.params.keys()[i].lower().startswith("e_"):
                        coeff_with_err.append(est.params.values[i])
                        coeff_with_err.append("(" + str(est.bse.values[i]) + ")")
                        # coeff_with_err.append("(" + str(est.pvalues) + ")")
                        keys.append(est.params.keys()[i])
                        keys.append(" ")
                        # keys.append(est.params.keys()[i]+"_p_value")
                # coeff["(" + str(k) + ")"] = np.array(coeff_with_err)
                coeff["(" + str(k) + ")"] = np.array(coeff_with_err)
                keys = np.array(keys)
                i = i + 1
            index = estimators.popitem()[1].summary2().tables[indice].iloc[:, 0::2].stack().values
            df = pd.DataFrame.from_dict(data_dict)
            df2 = pd.DataFrame.from_dict(coeff)
            df2.index = keys
            tbl2 = SimpleTable(df2.values.tolist(), df2.columns.values.tolist(), df2.index.tolist(),
                               title="Coefficients")
            tbl = SimpleTable(df.values.tolist(), df.columns.values.tolist(), index.tolist(),
                              title="Regression Results")
            print(tbl)
            print(tbl2)
            df.index = index
        else:
            raise 'waiting for a dictionnary for estimators parameter'
    else:
        raise 'Not working for the coeff table'


def pca_on_text_features(df):
    '''
    :param df: High dimensional dataframe on which pca
               is performed
    :return: top n components as mentioned in the dashboard
              file
    '''
    print('running pca')
    pca = PCA(n_components=pca_components)
    #df = df.transpose()
    a = pca.fit_transform(df)
    return a


def pls_regression_on_text_features(df):
    '''
    :param df: High dimensional dataframe on which pca
               is performed
    :return: top n components as mentioned in the dashboard
              file
    '''
    pls = PLSRegression(n_components=pca_components)
    df = df.transpose()
    pls.fit_transform(df)
    return pls.components_


def Grid_search_CV(X, run_level):
    pipeline = Pipeline([
        ('randomforestregressor', RandomForestRegressor()),
        # ('AdaBoost', AdaBoostRegressor()),
        # ('GradientBoosting', GradientBoostingRegressor())

        # ('lassoCV', Lasso()),
        # ('enetCV', ElasticNet())
    ])

    parameters = {
        # 'lassoCV__alpha' : [10,1,0.1,0.01,0.001,0.0001,0.00001,0.000001],
        # 'lassoCV__max_iter' : [10,100,1000],
        # 'lassoCV__tol' : [0.0001,0.001,0.01],
        # 'enetCV__l1_ratio' : [.01, .1,.5,.7,.9, .99, 1],
        # 'enetCV__n_alphas' : [20,50,100],
        'randomforestregressor__n_estimators': [10, 40, 50, 55, 75],
        'randomforestregressor__max_features': ['auto', 'sqrt', 'log2'],
        'randomforestregressor__max_depth': [1, 5, 10, 15, 20, 25],
        # 'AdaBoost__n_estimators': [10, 40, 50, 55, 75],
        # 'AdaBoost__loss': ['linear', 'square', 'exponential']

    }

    # parameters = {'n_estimators': [10, 50, 75], 'max_depth': [1, 5, 10, 15, 20, 25] , 'max_features' : ['auto','sqrt','log2']}

    # rf_clf = RandomForestRegressor(random_state=42)

    num_cores = multiprocessing.cpu_count()

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=num_cores)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    print(parameters)

    # print_log(gridclf.best_params_)
    # print_log(gridclf.best_score_)

    t0 = time()
    del X['Circuit']
    del X['year']
    # del X['caseid']
    Z = None
    if run_level is level.panel:
        Z = dp.read_panel_level_data()
    if run_level is level.circuityear:
        Z = dp.read_circuityear_level_data()

    grid_search.fit(Z, X)
    # print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


def run_feature_selection_for_model(merged_df, Yvars, run_model=model.random_forest):
    features_selected = None
    if run_model is model.lasso:
        if run_high_dimensional:
            features_selected = feature_selection(merged_df,
                                                  model=MultiTaskLasso(selection='cyclic',
                                                                       max_iter=1e5,
                                                                       tol=1e-4),
                                                  target=Yvars)
        else:
            features_selected = feature_selection(merged_df,
                                                  model=LassoCV(n_alphas=20,
                                                                n_jobs=4,
                                                                selection='cyclic',
                                                                max_iter=1e5,
                                                                tol=1e-4)
                                                  )
    elif run_model is model.random_forest:
        features_selected = feature_selection(merged_df,
                                              model=RandomForestRegressor(max_features='sqrt'),
                                              target=Yvars)
    elif run_model is model.elastic_net:
        if run_high_dimensional:
            features_selected = feature_selection(merged_df,
                                                  model=MultiTaskElasticNetCV(),
                                                  target=Yvars)
        else:
            features_selected = feature_selection(merged_df,
                                                  model=ElasticNetCV(),
                                                  target=Yvars)
    elif run_model is model.logistic:
        features_selected = feature_selection(merged_df,
                                              model=LogisticRegression(),
                                              target=Yvars)

    return features_selected
