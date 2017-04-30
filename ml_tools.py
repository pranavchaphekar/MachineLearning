import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from statsmodels.iolib import SimpleTable
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


def fit_stat_model(df, filter_col, target=lawvar):
    '''
    Train the model using the training data
    :return: Linear Regression with least OLS
    '''
    y = df[target]
    # X = sm.add_constant(df[filter_col])
    X = df[filter_col]
    model = sm.OLS(y, X).fit()
    # convert_textfile(model)
    print(model.summary())
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
def feature_selection(df, target=lawvar, model = LassoCV()):
    characteristics_cols = [col for col in list(df) if col.startswith('x_')]
    X, y = df[characteristics_cols], df[target]
    # clf = LassoCV()
    # Use ExtraTreesClassifier() for Random Forest
    sfm = SelectFromModel(model, threshold=0)
    sfm.fit(X, y)

    n_features = sfm.transform(X).shape[1]

    # Reset the threshold till the number of features equals two.
    # Note that the attribute can be set directly instead of repeatedly
    # fitting the metatransformer.
    while n_features > 5:
        sfm.threshold += 0.05
        X_transform = sfm.transform(X)
        n_features = X_transform.shape[1]

    features_selected = [x for (x, y) in zip(characteristics_cols, sfm.get_support()) if y == True]
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
                coeff_with_err=[]
                keys = []
                #for attr in dir(est):
                #    print("obj.%s = %s" % (attr, getattr(est, attr)))
                #print(type(est.params.values))
                for i in range(len(est.params.values)):
                    coeff_with_err.append(est.params.values[i])
                    coeff_with_err.append("("+str(est.bse.values[i])+")")
                    #coeff_with_err.append("(" + str(est.pvalues) + ")")
                    keys.append(est.params.keys()[i])
                    keys.append(est.params.keys()[i]+" ")
                    #keys.append(est.params.keys()[i]+"_p_value")
                #coeff["(" + str(k) + ")"] = np.array(coeff_with_err)
                coeff["(" + str(k) + ")"] = np.array(coeff_with_err)
                keys = np.array(keys)
                i = i + 1
            index = estimators.popitem()[1].summary2().tables[indice].iloc[:, 0::2].stack().values
            df = pd.DataFrame.from_dict(data_dict)
            df2 = pd.DataFrame.from_dict(coeff)
            df2.index = keys
            tbl2 = SimpleTable(df2.values.tolist(), df2.columns.values.tolist(), df2.index.tolist(),
                               title="Coefficients")
            tbl = SimpleTable(df.values.tolist(), df.columns.values.tolist(), index.tolist(), title="Regression Results")
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
    pca = PCA(n_components=pca_components)
    df = df.transpose()
    pca.fit_transform(df)
    return pca.components_

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