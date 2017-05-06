import pickle
import pandas as pd
import sys
import numpy as np
from sklearn.cross_decomposition import PLSRegression, PLSCanonical
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LinearRegression, ElasticNet, SGDClassifier, LogisticRegression, ElasticNetCV
from dashboard import *
import statsmodels.api as sm
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
import data_visualization as d
import statsmodels.formula.api as smf




def read_and_process_vote_level_data():

    '''
    :param case_ids: Takes the case ids which are related to the environments
    :return: A csv file conatining the subset of the original data
    '''

    reader = pd.read_stata('data/prepped-data.dta', iterator=True)
    df = pd.DataFrame()

    try:
        chunk = reader.get_chunk(10)
        ctr = 1
        while len(chunk) > 0:
            # chunk = chunk[chunk['caseid'].isin(case_ids)]
            df = df.append(chunk, ignore_index=True)
            sys.stdout.write(str(ctr) + ' ')
            sys.stdout.flush()
            ctr += 1
            if(ctr == 2):
                df.to_csv('prepped-small.csv')
                break
            chunk = reader.get_chunk(1000)
    except (StopIteration, KeyboardInterrupt):
        pass
    df.to_csv('filtered.csv')


def read_data_for_appending_e():
    # Read the environmental data and keep only 2 columns which will be useful further.
    df_environmental = pd.read_csv("data/filtered.csv")
    columns_to_be_kept = ['Circuit', 'year']
    df_environmental = df_environmental[columns_to_be_kept]
    df_environmental = df_environmental.dropna(subset=['year'])
    df_environmental = df_environmental.drop_duplicates()
    df_environmental = df_environmental.sort_values(by=columns_to_be_kept)

    # read the Bloomberg_to_use.dta in chunks
    # itr = pd.read_stata('data/BloombergVOTELEVEL_Touse.dta', chunksize=100000)
    reader = pd.read_csv('data/filtered.csv', iterator=True)
    df_final = []
    # print(type(itr))
    try:
        chunk = reader.get_chunk(100)
        ctr = 1
        while len(chunk) > 0:
            df = filter_chunk_acc_to_env_cases(chunk, df_environmental)
            df1 = add_col_and_grp(df)
            df_final.append(df1)
            sys.stdout.write(str(ctr) + ' ')
            sys.stdout.flush()
            ctr += 1
            chunk = reader.get_chunk(100)
    except:
        pass
    concat = pd.concat(df_final)
    concat = combine_with_env(concat)
    concat.to_csv('concat2.csv')
    concat = group_on_circuit_year(concat)
    concat.to_csv('concat1.csv')


def filter_chunk_acc_to_env_cases(df, df_environmental):
    keys = ['Circuit', 'year']
    df = df.dropna(subset=['year'])
    joined = pd.merge(df, df_environmental, on=keys, how='inner')
    # df = df.sort_values(by=keys)
    # df_environmental = df_environmental.sort_values(by=keys)
    # df.update(df_environmental)
    return joined


def add_col_and_grp(df):
    # add a new avg_x and total_x col for dem

    filter_col = [col for col in list(df) if col.startswith('x_')]
    for col in filter_col:
        count_ones = '1_' + col
        total = 'total_' + col
        df[count_ones] = df[col]
        df[total] = df[col]

    # df['1_x_dem'] = df['x_dem']
    # df['total_x_dem'] = df['x_dem']

    countFun = lambda x: len(x)
    sumFun = lambda x: (x == 1).sum()

    f = {}
    # f['1_x_dem'] = sumFun
    # f['total_x_dem'] = countFun

    for col in filter_col:
        count_ones = '1_' + col
        total = 'total_' + col
        f[count_ones] = sumFun
        f[total] = countFun

    df = df.groupby(['Circuit', 'year'], as_index=False).agg(f)

    return df


def combine_with_env(df):
    df_environmental = pd.read_csv("data/filtered.csv")
    keys = ['Circuit', 'year']
    df = pd.merge(df, df_environmental, on=keys, how='inner')
    # df = df.sort_values(by=keys)
    # df_environmental = df_environmental.sort_values(by=keys)
    # df.update(df_environmental)
    return df

    features_to_be_used_for_expectation = ['x_dem', 'x_republican',
                           'x_instate_ba',
                           'x_aba', 'x_protestant', 'x_evangelical', 'x_noreligion', 'x_catholic', 'x_jewish',
                           'x_black', 'x_nonwhite',
                           'x_female']  # keep only the limited set of variables (handpicked ones)

def group_on_circuit_year(df):
    filter_col = [col for col in list(df) if col.startswith('x_')]

    meanFun = lambda x: np.average(x)
    sumFun = lambda x: x.sum()
    f = {}
    # f['1_x_dem'] = sumFun
    # f['total_x_dem'] = sumFun

    for col in filter_col:
        count_ones = '1_' + col
        total = 'total_' + col
        f[count_ones] = sumFun
        f[total] = sumFun
        # f[col] = meanFun

    df = df.groupby(['Circuit', 'year'], as_index=False).agg(f)
    # df['e_x_dem'] = df['1_x_dem'] / df['total_x_dem']
    for col in filter_col:
        count_ones = '1_' + col
        total = 'total_' + col
        e_col = 'e_' + col
        df[e_col] = df[count_ones] / df[total]
        del df[count_ones]
        del df[total]

    return df


def merge_expectations_with_lvl_circuit():
    df1 = pd.read_csv('circuityear_level_agg.csv')
    df2 = pd.read_csv('concat1.csv')
    print(df1['Circuit'])
    print(df2['Circuit'])
    df = pd.merge(df1, df2, on=['Circuit', 'year'], how="inner")
    df = df.sort_values(['Circuit', 'year'])
    del df['Unnamed: 0']

    df.to_csv('final_filtered.csv')

def merge_for_panel():
    df1 = pd.read_csv('concat1.csv')
    df2 = pd.read_csv('data/result_panel.csv')

    df = pd.merge(df2, df1, on=['Circuit', 'year'])
    del df['Unnamed: 0_x']
    del df['Unnamed: 0_y']
    print(list(df))
    df = df.sort_values(['Circuit', 'year'])
    df.to_csv('filtered1.csv')
    # df2 = pd.read_csv('filtered1.csv')


def read_environmental_law_indicator():
    '''
    
    :return: 
    '''
    df = pickle.load(open('data/govt_winner.pkl', 'rb'))
    df.to_csv('govt_winner.csv')
    # environ_cases = df[df['govt_environ'] == 1]
    # environ_cases = environ_cases[['caseid', 'govt_wins']]
    # return environ_cases


def cleaned_CSV():
    '''
    
    :return: 
    '''
    df = pd.read_csv('filtered.csv', low_memory=False)
    # read the handpicked attributes from a file into a list
    features = []
    with open("handpicked_features.txt") as file:
        for line in file:
            line = line.strip()  # for removing the spaces
            features.append(line)
    df = df.drop_duplicates()  # drop the duplicate rows
    lines_subset = df[features]
    lines_subset.to_csv('filtered_subset.csv')


def add_X_col():
    '''
    
    :return: 
    '''
    df = pd.read_csv('filtered_subset.csv')
    df = pd.merge(df, read_environmental_law_indicator(), on='caseid')
    df.to_csv('add_X_col.csv')


def gen_inter(main_df, df1, df2, df1_col_name, df2_col_name):
    '''
    
    :param main_df: 
    :param df1: 
    :param df2: 
    :param df1_col_name: 
    :param df2_col_name: 
    :return: 
    '''
    name_of_dataframe = df1_col_name + ' X ' + df2_col_name
    main_df[name_of_dataframe] = df1 * df2
    return main_df


def lvl_judge():
    df = pd.read_csv('data/filtered.csv', low_memory=False)  # load into the data frame
    df = pd.merge(df, read_environmental_law_indicator(), on='caseid')
    features_to_be_kept = ['caseid', 'year', 'Circuit', 'month', 'govt_wins', 'songername', 'x_dem', 'x_republican',
                           'x_instate_ba',
                           'x_aba', 'x_protestant', 'x_evangelical', 'x_noreligion', 'x_catholic', 'x_jewish',
                           'x_black', 'x_nonwhite',
                           'x_female', 'dissentvote']  # keep only the limited set of variables (handpicked ones)
    df_lvl_judge = df[features_to_be_kept]  # creates a new data frame with only few handpicked features

    interaction_list, non_interaction_list = [], []

    for col in list(df_lvl_judge):  # only the variables chosen for cross product
        interaction_list.append(col) if col.startswith('x_') else non_interaction_list.append(col)

    df_subset_non_interactions = df_lvl_judge[non_interaction_list]
    df_subset_dataframe = df_lvl_judge[interaction_list]
    # Replace Nan with 0
    df_subset_dataframe = df_subset_dataframe.replace(np.nan, 0)
    # Merge the two data frames
    for feature in interaction_list:
        interaction_list_subset = interaction_list
        interaction_list_subset.remove(feature)
        for other_feature in interaction_list_subset:
            df1 = df_lvl_judge[feature]
            df2 = df_lvl_judge[other_feature]
            df_subset_dataframe = gen_inter(df_subset_dataframe, df1, df2, feature, other_feature)

    result = pd.concat([df_subset_non_interactions, df_subset_dataframe], axis=1)

    result['judge_opinion'] = 1 - result['dissentvote']  # not dissent
    result = result.fillna(0)
    # Order of elements, Sorting
    sort_order = ['Circuit', 'year', 'month']
    # Sorting by the column enteries and store that in result dataframe
    result = result.sort_values(by=sort_order)
    result.to_csv('data/result_judge.csv')


def lvl_panel():
    df = pd.read_csv('data/result_judge.csv', low_memory=False)  # load into the data frame
    filter_col = [col for col in list(df) if col.startswith('x_')]
    # print(filter_col)
    df.insert(5, 'num_judges', 1)
    countFun = lambda x: len(x)
    meanFun = lambda x: np.average(x)
    f = {}
    f['num_judges'] = countFun
    for col in filter_col:
        f[col] = meanFun
    grouped = df.groupby(['caseid', 'Circuit', 'year', 'month', 'govt_wins'], as_index=False).agg(f)
    # grouped = grouped[filter_col].apply(lambda x: (x == 1).sum()/len(x))
    grouped = merge_with_dummies(grouped)

    grouped.to_csv('data/result_panel.csv')


def merge_with_dummies(df):
    df_dummies = pd.get_dummies(df['Circuit'], prefix="dummy")
    df = pd.concat([df, df_dummies], axis=1)
    df_dummies = pd.get_dummies(df['year'], prefix="dummy")
    df = pd.concat([df, df_dummies], axis=1)
    return df

def regress(train, test):
    # df = pd.read_csv('data/result_panel.csv', low_memory=False)  # load into the data frame
    # filter_col = [col for col in list(df) if col.startswith('x_')]
    filter_col = ['x_dem', 'x_nonwhite', 'x_noreligion']
    target = 'govt_wins'
    linear_reg = LinearRegression(normalize=True)
    linear_reg.fit(train[filter_col], train[target])
    result = pd.DataFrame(list(zip(filter_col, linear_reg.coef_)), columns=['features', 'coefficients'])
    expected_insample = train['govt_wins']
    expected_outsample = test['govt_wins']
    predicted_insample = linear_reg.predict()
    predicted_outsample = linear_reg.predict(test[filter_col])
    print(predicted_insample)
    print(result)
    print()
    print('Intercept: ' + str(linear_reg.intercept_))
    print('R-sq: ' + str(linear_reg.score(train[filter_col], train[target])))
    print('in sample mse: ' + str(np.mean((predicted_insample - expected_insample) ** 2)))
    print('out sample mse: ' + str(np.mean((predicted_outsample - expected_outsample) ** 2)))


def fit_stat_model(train, target = 'govt_wins'):
    '''
    Train the model using the training data
    :return: Linear Regression with least OLS
    '''
    filter_col = ['x_dem', 'x_nonwhite', 'x_noreligion']
    target = 'govt_wins'
    model = sm.OLS(train[target], train[filter_col]).fit()
    convert_textfile(model)
    print(model.predict().shape, train.shape)
    return model


def convert_textfile(model):
    '''

    :param model: Takes in the model generated by the training data
    :return: saves the .txt file to output folder
    '''
    f = open('outputfiles/ols_output.txt', "w")
    f.write(str(model.summary()))
    f.close()


def test_stat_model(model, insample, outsample, target):
    '''
    Test the stat model on testing data
    :return: Accuracy summary
    '''
    filter_col = ['x_dem', 'x_nonwhite', 'x_noreligion']

    # In sample prediction
    ypred = model.predict(insample[filter_col])
    y_actual = insample[target]
    print('mse: (insample) ' + str(np.mean((ypred - y_actual)) ** 2))

    # Out of sample prediction
    ypred = model.predict(outsample[filter_col])
    y_actual = outsample[target]
    print('mse: (outsample) ' + str(np.mean((ypred - y_actual)) ** 2))


def lvl_circuityear():
    df = pd.read_csv("data/result_panel.csv", low_memory=False)
    # df = pd.merge(df, read_environmental_law_indicator(), on='caseid')

    X_star = []
    E_star = []
    for col in list(df.columns.values):
        if str(col).lower().startswith('x_'):
            X_star.append(str(col))
        elif str(col).lower().startswith('e_'):
            E_star.append(str(col))

    ##PANELVOTE & PROTAKING Columns Currently not There!!!!!!
    # df[df.panelvote in (2,3)]['proplaintiff'] = 1
    # df[df.protaking == 1]['proplaintiff'] = 0
    # df[df.protaking == 0]['proplaintiff'] = 1

    print(list(df))

    # Generating and renaming variables so that they have appropriate names after collapsing gen
    # df.rename(columns={lawvar:'numCasesPro','caseid' :'numCases'}, inplace=True)

    df['numCasesPro'] = df[lawvar]
    df['numCases'] = 1
    df['numCasesAnti'] = 1 - df[lawvar]

    sort_order = ['Circuit', 'year']
    # Sorting by the column enteries and store that in result dataframe
    df = df.sort_values(sort_order)
    df.fillna(df.mean())
    # df[df.numCases == 0]['present'] = 1

    # Define a lambda function to compute the weighted mean:
    meanFun = lambda x: np.average(x)
    sumFun = lambda x: (x == 1).sum()
    lenFun = lambda x: len(x)

    f = {}
    f['numCases'] = sumFun
    f['numCasesAnti'] = sumFun
    f['numCasesPro'] = sumFun
    f['govt_wins'] = sumFun
    for col in X_star:
        f[col] = meanFun

    df = df.groupby(["Circuit", "year"]).agg(f)
    # df = df.sort_values(sort_order)

    # Adding a NewColumn for Clustering CircuitXYear
    # df['circuitXyear'] = df.Circuit.astype(str).str.cat(df.year.astype(str), sep='X')

    df.to_csv('data/result_lvlcircuit.csv')

    df.to_csv('data/result_lvlcircuit.csv')

def actual_number_of_judges_circuit(circuit_no):
    '''
    This method adds the "actual number" column in the table generated after result_lvlcircuit
    Circuit years from 1974 to 2013
    Circuit no from
    :param circuit_no:
    :return:
    '''
    df = pd.read_csv("data/result_circuityear.csv", low_memory=False)
    df1 = df.loc[df['Circuit'] == circuit_no]
    year_X = []
    actual_no_of_democrats_per_seat_Y = []

    #finding the start and end year if not mentioned explicitly
    start_year = (int)(df1['year'].min())
    end_year = (int)(df1['year'].max())

    #Sort acc to the years
    df1 = df1.sort_values(['year'])

    print(start_year)
    print(end_year)

    for year_no in df1['year'].values:
        if year_no in range(start_year, end_year):
            year_X.append((int)(year_no))
            democrats_no = df1.loc[df1['year'] == year_no, 'x_dem'].values[0]
            republican_no = df1.loc[df1['year'] == year_no, 'x_republican'].values[0]
            actual_no_of_democrats_per_seat = (democrats_no) / (democrats_no + republican_no)
            actual_no_of_democrats_per_seat_Y.append(actual_no_of_democrats_per_seat)
    columns = ['Year', 'Actual democrats', 'headers']
    df = pd.DataFrame(columns=columns)
    a = d.DataVisualization()
    g = []
    for x in actual_no_of_democrats_per_seat_Y:
        g.append(x*2)
    a.scatter_plot(year_X,actual_no_of_democrats_per_seat_Y)
    a.scatter_plot(year_X, g)
    a.show_plot()
    #a.line_curve(df,["Year","Actual democrats"],"Year")
    return year_X, actual_no_of_democrats_per_seat_Y



# This function splits the file into test and train data
def split_into_train_and_test(data_path):
    df = pd.read_csv(data_path, low_memory=False)  # load into the data frame
    msk = np.random.rand(len(df)) < 1
    train = df[msk]
    test = df[~msk]
    return train, test
    # train.to_csv('data/result_panel_train.csv')
    # test.to_csv('data/result_panel_test.csv')


def lasso_for_feature_selection(df, target='govt_wins'):
    characteristics_cols = [col for col in list(df) if col.startswith('x_')]
    # characteristics_cols += [col for col in list(df) if col.startswith('e_x_')]
    # characteristics_cols += [col for col in list(df) if col.startswith('dummy_')]
    X, y = df[characteristics_cols].fillna(0), df[target]
    clf = LassoCV()
    sfm = SelectFromModel(clf, threshold= 0 )
    sfm.fit(X, y)

    n_features = sfm.transform(X).shape[1]

    # Reset the threshold till the number of features equals two.
    # Note that the attribute can be set directly instead of repeatedly
    # fitting the metatransformer.
    while n_features > 5:
        sfm.threshold += 0.0001
        X_transform = sfm.transform(X)
        n_features = X_transform.shape[1]

    features_selected = [x for (x, y) in zip(characteristics_cols, sfm.get_support()) if y == True]
    print(features_selected)
    return features_selected


def random_forest_for_feature_selection(df, target = 'govt_wins'):
    characteristics_col = [col for col in list(df) if col.startswith('x_')]
    X, y = df[characteristics_col].fillna(0), df[target]
    # del df['Unnamed: 0']
    enetcv = ElasticNetCV(tol=1e-2)
    # enetcv.fit(X, y)
    # print(enetcv.coef_)
    sfm = SelectFromModel(enetcv, threshold = -3 )
    sfm.fit(X, y)

    n_features = sfm.transform(X).shape[1]

    # Reset the threshold till the number of features equals two.
    # Note that the attribute can be set directly instead of repeatedly
    # fitting the metatransformer.
    while n_features > 5:
        sfm.threshold += 0.0001
        X_transform = sfm.transform(X)
        n_features = X_transform.shape[1]

    features_selected = [x for (x, y) in zip(characteristics_col, sfm.get_support()) if y == True]
    print(features_selected)
    return features_selected

def clustering_SE():
    df = pd.read_csv('data/circuityear_level_agg.csv')
    del df['Unnamed: 0']
    y = df['govt_wins']
    X = df
    del X['Circuit']
    del X['year']
    # # print(list(X))
    # X.drop('Circuit', axis=1, inplace=True)
    # X.drop('year', axis=1, inplace=True)
    # # # del df['govt_wins']
    # cols_to_iterate = df.columns.values.tolist()
    # cols_to_iterate.remove('Circuit')
    # cols_to_iterate.remove('year')
    # cols = cols_to_iterate[0]
    # del cols_to_iterate[0]
    # for col_name in cols_to_iterate:
    #          cols = cols + " + " + col_name
    # cols = cols + "C(Circuit)"
    # my_formula = "govt_wins ~ " + cols
    model = sm.OLS(y, X).fit()
    # model = smf.ols(formula='govt_wins ~ C(Circuit) + C(year) + x_dem + x_noreligion ', data=df)
    # print(model.summary())
    # result = model.fit(cov_type='cluster', cov_kwds={'groups': (df['Circuit'], df['year'])})
    print(model.summary())


        # importances = model.feature_importances_
    # std = np.std([tree.feature_importances_ for tree in model.estimators_],
    #              axis=0)
    # indices = np.argsort(importances)[::-1]
    #
    # # Print the feature ranking
    # print("Feature ranking:")
    #
    # for f in range(X.shape[1]):
    #     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


    # print(list(df.columns.values))
    # print(model.feature_importances_)
    # print("Features sorted by their score:")
    # print(sorted(zip(map(lambda x: round(x, 4), model.feature_importances_)),
    #        reverse=True))

# read_environmental_law_indicator()
read_and_process_vote_level_data()
# cleaned_CSV()
# add_X_col()
# lvl_judge()
lvl_panel()
# split_into_train_and_test()
# lvl_circuityear()
# train, test = split_into_train_and_test('data/result_panel.csv')
# regress(train, test)
# fit_stat_model(train)
# model_judge_lvl = fit_stat_model(train, 'judge_opinion')
# model_panel_lvl = fit_stat_model(train, 'govt_wins')

# test_stat_model(model_judge_lvl, train, test, 'judge_opinion')
# lasso_for_feature_selection(train)
# test_stat_model(model_panel_lvl, train, test, 'govt_wins')

# df = pd.read_csv('data/result_panel.csv', low_memory=False)  # load into the data frame
# lasso_for_feature_selection(train)
# actual_number_of_judges_circuit(8)
# group_and_aggregate()
# read_data_for_appending_e()
# merge_expectations_with_lvl_circuit()
# merge_for_panel()
# random_forest_for_feature_selection(train)
# read_court_rulings()
# read_environmental_law_indicator()
# clustering_SE()