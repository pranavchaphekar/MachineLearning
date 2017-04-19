import pickle
import pandas as pd
import sys
import numpy as np
from sklearn.linear_model import LinearRegression
from dashboard import *
import statsmodels.api as sm


def read_and_process_vote_level_data(case_ids):

    '''
    :param case_ids: Takes the case ids which are related to the environments
    :return: A csv file conatining the subset of the original data
    '''

    reader = pd.read_stata('data/BloombergVOTELEVEL_Touse.dta', iterator=True)
    df = pd.DataFrame()

    try:
        chunk = reader.get_chunk(1000)
        ctr = 1
        while len(chunk) > 0:
            chunk = chunk[chunk['caseid'].isin(case_ids)]
            df = df.append(chunk, ignore_index=True)
            sys.stdout.write(str(ctr) + ' ')
            sys.stdout.flush()
            ctr += 1
            chunk = reader.get_chunk(1000)
    except (StopIteration, KeyboardInterrupt):
        pass
    df.to_csv('filtered.csv')


def read_environmental_law_indicator():
    '''
    
    :return: 
    '''
    df = pickle.load(open('data/govt_winner.pkl', 'rb'))
    environ_cases = df[df['govt_environ'] == 1]
    environ_cases = environ_cases[['caseid', 'govt_wins']]
    return environ_cases


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
                           'x_female']  # keep only the limited set of variables (handpicked ones)
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
    grouped = df.groupby(['caseid', 'Circuit', 'year', 'month', 'govt_wins']).agg(f)
    # grouped = grouped[filter_col].apply(lambda x: (x == 1).sum()/len(x))
    grouped.to_csv('data/result_panel.csv')


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
    predicted_insample = linear_reg.predict(train[filter_col])
    predicted_outsample = linear_reg.predict(test[filter_col])
    print(result)
    print()
    print('Intercept: ' + str(linear_reg.intercept_))
    print('R-sq: ' + str(linear_reg.score(train[filter_col], train[target])))
    print('in sample mse: ' + str(np.mean((predicted_insample - expected_insample) ** 2)))
    print('out sample mse: ' + str(np.mean((predicted_outsample - expected_outsample) ** 2)))


def fit_stat_model(train, test):
    '''
    Train the model using the training data
    :return: Linear Regression with least OLS
    '''
    filter_col = ['x_dem', 'x_nonwhite', 'x_noreligion']
    target = 'govt_wins'
    model = sm.OLS(train[target], train[filter_col]).fit()
    convert_textfile(model)
    return model


def convert_textfile(model):
    '''

    :param model: Takes in the model generated by the training data
    :return: saves the .txt file to output folder
    '''
    f = open('outputfiles/ols_output.txt', "w")
    f.write(str(model.summary()))
    f.close()


def test_stat_model(model, insample, outsample):
    '''
    Test the stat model on testing data
    :return: Accuracy summary
    '''
    filter_col = ['x_dem', 'x_nonwhite', 'x_noreligion']
    target = 'govt_wins'

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
    df = pd.read_csv("data/result_circuityear.csv",low_memory=False)
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


    return year_X, actual_no_of_democrats_per_seat_Y



# This function splits the file into test and train data
def split_into_train_and_test():
    df = pd.read_csv('data/result_panel.csv', low_memory=False)  # load into the data frame
    msk = np.random.rand(len(df)) < 0.8
    train = df[msk]
    test = df[~msk]
    return train, test
    # train.to_csv('data/result_panel_train.csv')
    # test.to_csv('data/result_panel_test.csv')



# read_environmental_law_indicator()
# read_and_process_vote_level_data(read_environmental_law_indicator())
# cleaned_CSV()
# add_X_col()
# lvl_judge()
lvl_panel()
# split_into_train_and_test()
#lvl_circuityear()
#train, test = split_into_train_and_test()
# regress(train, test)
#model = fit_stat_model(train, test)
#test_stat_model(model, train, test)
actual_number_of_judges_circuit(8)
