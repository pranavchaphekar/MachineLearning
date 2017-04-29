import pickle
import pandas as pd
import sys
import numpy as np
from dashboard import *
from sklearn.ensemble import ExtraTreesClassifier


######################
# Read Data Methods
######################
def read_and_process_vote_level_data():
    """
    Takes the case ids which are related to the the case types,
    eg here environmental cases
    :return: A csv file containing the subset of the original data
    """
    case_ids = read_case_ids()
    reader = pd.read_stata(characteristic_data_path, iterator=True)
    df = pd.DataFrame()

    try:
        chunk = reader.get_chunk(1000)
        ctr = 1
        sys.stdout.write("Loading Chunk : " + ' ')
        while len(chunk) > 0:
            chunk = chunk[chunk[case_id_column].isin(case_ids)]
            df = df.append(chunk, ignore_index=True)
            sys.stdout.write(str(ctr) + ' ')
            sys.stdout.flush()
            ctr += 1
            chunk = reader.get_chunk(1000)
    except (StopIteration, KeyboardInterrupt):
        pass
    df.to_csv(filtered_char_data_path)


def read_filtered_data_into_dataframe():
    df = pd.read_csv(filtered_char_data_path, low_memory=False)  # load into the data frame
    return df


def read_handpicked_features_data_into_dataframe():
    df = pd.read_csv(handpicked_char_data_path, low_memory=False)  # load into the data frame
    return df


def read_char_with_legal_data():
    df = pd.read_csv(char_with_legal_data, low_memory=False)  # load into the data frame
    return df


def read_case_ids():
    """
    Reads the case ids of the cases with lawvars
    and returns the list
    :return:
    """
    df = pickle.load(open(lawvar_caseid_decision_file, 'rb'))
    cases = df[df[case_type] == 1]
    cases = cases[['caseid', lawvar]]
    return cases


######################
# Clean Data Methods
######################
def clean_nan_values(df):
    return df.replace(np.nan, 0)


def clean_na_values(df):
    return df.fillna(0)


def handpick_features_from_char_data(df):
    """

    :return:
    """
    df[features_to_use].to_csv(handpicked_char_data_path)
    return df[features_to_use]


######################
# Merge Data Methods
######################

def merge_with_legal_area_data(df, legal_area_data):
    """

    :return:
    """
    # df = pd.read_csv('filtered_subset.csv')
    df = pd.merge(df, legal_area_data, on='caseid')
    df.to_csv(char_with_legal_data)
    return df


def gen_interactions(main_df, df1, df2, df1_col_name, df2_col_name):
    """

    :param main_df:
    :param df1:
    :param df2:
    :param df1_col_name:
    :param df2_col_name:
    :return:
    """
    name_of_dataframe = df1_col_name + ' X ' + df2_col_name
    main_df[name_of_dataframe] = df1 * df2
    return main_df


def merge_char_with_legal_data(df):
    df = pd.merge(df, read_case_ids(), on='caseid')
    del df['Unnamed: 0']
    df.to_csv(char_with_legal_data)
    return df


######################
# Data Agregation Methods
######################

def aggregate_on_judge_level(df):
    '''
    Aggregates data on Judge Level.Each case has 3 rows.
    Used to generate panel level files
    '''
    interaction_list, non_interaction_list = [], []

    for col in list(df):  # only the variables chosen for cross product
        interaction_list.append(col) if col.startswith('x_') else non_interaction_list.append(col)

    df_subset_non_interactions = df[non_interaction_list]
    df_subset_dataframe = df[interaction_list]

    # Replace Nan with 0
    df_subset_dataframe = df_subset_dataframe.replace(np.nan, 0)

    # Merge the two data frames
    for feature in interaction_list:
        interaction_list_subset = interaction_list
        interaction_list_subset.remove(feature)
        for other_feature in interaction_list_subset:
            df1 = df[feature]
            df2 = df[other_feature]
            df_subset_dataframe = gen_interactions(df_subset_dataframe, df1, df2, feature, other_feature)

    result = pd.concat([df_subset_non_interactions, df_subset_dataframe], axis=1)
    result['judge_opinion'] = 1 - result['dissentvote']  # not dissent
    result = result.fillna(0)
    # Order of elements, Sorting
    sort_order = ['Circuit', 'year', 'month']
    # Sorting by the column enteries and store that in result dataframe
    result = result.sort_values(by=sort_order)
    result.to_csv(judge_level_file)


def aggregate_on_panel_level():
    df = pd.read_csv(judge_level_file, low_memory=False)  # load into the data frame
    filter_col = [col for col in list(df) if col.startswith('x_')]

    df.insert(5, 'num_judges', 1)
    countFun = lambda x: len(x)
    meanFun = lambda x: np.average(x)
    f = dict()
    f['num_judges'] = countFun
    for col in filter_col:
        f[col] = meanFun
    grouped = df.groupby(panel_level_grouping_columns).agg(f)
    grouped.to_csv(panel_level_file)


def aggregate_on_circuityear_level():
    df = pd.read_csv(panel_level_file, low_memory=False)

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
    f[lawvar] = sumFun
    for col in X_star:
        f[col] = meanFun

    df = df.groupby(["Circuit", "year"], as_index=False).agg(f)
    # df = df.sort_values(sort_order)

    # Adding a NewColumn for Clustering CircuitXYear
    # df['circuitXyear'] = df.Circuit.astype(str).str.cat(df.year.astype(str), sep='X')

    df.to_csv(circuityear_level_file)


def read_judge_level_data():
    df = pd.read_csv(judge_level_file, low_memory=False)  # load into the data frame
    return df


def read_panel_level_data():
    df = pd.read_csv(panel_level_file, low_memory=False)  # load into the data frame
    return df


def read_circuityear_level_data():
    df = pd.read_csv(circuityear_level_file, low_memory=False)  # load into the data frame
    return df


def read_expectations_data():
    df = pd.read_csv(generated_circuityear_expectations_file, low_memory=False)  # load into the data frame
    return df

def read_lags_leads_data():
    df = pd.read_csv(lags_leads_file, low_memory=False)
    return df


# This function splits the file into test and train data
def split_into_train_and_test(df):
    msk = np.random.rand(len(df)) < train_test_split
    train = df[msk]
    test = df[~msk]
    return train, test


#######################
# Epectations
#######################
def generate_expectations():
    # Read the environmental data and keep only 2 columns which will be useful further.
    df_environmental = pd.read_csv(characteristic_data_path)
    columns_to_be_kept = ['Circuit', 'year']
    df_environmental = df_environmental[columns_to_be_kept]
    df_environmental = df_environmental.dropna(subset=['year'])
    df_environmental = df_environmental.drop_duplicates()
    df_environmental = df_environmental.sort_values(by=columns_to_be_kept)

    reader = pd.read_csv('data/filtered.csv', iterator=True)
    df_final = []
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
    concat = group_on_circuit_year(concat)
    concat.to_csv('concat1.csv')


def filter_chunk_acc_to_env_cases(df, df_environmental):
    keys = ['Circuit', 'year']
    df = df.dropna(subset=['year'])
    joined = pd.merge(df, df_environmental, on=keys, how='inner')
    return joined


def add_col_and_grp(df):
    filter_col = [col for col in list(df) if col.startswith('x_')]
    for col in filter_col:
        count_ones = '1_' + col
        total = 'total_' + col
        df[count_ones] = df[col]
        df[total] = df[col]

    countFun = lambda x: len(x)
    sumFun = lambda x: (x == 1).sum()

    f = {}

    for col in filter_col:
        count_ones = '1_' + col
        total = 'total_' + col
        f[count_ones] = sumFun
        f[total] = countFun

    df = df.groupby(['Circuit', 'year'], as_index=False).agg(f)

    return df


def combine_with_env(df):
    df_environmental = pd.read_csv(filtered_char_data_path)
    keys = ['Circuit', 'year']
    df = pd.merge(df, df_environmental, on=keys, how='inner')
    return df


def group_on_circuit_year(df):
    # X_* Features
    filter_col = [col for col in list(df) if col.startswith('x_')]
    sumFun = lambda x: x.sum()
    f = {}
    for col in filter_col:
        count_ones = '1_' + col
        total = 'total_' + col
        f[count_ones] = sumFun
        f[total] = sumFun
    df = df.groupby(['Circuit', 'year'], as_index=False).agg(f)
    for col in filter_col:
        count_ones = '1_' + col
        total = 'total_' + col
        e_col = 'e_' + col
        df[e_col] = df[count_ones] / df[total]
        del df[count_ones]
        del df[total]
    return df


#######################
# Lags and Leads
#######################
def generate_lags_and_leads(features, n_lags=1, n_leads=1):
    df = read_circuityear_level_data()
    keys = ['Circuit', 'year']
    df.sort_values(keys)
    keys.append(lawvar)
    keys.extend(features)
    df = df[keys]
    for i in range(n_lags):
        for f in features:
            f_lag = f + '_t' + str(i + 1)
            df[f_lag] = df.groupby('Circuit')[f].shift(i+1)

    for i in range(n_leads):
        for f in features:
            f_lag = f + '_f' + str(i + 1)
            df[f_lag] = df.groupby('Circuit')[f].shift(-(i+1))

    df.to_csv(lags_leads_file)


generate_lags_and_leads(ols_filter_col)
