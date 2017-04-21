import pickle
import pandas as pd
import sys
import numpy as np
from dashboard import *


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


def read_vote_level_data_into_dataframe():
    reader = pd.read_stata(characteristic_data_path, iterator=True)
    # reader = pd.read_csv('data/filtered.csv',iterator=True)
    df = pd.DataFrame()

    try:
        chunk = reader.get_chunk(1000)
        ctr = 1
        while len(chunk) > 0:
            sys.stdout.write(str(ctr) + ' ')
            df = df.append(chunk, ignore_index=True)
            sys.stdout.flush()
            ctr += 1
            chunk = reader.get_chunk(1000)
    except (StopIteration, KeyboardInterrupt):
        pass
    return df

def read_filtered_data_into_dataframe():
    df = pd.read_csv(filtered_char_data_path, low_memory=False)  # load into the data frame
    return df

def group_and_aggregate():
    df = read_vote_level_data_into_dataframe()

    features_to_be_used_for_expectation = ['x_dem', 'x_republican',
                                           'x_instate_ba',
                                           'x_aba', 'x_protestant', 'x_evangelical', 'x_noreligion', 'x_catholic',
                                           'x_jewish',
                                           'x_black', 'x_nonwhite',
                                           'x_female']  # keep only the limited set of variables (handpicked ones)

    features_after_adding_e = []

    # Adding new columns to the dataframe
    for feature in features_to_be_used_for_expectation:
        expected_feature = 'e_' + feature
        features_after_adding_e.append(expected_feature)
        df[expected_feature] = df[feature]  # initialise that with the feature

    df = df[np.isfinite(df['year'])]
    df = df.fillna(0)

    df1 = df.groupby(['Circuit', 'year'], as_index=False)[features_after_adding_e].mean()

    df2 = pd.merge(df1, df, on=['Circuit', 'year'])

    # make the new dataframe of environmental cases only
    df2.to_csv('test.csv')

    return df2


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


def handpick_features_from_char_data(df):
    """

    :return:
    """
    return df[features_to_use]


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


def aggregate_on_judge_level(df):
    '''
    Aggregates data on Judge Level.Each case has 3 rows.
    Used to generate panel level files
    '''
    df = pd.merge(df, read_case_ids(), on='caseid')
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

    df = df.groupby(["Circuit", "year"]).agg(f)
    # df = df.sort_values(sort_order)

    # Adding a NewColumn for Clustering CircuitXYear
    # df['circuitXyear'] = df.Circuit.astype(str).str.cat(df.year.astype(str), sep='X')

    df.to_csv(circuityear_level_file)


# This function splits the file into test and train data
def split_into_train_and_test(df):
    msk = np.random.rand(len(df)) < train_test_split
    train = df[msk]
    test = df[~msk]
    return train, test
