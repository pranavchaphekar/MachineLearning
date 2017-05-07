import os
import pickle
from collections import Counter
from glob import glob
from zipfile import ZipFile

import pandas as pd
import sys
import numpy as np
import dashboard as db
from ml_tools import pca_on_text_features, pls_regression_on_text_features


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
    reader = pd.read_stata(db.characteristic_data_path, iterator=True)
    df = pd.DataFrame()

    try:
        chunk = reader.get_chunk(1000)
        ctr = 1
        sys.stdout.write("Loading Chunk : " + ' ')
        while len(chunk) > 0:
            chunk = chunk[chunk[db.case_id_column].isin(case_ids)]
            df = df.append(chunk, ignore_index=True)
            sys.stdout.write(str(ctr) + ' ')
            sys.stdout.flush()
            ctr += 1
            chunk = reader.get_chunk(1000)
    except (StopIteration, KeyboardInterrupt):
        pass
    df.to_csv(db.filtered_char_data_path)


def read_filtered_data_into_dataframe():
    df = pd.read_csv(db.filtered_char_data_path, low_memory=False)  # load into the data frame
    return df


def read_handpicked_features_data_into_dataframe():
    df = pd.read_csv(db.handpicked_char_data_path, low_memory=False)  # load into the data frame
    return df


def read_char_with_legal_data():
    df = pd.read_csv(db.char_with_legal_data, low_memory=False)  # load into the data frame
    return df


def read_case_ids():
    """
    Reads the case ids of the cases with lawvars
    and returns the list
    :return:
    """
    df = pickle.load(open(db.lawvar_caseid_decision_file, 'rb'))
    cases = df[df[db.case_type] == 1]
    cases = cases[['caseid', db.lawvar]]
    cases = cases.drop_duplicates()
    return cases


######################
# Clean Data Methods
######################
def clean_nan_values(df):
    return df.replace(np.nan, 0)


def clean_na_values(df):
    # return df.fillna(0)
    return df.dropna()


def handpick_features_from_char_data(df):
    """

    :return:
    """
    df[db.features_to_use].to_csv(db.handpicked_char_data_path)
    return df[db.features_to_use]


######################
# Merge Data Methods
######################

def merge_with_legal_area_data(df, legal_area_data):
    """

    :return:
    """
    # df = pd.read_csv('filtered_subset.csv')
    df = pd.merge(df, legal_area_data, on='caseid', index=False)
    df.to_csv(db.char_with_legal_data)
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
    df = df.drop_duplicates()
    df.to_csv(db.char_with_legal_data)
    return df


def merge_expectations_with_lvl_circuit(df2):
    df1 = pd.read_csv(db.generated_circuityear_expectations_file)
    df = pd.merge(df1, df2, on=['Circuit', 'year'], how="inner")

    # drops the duplicate _y cols
    to_drop = [x for x in df if x.endswith('_y')]
    df.drop(to_drop, axis=1, inplace=True)
    # trim _x
    for col in df:
        if col.endswith('_x'):
            df.rename(columns={col: col.rstrip('_x')}, inplace=True)

    # del df['Unnamed: 0']
    df = df.sort_values(['Circuit', 'year'])
    return df


def merge_expectations_with_lvl_panel(df2):
    df1 = pd.read_csv(db.generated_circuityear_expectations_file)
    df = pd.merge(df2, df1, on=['Circuit', 'year'])
    del df['Unnamed: 0']
    df = df.sort_values(['Circuit', 'year'])
    return df


def merge_with_dummies(df):
    df_dummies = pd.get_dummies(df['Circuit'], prefix="dummy")
    df = pd.concat([df, df_dummies], axis=1)
    df_dummies = pd.get_dummies(df['year'], prefix="dummy")
    df = pd.concat([df, df_dummies], axis=1)
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
    result.to_csv(db.judge_level_file)
    return result


def aggregate_on_panel_level():
    df = pd.read_csv(db.judge_level_file, low_memory=False)  # load into the data frame
    filter_col = [col for col in list(df) if col.startswith('x_')]
    df.insert(5, 'num_judges', 1)
    countFun = lambda x: len(x)
    meanFun = lambda x: np.average(x)
    f = dict()
    f['num_judges'] = countFun
    for col in filter_col:
        f[col] = meanFun
    if db.run_high_dimensional:
        high_dem_col = [col for col in list(df) if col.startswith('pca_')]
        for col in high_dem_col:
            f[col] = meanFun
    grouped = df.groupby(db.panel_level_grouping_columns, as_index=False).agg(f)

    grouped = merge_expectations_with_lvl_panel(grouped)

    grouped = merge_with_dummies(grouped)

    grouped.to_csv(db.panel_level_file)


def aggregate_on_circuityear_level():
    df = pd.read_csv(db.panel_level_file, low_memory=False)

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

    df['numCasesPro'] = df[db.lawvar]
    df['numCases'] = 1
    df['numCasesAnti'] = 1 - df[db.lawvar]

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
    f[db.lawvar] = meanFun
    for col in X_star:
        f[col] = meanFun

    # if run_high_dimensional:
    #     high_dem_col = [col for col in list(df) if col.startswith('pca_')]
    #     for col in high_dem_col:
    #         f[col] = meanFun

    df = df.groupby(["Circuit", "year"], as_index=False).agg(f)
    # df = df.sort_values(sort_order)

    # Add the expectations
    df = merge_expectations_with_lvl_circuit(df)

    # Merge with Dummies
    df = merge_with_dummies(df)

    df.to_csv(db.circuityear_level_file)


def read_judge_level_data():
    df = pd.read_csv(db.judge_level_file, low_memory=False)  # load into the data frame
    return df


def read_panel_level_data():
    df = pd.read_csv(db.panel_level_file, low_memory=False)  # load into the data frame
    return df


def read_circuityear_level_data():
    df = pd.read_csv(db.circuityear_level_file, low_memory=False)  # load into the data frame
    return df


def read_expectations_data():
    df = pd.read_csv(db.generated_circuityear_expectations_file, low_memory=False)  # load into the data frame
    return df


def read_lags_leads_data():
    df = pd.read_csv(db.lags_leads_file, low_memory=False)
    return df


# This function splits the file into test and train data
def split_into_train_and_test(df):
    msk = np.random.rand(len(df)) < db.train_test_split
    train = df[msk]
    test = df[~msk]
    return train, test


#######################
# Epectations
#######################
def generate_expectations():
    # Read the environmental data and keep only 2 columns which will be useful further.
    df_environmental = pd.read_csv(db.characteristic_data_path)
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
    df_environmental = pd.read_csv(db.filtered_char_data_path)
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
    keys.append(db.lawvar)
    keys.extend(features)
    df = df[keys]
    for i in range(n_lags):
        for f in features:
            f_lag = f + '_t' + str(i + 1)
            df[f_lag] = df.groupby('Circuit')[f].shift(i + 1)

    for i in range(n_leads):
        for f in features:
            f_lag = f + '_f' + str(i + 1)
            df[f_lag] = df.groupby('Circuit')[f].shift(-(i + 1))

    df.to_csv(db.lags_leads_file)


# generate_lags_and_leads(ols_filter_col)

#######################
# Text Features
#######################

def generate_text_features_for_lawvar_cases():
    '''
    Assumes the text features are present in year wise zip files
    containing case number wise pickle files.
    :return: data frame containing
    '''
    if not db.use_existing_files:
        zipfiles = glob('./' + db.text_feature_files_dir + '/*zip')
        lawvar_case_ids = read_case_ids()
        lawvar_case_ids = lawvar_case_ids.caseid.unique()
        text_df = pd.DataFrame(index=lawvar_case_ids)
        # text_df['caseid'] = lawvar_case_ids['caseid']
        lawvar_case_ids = set(lawvar_case_ids)
        print(lawvar_case_ids)
        for zfname in zipfiles:
            zfile = ZipFile(zfname)
            year = zfname.split('/')[-1][:-4]
            members = zfile.namelist()
            # threshold = len(members) / 200
            # docfreqs = Counter()

            for fname in members:
                if not fname.endswith('-maj.p'):
                    continue
                docid = fname.split('/')[-1][:-6]
                if docid in lawvar_case_ids:
                    text = pickle.load(zfile.open(fname))
                    for citation, num_citation in text.items():
                        if citation not in text_df:
                            text_df[citation] = 0
                        row = text_df[text_df.index == docid].index
                        text_df.set_value(row, citation, num_citation)
        text_df.to_csv(db.text_features_file)
    else:
        text_df = pd.read_csv(db.text_features_file, low_memory=False, index_col=0)
    return text_df


def agg_text_features_on_circuityear_level(df):
    df['caseid'] = read_case_ids().caseid.unique()
    cols = [col for col in list(df.columns) if col not in ['Circuit', 'year', 'caseid']]
    sumFun = lambda x: x.sum()
    f = {}
    for col in cols:
        f[col] = sumFun
    panel_data = read_panel_level_data()
    df = pd.merge(df, panel_data[['Circuit', 'year', 'caseid']], on='caseid')
    df.to_csv('data/abc2.csv')
    df = df.groupby(["Circuit", "year"], as_index=False).agg(f)
    df.to_csv('data/abc.csv')
    return df


def generate_pca_of_text_features(level, text_features):
    panel_data = read_panel_level_data()
    # text_features = pd.read_csv(text_features_file, low_memory=False, index_col=0)
    df_comb = None
    if level is db.level.circuityear:
        df_comb = text_features[['Circuit', 'year']]
    else:
        df_comb = text_features['caseid']

    pca_comp = pca_on_text_features(
        text_features[[col for col in list(text_features.columns) if col not in ['Circuit', 'year', 'caseid']]])
    #pca_comp = pca_comp.transpose()
    col_names = []
    for i in range(pca_comp.shape[1]):
        col_names.append('pca_' + str(i))

    pca_comp = pd.DataFrame(pca_comp)
    pca_comp.columns = col_names
    if level is db.level.circuityear:
        pca_comp['Circuit'] = df_comb['Circuit']
        pca_comp['year'] = df_comb['year']
        pca_comp.to_csv(db.text_features_lvl_circuityear)
    else:
        pca_comp['caseid'] = df_comb['caseid']
        pca_comp.to_csv(db.text_features_lvl_panel)

    return pca_comp


def generate_lags_and_leads_text_features(df, n_lags=1):
    features = [col for col in list(df.columns) if col not in ['Circuit', 'year', 'caseid', db.lawvar]]
    keys = ['Circuit', 'year']
    df.sort_values(keys)
    keys.append(db.lawvar)
    keys.extend(features)
    df = df[keys]

    for i in range(n_lags):
        for f in features:
            f_lag = f + '_t' + str(i + 1)
            df[f_lag] = df.groupby('Circuit')[f].shift(i + 1)

    df.to_csv(db.X_with_lags_leads_file)
    return df


def read_X(text_feature_lag):
    df = pd.read_csv(db.X_with_lags_leads_file, low_memory=False)
    df = df[df.columns[~df.columns.str.contains('Unnamed:')]]
    x_features_to_include = []
    x_features_to_include.extend([col for col in list(df.columns) if '_t' not in col])
    x_features_to_include.extend([col for col in list(df.columns) if col.startswith('pca_') and col.endswith('_t'+str(text_feature_lag))])
    df =df[x_features_to_include]
    df =df.dropna(how='any')
    return df


def level_wise_merge(df1, df2, level):
    merged = None
    if level is db.level.circuityear:
        merged = pd.merge(df1, df2, on=['Circuit', 'year'], how='inner')
    else:
        merged = pd.merge(df1, df2, on='caseid', how='inner')
    merged = merged[merged.columns[~merged.columns.str.contains('Unnamed:')]]
    return merged


def merge_text_features(df, level):
    '''
    Merges text features with a data frame on the basis
    of a case id.
    :return: dataframe conatining merged features
    '''
    merged = None

    text_features = pd.read_csv(db.text_features_lvl_panel, low_memory=False, index_col=0)

    if level is level.circuityear:
        text_features = pd.read_csv(db.text_features_lvl_circuityear, low_memory=False, index_col=0)

    merged = level_wise_merge(text_features, df, level)

    return merged


def level_wise_lawvar(level):
    df = read_case_ids()
    if level is db.level.circuityear:
        sumFun = lambda x: np.sum(x)
        f = dict()
        cols = [col for col in list(df)]
        f[db.lawvar] = sumFun
        # for col in cols:
        #    f[col] = meanFun
        data = read_panel_level_data()
        merged = level_wise_merge(data[['Circuit', 'year', 'caseid']], df, db.level.panel)
        del merged['caseid']
        df = merged.groupby(['Circuit', 'year'], as_index=False).agg(f)
    return df


def get_lags_features(df, lag_year=1):
    i = 0
    lag_features = list()
    for feature in [col for col in list(df.columns) if '_t' + str(lag_year) in col]:
        # lag_features.append(feature + '_t' + str(lag_year + 1))
        # df_clean = df.dropna(subset=lag_features)
        lag_features.append(feature)
        i += 1
    return lag_features


def generate_X(run_level, use_text_features_lag):
    X = level_wise_lawvar(run_level)
    if db.run_high_dimensional:
        df = generate_text_features_for_lawvar_cases()
        df = agg_text_features_on_circuityear_level(df)
        df = generate_pca_of_text_features(run_level, text_features=df)
        X = level_wise_merge(X, df, run_level)
        if use_text_features_lag:
            X = generate_lags_and_leads_text_features(X, n_lags=db.num_lags)
    X.to_csv(db.X_file)
    X = X.dropna(how='any')
    return X


def get_cols_not_included_in_1LS(X):
    not_included = ['Circuit', 'year', 'caseid']
    for lag_yr in range(1, db.num_lags + 1):
        not_included.extend(get_lags_features(X, lag_yr))
    return not_included

    # text_features_for_lawvar_cases()


def get_expectation_col_names_for_features(features):
    expectations = set()
    for col in features:
        if col.find("X") is -1:  # if not an interaction, interaction format a 'X' b
            expec_col = "e_" + col
            expectations.add(expec_col)
        elif col.find("X") >= 0:
            expectations.add('e_' + col.split('X')[0].strip())
            expectations.add('e_' + col.split('X')[1].strip())
    return list(expectations)