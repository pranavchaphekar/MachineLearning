import pandas as pd
import numpy as np

######################
# Read Data Methods
######################

#Read polution Data
def read_pollution_data():
    df = pd.read_excel("data/state_tier1_90-16.xls")
    return df

def aggregate_data_by_pollutants(df):

    meanFun = lambda x: np.average(x)
    filter_col = [col for col in list(df) if col.startswith('emissions')]

    f = {}

    for col in filter_col:
        f[col] = meanFun
    df = df.fillna(value=0)
    df = df.groupby(['STATE_ABBR','pollutant_code'], as_index=False).agg(f)
    return df

def map_state_to_circuit_no(df):
    df_abbr = pd.read_csv('data/abbr.csv')
    # df_abbrdict = df_abbr.set_index('Abbreviation').T.to_dict()
    # df_abbrdict = df.set_index('Abbreviation')['value'].to_dict()
    df_abbrdict = dict(zip(df_abbr['Abbreviation'], df_abbr['Circuit']))
    df['Circuit'] = df['STATE_ABBR'].map(df_abbrdict)

    del df['STATE_ABBR']
    return df

def group_by_circuit(df):
    meanFun = lambda x: np.average(x)
    filter_col = [col for col in list(df) if col.startswith('emissions')]
    f = {}

    for col in filter_col:
        f[col] = meanFun

    df = df.groupby(['Circuit', 'pollutant_code'], as_index=False).agg(f)
    return df


def create_output_data_cy_level():
    df = pd.read_csv('data/pollutants.csv')
    df_new = pd.wide_to_long(df, ['emissions'], i='Circuit', j='year')
    wtl = df_new.pivot(columns='pollutant_code')['emissions']
    wtl.to_csv('data/pollutants_final.csv')


def fix_year_values():
    df = pd.read_csv('data/pollutants_final.csv')
    df['year'] = pd.Series([(1900+val) if (val >= 90) else (2000+val) for val in df['year']])
    df.sort_values(['Circuit', 'year'])
    df.to_csv('data/pollutants_final.csv', index=False)


def read_final_pollution_data():
    df = pd.read_csv('data/pollutants_final.csv')
    return df


#######################
# Lags and Leads
#######################
def use_lag_or_lead(df, y_cols, n_lags=0, n_leads=0):
    keys = ['Circuit', 'year']
    df.sort_values(keys, inplace=True)
    if n_lags > 0:
        for f in y_cols:
            df[f] = df.groupby('Circuit')[f].shift(n_lags)
    elif n_leads > 0:
        for f in y_cols:
            df[f] = df.groupby('Circuit')[f].shift(-n_leads)

    return df

# df = read_pollution_data()
# df = aggregate_data_by_pollutants(df)
# df = map_state_to_circuit_no(df)
# df = group_by_circuit(df)
# df.to_csv('pollutants.csv')
# create_output_data_cy_level()
# fix_year_values()
