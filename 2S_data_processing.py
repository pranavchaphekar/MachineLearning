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

def rotate_around_pivot():
    df = pd.read_csv('pollutants.csv')
    # tuples = list(zip(df.columns, df.T.values.tolist()))
    filter_col = [col for col in list(df) if col.startswith('emissions')]
    df_intermediate = pd.melt(df.reset_index(), id_vars=['Circuit', 'pollutant_code'], value_vars=['emissions90', 'emissions96'])
    df_final = pd.pivot_table(df_intermediate,index=['Circuit', 'pollutant_code'], columns=[filter_col])
    df_final.to_csv('intermediate.csv')
    return df_final

# df = read_pollution_data()
# df = aggregate_data_by_pollutants(df)
# df = map_state_to_circuit_no(df)
# df = group_by_circuit(df)
df = rotate_around_pivot()
# df.to_csv('pollutants1.csv')