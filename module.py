import pickle
import pandas as pd
import sys
import numpy as np
from dashboard import *

def read_and_process_vote_level_data(case_ids):
    reader = pd.read_stata('BloombergVOTELEVEL_Touse.dta', iterator=True)
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
    df = pickle.load(open('govt_winner.pkl ', 'rb'))
    environ_cases = df[df['govt_environ'] == 1]
    environ_cases = environ_cases[['caseid', 'govt_wins']]
    return environ_cases

def cleaned_CSV():
    df = pd.read_csv('filtered.csv',low_memory=False)
    #read the handpicked attributes from a file into a list
    features = []
    with open("handpicked_features.txt") as file:
        for line in file:
            line = line.strip() #for removing the spaces
            features.append(line)
    df = df.drop_duplicates() #drop the duplicate rows
    lines_subset = df[features]
    lines_subset.to_csv('filtered_subset.csv')

def add_X_col(): #adds the govt wins col.
    df = pd.read_csv('filtered_subset.csv')
    df = pd.merge(df,read_environmental_law_indicator(),on='caseid')
    df.to_csv('add_X_col.csv')

# A function that creates the interactions
def gen_inter(main_df, df1, df2,df1_col_name, df2_col_name):
    name_of_dataframe = df1_col_name + ' X ' + df2_col_name
    main_df[name_of_dataframe] = df1 * df2
    return main_df


def lvl_judge():
    df = pd.read_csv('filtered.csv',low_memory=False) #load into the data frame
    features_to_be_kept = ['caseid','year','Circuit','month','songername','x_dem','x_republican','x_instate_ba','x_aba','x_protestant','x_evangelical','x_noreligion','x_catholic','x_jewish','x_black','x_nonwhite'
,'x_female'] #keep only the limited set of variables (handpicked ones)
    df_lvl_judge = df[features_to_be_kept] #creates a new data frame with only few handpicked features
    interaction_list = ['x_dem','x_republican','x_instate_ba','x_aba','x_protestant','x_evangelical','x_noreligion','x_catholic','x_jewish','x_black','x_nonwhite'
,'x_female'] #only the variables chosen for cross product
    non_interaction_list = ['caseid','year','Circuit','month','songername']
    df_subset_non_interactions = df_lvl_judge[non_interaction_list]
    df_subset_dataframe = df_lvl_judge[interaction_list]
    #Replace Nan with 0
    df_subset_dataframe = df_subset_dataframe.replace(np.nan, 0)
    #Merge the two data frames
    for feature in interaction_list:
        interaction_list_subset = interaction_list
        interaction_list_subset.remove(feature)
        for other_feature in interaction_list_subset:
            df1 = df_lvl_judge[feature]
            df2 = df_lvl_judge[other_feature]
            df_subset_dataframe = gen_inter(df_subset_dataframe,df1,df2,feature,other_feature)



    result = pd.concat([df_subset_non_interactions, df_subset_dataframe], axis=1)
    result.to_csv('result.csv')

    #df1 = PolynomialFeatures(2, interaction_only=True, include_bias=False).fit_transform(df_subset_dataframe)
    #Order of elements, Sorting
    sort_order = ['Circuit','year','month']
    #Sorting by the column enteries and store that in result dataframe
    #result = df1.sort_values(['Circuit'])


# read_environmental_law_indicator()
#read_and_process_vote_level_data(read_environmental_law_indicator())
#cleaned_CSV()
#add_X_col()
lvl_judge()


def lvl_circuityear():
    df = pd.read_csv("filtered.csv",low_memory=False)
    print(list(df.columns.values))

    X_star = []
    E_star = []
    for col in list(df.columns.values):
        if str(col).lower().startswith('x_'):
            X_star.append(str(col))
        elif str(col).lower().startswith('e_'):
            E_star.append(str(col))

    ##PANELVOTE & PROTAKING Columns Currently not There!!!!!!
    #df[df.panelvote in (2,3)]['proplaintiff'] = 1
    #df[df.protaking == 1]['proplaintiff'] = 0
    #df[df.protaking == 0]['proplaintiff'] = 1

    print(list(df.columns.values))

    #Generating and renaming variables so that they have appropriate names after collapsing gen
    df.rename(columns={lawvar:'numCasesPro','caseid' :'numCases'}, inplace=True)

    df['numCasesAnti'] = 1 - df[lawvar]

    sort_order = ['Circuit', 'year']
    # Sorting by the column enteries and store that in result dataframe
    df = df.sort_values(sort_order)
    df.fillna(df.mean())
    #df[df.numCases == 0]['present'] = 1

    # Define a lambda function to compute the weighted mean:
    wMean = lambda x: np.average(x)
    f = {'numCases': ['count'],'numJudges':['sum'],'numCasesPro':['sum'],'numCasesAnti':['sum']}
    df.groupby(["Circuit", "year"]).agg(f)

    '''
    qui collapse(count) numCases(sum) numJudges numCasesPro numCasesAnti(mean) x_ * E_ * proplaintiff protaking, by(circuit,year)

    df.groupby("subid", sort=True).sum()

    # Define a lambda function to compute the weighted mean:
    wMean = lambda x: np.average(x)

    # Define a dictionary with the functions to apply for a given column:
    f = {'numCases': ['count'],'numJudges':['sum'],'numCasesPro':['sum'],'numCasesAnti':['sum']}

    # Groupby and aggregate with your dictionary:
    df.groupby(["contract", "month", "year", "buys"]).agg(f)
    '''

lvl_circuityear()