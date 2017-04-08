import pickle
import pandas as pd
import sys
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
    df = pickle.load(open('govt_winner.pkl', 'rb'))
    environ_cases = df[df['govt_environ'] == 1]
    environ_cases = environ_cases[['caseid', 'govt_wins']]
    return list(environ_cases['caseid'])


# read_environmental_law_indicator()
#read_and_process_vote_level_data(read_environmental_law_indicator())



def lvl_circuityear():
    df = pd.read_csv("filtered.csv",low_memory=False)
    print(list(df.columns.values))
    #df.loc["panelvote" in (2,3),"proplaintiff"] =1
    #df.loc["panelvote" in (0, 1), "proplaintiff"] = 0
    #df.loc["protaking" is 1, "proplaintiff"] = 0
    #df.loc["protaking" is 0, "proplaintiff"] = 1
    #gen numCasesPro = ${lawvar}
    #gen numCasesAnti = 1-${lawvar}
    #encode case_ID, generate(numCases)
    #rename nr_Judges numJudges

lvl_circuityear()