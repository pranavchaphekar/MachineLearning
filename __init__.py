import sys
from enum import Enum

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LassoCV, ElasticNetCV, LogisticRegression

import data_processing as dp
import ml_tools as mlt
import dashboard as db

# Variable Declaration
df = None


class Level:
    judge, panel, circuityear = range(3)

# Run Parameters
run_level = Level.circuityear
run_lasso = False  # chooses handpicked variables if False or lasso chooses the features
run_random_forest = False
run_elastic_net = False
run_Logistic_regression = False


def _read_data_():
    global df
    sys.stdout.write("\nReading & Loading Data".ljust(40))
    # dp.read_and_process_vote_level_data()
    # dp.read_vote_level_data_into_dataframe()
    # df = dp.read_filtered_data_into_dataframe()
    df = dp.read_judge_level_data()
    sys.stdout.write("--complete\n")


def _clean_data_():
    global df
    sys.stdout.write("\nCleaning Data".ljust(40))
    df = dp.clean_nan_values(df)
    df = dp.clean_na_values(df)
    sys.stdout.write("--complete\n")


def _generate_level_files_():
    global df
    sys.stdout.write("\nAggregating Data".ljust(40))
    df = dp.handpick_features_from_char_data(df)
    sys.stdout.write("\nJudge Level".ljust(40))
    dp.aggregate_on_judge_level(df)
    sys.stdout.write("--complete" + ' ')
    if run_level != Level.judge:
        sys.stdout.write("\nPanel Level".ljust(40))
        dp.aggregate_on_panel_level()
        sys.stdout.write("--complete" + ' ')
    elif run_level == Level.circuityear:
        sys.stdout.write("\nCircuit Year Level".ljust(40))
        dp.aggregate_on_circuityear_level()
        sys.stdout.write("--complete\n")


def _run_regression_():
    global df
    models = {}
    sys.stdout.write("\nRunning Regression".ljust(40))
    train, test = dp.split_into_train_and_test(df)
    # mlt.ols_sklearn(train, test)
    features_selected = db.ols_filter_col
    if run_lasso:
        features_selected = mlt.feature_selection(df, LassoCV())
    elif run_random_forest:
        features_selected = mlt.feature_selection(df, ExtraTreesClassifier())
    elif run_elastic_net:
        features_selected = mlt.feature_selection(df, ElasticNetCV())
    elif run_Logistic_regression:
        features_selected = mlt.feature_selection(df, LogisticRegression())
    i = 1
    models[0] = mlt.fit_stat_model(df, features_selected)
    if Level.panel or Level.circuityear:
        df = dp.read_panel_level_data()
        _clean_data_()
        models[i] = mlt.fit_stat_model(df, features_selected)
        i=i+1
    if Level.circuityear:
        df = dp.read_circuityear_level_data()
        _clean_data_()
        models[i] = mlt.fit_stat_model(df, features_selected)
    mlt.compare_and_print_statsmodels(models)
    sys.stdout.write("--complete\n")


def _generate_plots_():
    sys.stdout.write("")


def pipeline():
    _read_data_()
    _clean_data_()
    # _generate_level_files_()
    _run_regression_()
    # _generate_plots_()


if __name__ == "__main__":
    print("\n\n\nStaring the Project Pipeline\n\n")
    pipeline()
