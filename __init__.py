import sys

import data_processing as dp
import ml_tools as mlt

df = None


def _read_data_():
    sys.stdout.write("\nReading & Loading Data".ljust(40))
    # dp.read_and_process_vote_level_data()
    # dp.read_vote_level_data_into_dataframe()
    global df
    # df = dp.read_filtered_data_into_dataframe()
    df = dp.read_panel_level_data()
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
    sys.stdout.write("\nPanel Level".ljust(40))
    dp.aggregate_on_panel_level()
    sys.stdout.write("--complete" + ' ')
    sys.stdout.write("\nCircuit Year Level".ljust(40))
    dp.aggregate_on_circuityear_level()
    sys.stdout.write("--complete\n")


def _run_regression_():
    global df
    models = []
    sys.stdout.write("\nRunning Regression".ljust(40))
    train, test = dp.split_into_train_and_test(df)
    # mlt.ols_sklearn(train, test)
    features_selected = mlt.lasso_for_feature_selection(df)
    mlt.fit_stat_model(df, features_selected)
    # models.append(mlt.fit_stat_model(train, test))
    # compare_and_print_statsmodels()
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
