import sys

# Importing inbuilt function
import data_processing as dp
import ml_tools as mlt
import dashboard as db
import data_visualization as dv

from dashboard import level, model

# Variable Declaration
features_selected = list()

# Run Parameters
run_level = level.circuityear
model = model.elastic_net
use_dummies = True
use_expectations = True
generate_plots = False

# Text Features Param
text_feature_lag = 1
use_text_features_lag = True


def _filter_data_():
    sys.stdout.write(("\nFiltering data for legal area: " + db.legal_area).ljust(50))
    # dp.read_and_process_vote_level_data()
    sys.stdout.write("--complete\n")


def _handpick_features_from_filtered_data_():
    sys.stdout.write("\nHandpicking imp features from filtered data".ljust(50))
    df = dp.read_filtered_data_into_dataframe()
    dp.handpick_features_from_char_data(df)
    sys.stdout.write("--complete\n")


def _generate_level_files_():
    if not db.use_existing_files:
        df = dp.read_char_with_legal_data()
        sys.stdout.write("\nAggregating Data".ljust(50))
        sys.stdout.write("\nJudge Level".ljust(50))
        df = dp.aggregate_on_judge_level(df)
        sys.stdout.write("--complete" + ' ')
        sys.stdout.write("\nPanel Level".ljust(50))
        dp.aggregate_on_panel_level()
        sys.stdout.write("--complete" + ' ')
        if run_level == level.circuityear:
            sys.stdout.write("\nCircuit Year Level".ljust(50))
            dp.aggregate_on_circuityear_level()
            sys.stdout.write("--complete\n")


def _generate_expectations_at_circuityear_level_():
    if not db.use_existing_files:
        sys.stdout.write("\nGenerating Expectations".ljust(50))
        dp.generate_expectations()
    else:
        sys.stdout.write("\nUsing existing file of Expectations".ljust(50))
        sys.stdout.write("--done\n")


def _clean_data_(df):
    sys.stdout.write("\nCleaning Data".ljust(50))
    df = dp.clean_nan_values(df)
    df = dp.clean_na_values(df)
    sys.stdout.write("--complete\n")
    return df


def _generate_X_():
    X = None
    if not db.use_existing_files:
        sys.stdout.write("\nGenerating X".ljust(50))
        dp.generate_X(run_level=run_level, use_text_features_lag=use_text_features_lag)
        X = dp.read_X(text_feature_lag)
        sys.stdout.write("--complete\n")
    else:
        sys.stdout.write("\nReading X from file".ljust(50))
        X = dp.read_X(text_feature_lag)
        sys.stdout.write("--loaded\n")
    return X


def _run_regression_(X):
    global features_selected

    Z = None
    sys.stdout.write("\nStarting Regression".ljust(50))

    if run_level == level.circuityear:
        Z = dp.read_circuityear_level_data()
    else:
        Z = dp.read_panel_level_data()

    pca_lags = dp.generate_lags_and_leads(n_lags=1)

    Z = dp.level_wise_merge(Z, pca_lags, run_level)

    # Getting Column names for which 1st stage regression would run
    not_included = dp.get_cols_not_included_in_1LS(X=X)
    Xvars = [col for col in list(X) if col not in not_included]

    del Z['govt_wins']
    # Merging X and Z
    merged_X_Z = dp.level_wise_merge(X, Z, run_level)

    # Inititalising with handpicked features in feature selection
    _clean_data_(merged_X_Z)
    # Grid_search_CV(merged_X_Z)
    features_selected = db.ols_filter_col

    # Running feature selection
    if model is not db.model.handpicked:
        features_selected = mlt.run_feature_selection_for_model(merged_df=merged_X_Z,
                                                                Yvars=Xvars,
                                                                run_model=model,
                                                                )
    # Saving the final merged dataframe
    merged_X_Z.to_csv(db.final_merged_file)

    # Running Regression
    if use_text_features_lag:
        mlt.fit_stat_model(merged_X_Z,
                           yvars=Xvars,
                           filter_col=features_selected,
                           includetext_feature_lags=True,
                           text_feature_lag=text_feature_lag)
    else:
        mlt.fit_stat_model(merged_X_Z,
                           yvars=Xvars,
                           filter_col=features_selected,
                           includetext_feature_lags=False)

    sys.stdout.write("--complete\n")


def _run_regression_for_lags_leads_():
    global features_selected
    df = dp.read_lags_leads_data()
    models = {}
    i = 1
    models[0] = mlt.fit_stat_model(df, features_selected)

    lag_features = list()
    for itr in range(db.num_lags):
        for feature in features_selected:
            lag_features.append(feature + '_t' + str(itr + 1))
        df_clean = df.dropna(subset=lag_features)
        models[i] = mlt.fit_stat_model(df_clean, lag_features)
        i += 1
        lag_features = list()

    lead_features = list()
    for itr in range(db.num_leads):
        for feature in features_selected:
            lead_features.append(feature + '_f' + str(itr + 1))
        df_clean = df.dropna(subset=lead_features)
        models[i] = mlt.fit_stat_model(df_clean,
                                       lead_features,
                                       use_expectations=use_expectations,
                                       use_dummies=use_dummies)
        i += 1
        lead_features = list()

    mlt.compare_and_print_statsmodels(models)


def _generate_lags_leads_():
    global features_selected
    dp.generate_lags_and_leads(features_selected, db.num_lags, db.num_leads)


def _generate_plots_():
    if generate_plots:
        sys.stdout.write("\nGenerating Plots".ljust(50))
        df = dp.read_circuityear_level_data()
        df2 = dp.read_expectations_data()
        dv.all_circuit_comparison(expected=df, actual=df)
        sys.stdout.write("--complete\n")
    else:
        sys.stdout.write("\nGenerate Plot value set to".ljust(50))
        sys.stdout.write("--false\n")


def _merge_with_legal_data_():
    df = dp.read_handpicked_features_data_into_dataframe()
    dp.merge_char_with_legal_data(df)


def pipeline():
    _filter_data_()
    _handpick_features_from_filtered_data_()
    _merge_with_legal_data_()
    _generate_level_files_()
    _generate_expectations_at_circuityear_level_()
    X = _generate_X_()
    _run_regression_(X)
    # _generate_lags_leads_()
    # _run_regression_for_lags_leads_()
    _generate_plots_()




if __name__ == "__main__":
    print("\n\n\nStaring the Project Pipeline\n\n")
    pipeline()
