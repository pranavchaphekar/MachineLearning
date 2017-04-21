
import data_processing as dp
import ml_tools as mlt

global df
df = None

def _read_data_():
    print("\nReading & Loading Data")
    dp.read_and_process_vote_level_data()
    df = dp.read_vote_level_data_into_dataframe()

def _clean_data_():
    print("\nCleaning Data")

def _generate_level_files_():
    print("\nAggregating data on Judge Level")
    dp.aggregate_on_judge_level(df)
    print("\nAggregating data on Panel Level")
    dp.aggregate_on_panel_level(df)
    print("\nAggregating data on Circuit Year Level")
    dp.aggregate_on_circuityear_level(df)

def _run_regression_():
    print("")

def _generate_plots_():
    print("")

def _pipeline_():
    print("Do Something")
    _read_data_()
    _clean_data_()
    _run_regression_()
    _generate_plots_()

if __name__ == "__main__":
    print("\n\n\nStaring the Project Pipeline\n\n\n")
    _pipeline_()