# List of Global Variables

# Main Pipeline Variables
case_id_column = 'caseid'
legal_area = "evironment"
lawvar = 'govt_wins'
case_type = 'govt_environ'
num_lags = 3
num_leads = 2
run_high_dimensional = True
use_existing_files = True

# Feature and Grouping Filters
features_to_use = ['caseid', 'year', 'Circuit', 'month', 'songername', 'x_dem', 'x_republican',
                   'x_instate_ba', 'x_aba', 'x_protestant', 'x_evangelical', 'x_noreligion', 'x_catholic',
                   'x_jewish', 'x_black', 'x_nonwhite', 'x_female', 'dissentvote']
ols_filter_col = ['x_dem', 'x_nonwhite', 'x_noreligion']  # Hand picked features
panel_level_grouping_columns = ['caseid', 'Circuit', 'year', 'month', 'govt_wins']

# Regression Variables
train_test_split = 0.8
epectations_generated = True
pca_components = 10  # can be a number or n_components == min(n_samples, n_features) if None or can be 'mle'

# Input Files
characteristic_data_path = 'data/BloombergVOTELEVEL_Touse.dta'
filtered_char_data_path = 'data/filtered_characteristics.csv'
handpicked_char_data_path = 'data/handpicked_characteristics.csv'
char_with_legal_data = 'data/char_with_legal_data.csv'
lawvar_caseid_decision_file = 'data/govt_winner.pkl'  # Maps Case with boolean(0-1) decision of Lawvar Cases
generated_circuityear_expectations_file = 'data/concat1.csv'
lags_leads_file = 'data/selected_features_with_lags_leads.csv'
text_feature_files_dir = 'data/cleaned/textfeatures'
text_features_lvl_panel = 'data/text_features_lvl_panel.csv'
text_features_lvl_circuityear = 'data/text_features_lvl_circuityear.csv'

# Output Files
judge_level_file = 'data/judge_level_agg.csv'
panel_level_file = 'data/panel_level_agg.csv'
circuityear_level_file = 'data/circuityear_level_agg.csv'
text_features_file = 'data/text_features.csv'
char_with_text_features = 'data/char_with_text_features.csv'
X_file = 'data/X.csv'
X_with_lags_leads_file = 'data/X_lags_leads.csv'
final_merged_file = 'data/final_merged_file.csv'

# Graphs
all_circuit_actual_expected_comparison = {'Democrat': 'x_dem'}


class level:
    judge, panel, circuityear = range(3)


class model:
    handpicked, lasso, elastic_net, random_forest, logistic = range(5)
