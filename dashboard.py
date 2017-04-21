# List of Global Variables

characteristic_data_path = 'data/BloombergVOTELEVEL_Touse.dta'
filtered_char_data_path = 'filtered_characteristics.csv'
handpicked_char_data_path = 'handpicked_characteristics.csv'
char_with_legal_data = 'char_with_legal_data.csv'
case_id_column = 'caseid'
legal_area = "evironment"
lawvar = "govt_wins"
features_to_use = ['caseid', 'year', 'Circuit', 'month', 'govt_wins', 'songername', 'x_dem', 'x_republican',
                       'x_instate_ba', 'x_aba', 'x_protestant', 'x_evangelical', 'x_noreligion', 'x_catholic',
                       'x_jewish', 'x_black', 'x_nonwhite', 'x_female']
judge_level_file = 'data/judge_level_agg.csv'
panel_level_file = 'data/panel_level_agg.csv'
circuityear_level_file = 'data/circuityear_level_agg.csv'
panel_level_grouping_columns = ['caseid', 'Circuit', 'year', 'month', 'govt_wins']
