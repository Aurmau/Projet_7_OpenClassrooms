from P7_functions.P7_functions_preprocessing import *
from P7_functions.P7_functions_ML import *
from P7_functions.P7_functions_comparison_data import *
from P7_functions.P7_functions_data_for_API import *

# Preprocessing new customer
data_preprocessing(name_file_in="application_test.csv",
                   name_file_out="data_new_customer.csv", path_folder_out="P7_API",
                   sampling=1, return_df=False, export_csv=True,
                   name_data_directory="P7_data", name_python_file="P7_preprocessing.py")

# Preprocessing former customer
data_former_customer = data_preprocessing(name_file_in="application_train.csv",
                                          name_file_out=None, path_folder_out=None,
                                          sampling=1, return_df=True, export_csv=False,
                                          name_data_directory="P7_data", name_python_file="P7_preprocessing.py")
# Calculate comparison data from former customer
data_former_customer = lgbm_scoring_prediction(data_former_customer, name_model_file="model_lgbm.pkl")
data_former_customer = data_for_api(data_former_customer)
comparison_data = calc_comparison_data(data_former_customer,
                                       list_var=["AMT_CREDIT", "DAYS_BIRTH", "ANNUITY_INCOME_PERCENT"])
export_datas(comparison_data, name_file_out="comparison_data.csv", path_folder_out="P7_Dashboard",
             name_python_file="P7_preprocessing.py")