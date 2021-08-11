# import
import numpy as np
import pandas as pd
# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder
# imputer for handling missing values
from sklearn.impute import SimpleImputer
# Polynomial features
from sklearn.preprocessing import PolynomialFeatures
# File system management
import pathlib
import random


# functions for the API

# for preprocessing:
# 1 - load_data()
# 2 - encoding()
# 3 - aligning()
# 4 - feature_engineering()


def load_data(name_file=None, name_file_directory="P7_data", name_python_file="P7_preprocessing.py"):
    """
    load data from an other folder
    :param name_file: (str) name of the csv file
    :param name_file_directory: (str) name of the directory where the csv file is
    :param name_python_file: (str) name of the current python file
    :return: dataframe of the csv file
    """

    # Get datas directory path
    path = str(pathlib.Path(name_python_file).parent.resolve())
    path_datas = path + "\\" + name_file_directory
    # get datas
    datas = pd.read_csv(path_datas + "\\" + name_file)
    return datas


def encoding(df):
    """
    encoding a dataframe with a label encore if 2 labels or with one hot encoding if more than 2 labels
    :param df: dataframe
    :return: encoded dataframe
    """

    # Create a label encoder object
    le = LabelEncoder()

    # Iterate through the columns
    for col in df:
        if df[col].dtype == 'object':
            # If 2 or fewer unique categories
            if len(list(df[col].unique())) <= 2 and col != "CODE_GENDER":
                # Train
                le.fit(df[col])
                # Transform
                df[col] = le.transform(df[col])

    # one-hot encoding of categorical variables
    df = pd.get_dummies(df)
    return df


# Drop columns not in same dataframe
def aligning(df):
    """
    Drop the columns ['CODE_GENDER_XNA', 'NAME_FAMILY_STATUS_Unknown', 'NAME_INCOME_TYPE_Maternity leave'] of a
    dataframe
    :param df: dataframe
    :return: dataframe without ['CODE_GENDER_XNA', 'NAME_FAMILY_STATUS_Unknown', 'NAME_INCOME_TYPE_Maternity leave']
    """
    try:
        df.drop(columns=['CODE_GENDER_XNA', 'NAME_FAMILY_STATUS_Unknown', 'NAME_INCOME_TYPE_Maternity leave'],
                inplace=True)
    except:
        pass
    return df


def feature_engineering(df):
    """
    Feature engineering on the dataframes used in this project
    :param df: dataframe
    :return: dataframe after feature engineering
    """

    # Create an anomalous flag column
    df['DAYS_EMPLOYED_ANOM'] = df["DAYS_EMPLOYED"] == 365243
    # Replace the anomalous values with nan
    df['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)

    # Replace XNA
    df['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)

    # Days positive
    df['DAYS_BIRTH'] = abs(df['DAYS_BIRTH'])
    df['DAYS_EMPLOYED'] = abs(df['DAYS_EMPLOYED'])

    # FEATURE ENGINEERING

    # Polynomial features
    # Make a new dataframe for polynomial features
    try:
        poly_features = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'TARGET']]
        poly_target = poly_features['TARGET']
        poly_features = poly_features.drop(columns=['TARGET'])
    except KeyError:
        poly_features = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]

    # imputer for handling missing values
    imputer = SimpleImputer(strategy='median')

    # Need to impute missing values
    poly_features = imputer.fit_transform(poly_features)

    # Create the polynomial object with specified degree
    poly_transformer = PolynomialFeatures(degree=3)

    # Train the polynomial features
    poly_transformer.fit(poly_features)

    # Transform the features
    poly_features = poly_transformer.transform(poly_features)

    # Create a dataframe of the features
    poly_features = pd.DataFrame(poly_features,
                                 columns=poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2',
                                                                             'EXT_SOURCE_3', 'DAYS_BIRTH']))

    # Add in the target
    try:
        poly_features['TARGET'] = poly_target
    except:
        pass

    # Merge polynomial features into training dataframe
    poly_features['SK_ID_CURR'] = df['SK_ID_CURR']
    df_poly = df.merge(poly_features, on='SK_ID_CURR', how='left', suffixes=('', '_todrop'))
    df_poly.drop(df_poly.filter(regex='_todrop$').columns.tolist(), axis=1, inplace=True)

    # # train target
    # df_targets = df_poly['TARGET']
    #
    # # Align the dataframes
    # # app_train_poly, app_test_poly = df_poly.align(app_test_poly, join='inner', axis=1)
    #
    # # Add the target back in
    # df_poly['TARGET'] = df_targets

    # Domain knowledge features
    df_poly_domain = df_poly.copy()

    df_poly_domain['CREDIT_INCOME_PERCENT'] = df_poly_domain['AMT_CREDIT'] / df_poly_domain['AMT_INCOME_TOTAL']
    df_poly_domain['ANNUITY_INCOME_PERCENT'] = df_poly_domain['AMT_ANNUITY'] / df_poly_domain['AMT_INCOME_TOTAL']
    df_poly_domain['CREDIT_TERM'] = df_poly_domain['AMT_CREDIT'] / df_poly_domain['AMT_ANNUITY']
    df_poly_domain['DAYS_EMPLOYED_PERCENT'] = df_poly_domain['DAYS_EMPLOYED'] / df_poly_domain['DAYS_BIRTH']

    return df_poly_domain


def export_datas(df, name_python_file="P7_preprocessing.py", name_file_out=None, path_folder_out=None):
    """
    Export a dataframe into a csv file
    :param df: dataframe to export
    :param name_python_file: (str) name of the current python file
    :param name_file_out: (str) name of the csv file to export
    :param path_folder_out: (str) path to the folder to export the csv file
    :return: no return
    """
    if path_folder_out is None:
        df.to_csv(name_file_out)
    else:
        path_python_file = str(pathlib.Path(name_python_file).parent.resolve())
        df.to_csv(path_python_file + "\\" + path_folder_out + "\\" + name_file_out)


def data_preprocessing(name_file_in=None, name_file_out=None, path_folder_out=None, sampling=1, return_df=False,
                       export_csv=True, name_data_directory="P7_data", name_python_file="P7_preprocessing.py"):
    """
    Apply all the previous functions
    :param name_file_in: (str) name of the csv file to load
    :param name_file_out: (str) name of the csv file to export
    :param path_folder_out: (str) path to the folder to export csv file
    :param sampling: (float between 0 and 1) select a random sample of data
    :param return_df: if True return a dataframe of the results
    :param export_csv: if True export a csv file
    :param name_data_directory: (str) name of the directory where the csv file to upload is
    :param name_python_file: (str) name of the current python file
    :return: dataframe of the results if return_df is True
    """

    df = load_data(name_file=name_file_in, name_file_directory=name_data_directory, name_python_file=name_python_file)
    df = encoding(df)
    df = aligning(df)
    if 1 > sampling > 0:
        df = df.sample(frac=sampling)
    df = feature_engineering(df)
    if export_csv is True:
        export_datas(df, name_file_out=name_file_out, path_folder_out=path_folder_out,
                     name_python_file=name_python_file)
    if return_df is True:
        return df
