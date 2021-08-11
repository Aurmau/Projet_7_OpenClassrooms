# import
import numpy as np
import pandas as pd


#############################################
# FOR API
# One hot encoding to a categorical variable
def ohe_to_categ(data_dummies, categ):
    """
    Create a list from a dataframe of a variable of one hot encoding
    :param data_dummies: dataframe with variable in one hot encoding
    :param categ: str name of the variable
    :return: list which can be store in a df[column]
    """

    df = data_dummies.filter(regex=categ)
    df.columns = df.columns.str.replace(categ + '_', '')
    df = df.idxmax(axis=1)
    return list(df)


def data_for_api(df):
    """
    Transform dataframe to be upload in the API of the project.
    :param df: dataframe
    :return: dataframe
    """

    # Selection of the most important features for the API
    feature_selection = ["SK_ID_CURR",
                         "TARGET",
                         "SCORING_PREDICT",
                         "AMT_GOODS_PRICE",
                         "AMT_CREDIT",
                         "AMT_ANNUITY",
                         "AMT_INCOME_TOTAL",
                         "ANNUITY_INCOME_PERCENT",
                         "CREDIT_TERM",
                         "DAYS_BIRTH",
                         "DAYS_EMPLOYED",
                         "DAYS_EMPLOYED_PERCENT",
                         "REGION_POPULATION_RELATIVE",
                         "CNT_CHILDREN"]

    try:
        df_api = df[feature_selection]
    except KeyError:
        del feature_selection[1]
        df_api = df[feature_selection]

    # create categ from dummies
    ohe_categ = ["CODE_GENDER",
                 "NAME_INCOME_TYPE",
                 "NAME_EDUCATION_TYPE",
                 "NAME_FAMILY_STATUS",
                 "NAME_HOUSING_TYPE",
                 "ORGANIZATION_TYPE",
                 "OCCUPATION_TYPE"]

    for categ in ohe_categ:
        df_api[categ] = ohe_to_categ(df, categ)

    # Round
    feature_to_round = ["AMT_GOODS_PRICE",
                        "AMT_CREDIT",
                        "AMT_ANNUITY",
                        "AMT_INCOME_TOTAL",
                        "ANNUITY_INCOME_PERCENT",
                        "CREDIT_TERM",
                        "DAYS_BIRTH",
                        "DAYS_EMPLOYED",
                        "DAYS_EMPLOYED_PERCENT",
                        "REGION_POPULATION_RELATIVE",
                        "CNT_CHILDREN"]

    for feature in feature_to_round:
        df_api[feature] = np.select(
            [df_api[feature] > 1, df_api[feature] <= 1],
            [df_api[feature].round(0), df_api[feature].round(2)],
        )

    df_api.set_index("SK_ID_CURR", inplace=True)

    return df_api
