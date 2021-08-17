# import
import numpy as np
import pandas as pd
# Modeling
import lightgbm as lgb
import shap
import joblib


#########################################
# MACHINE LEARNING

def lgbm_scoring_prediction(df, name_model_file="model_lgbm.pkl"):
    """
        Apply the model name_model_file to a dataframe and return the same dataframe with the prediction in %
        :param df: Dataframe
        :param name_model_file: name of the model (pickel file .pkl)
        :return: dataframe df with the prediction of the model
        """

    model_lgbm = joblib.load(name_model_file)

    # Prepared datas
    try:
        X = df.drop(columns=["SK_ID_CURR", "TARGET"])
    except KeyError:
        X = df.drop(columns=["SK_ID_CURR"])

    # Apply the model
    preds = model_lgbm.predict_proba(X)[:, 1]

    # Add the scoring prediction to app_train
    df_scoring = df.copy()
    df_scoring["SCORING_PREDICT"] = preds
    df_scoring["SCORING_PREDICT"] = ((1 - df_scoring["SCORING_PREDICT"]) * 100).round()

    return df_scoring


# Shapley values
# Max shap value of a categorical variable
def shap_to_categ(data_dummies, categ):
    """
        Calculate the maximum shapley values of a dummies variables
        :param data_dummies: dataframe of the shapley values with dummies variables
        :param categ: str of the name of the variable with dummies
        :return: a list of shapley values than can be store in df[column]
        """

    df = data_dummies.filter(regex=categ)
    df_abs = df.abs()
    max_loc = df_abs.idxmax(axis="columns")
    max_shap = [df.loc[id_cust, max_loc[id_cust]] for id_cust in df.index]
    return max_shap


def shapley_values(df, name_model_file="model_lgbm.pkl"):
    """
        Calculate shapley values of a dataframe for a model
        :param df: dataframe
        :param name_model_file: str of the name of the machine learning model (pickel file .pkl)
        :return: dataframe of the shapley values
        """

    model_lgbm = joblib.load(name_model_file)

    # Prepared datas
    X = df.set_index("SK_ID_CURR")
    try:
        X.drop(columns=["TARGET"], inplace=True)
    except KeyError:
        pass

    explainer = shap.TreeExplainer(model_lgbm)
    shap_values = explainer.shap_values(X)
    shap_values_df = pd.DataFrame(shap_values[0], index=X.index, columns=X.columns)

    feature_selection = ["AMT_GOODS_PRICE",
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

    categories = ["CODE_GENDER",
                  "NAME_INCOME_TYPE",
                  "NAME_EDUCATION_TYPE",
                  "NAME_FAMILY_STATUS",
                  "NAME_HOUSING_TYPE",
                  "ORGANIZATION_TYPE",
                  "OCCUPATION_TYPE"]

    shap_values_df_select = shap_values_df[feature_selection]

    for categ in categories:
        shap_values_df_select[categ] = shap_to_categ(shap_values_df, categ)

    return shap_values_df_select


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
