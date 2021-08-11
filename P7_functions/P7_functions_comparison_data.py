import numpy as np
import pandas as pd
import itertools


# Calculate statistics
# def Percentile
def q10(x):
    """
    Return values at the given quantile 0.1 on x
    :param x: dataframe requested axis
    :return: quantile 0.1
    """
    return x.quantile(0.1)


def q25(x):
    """
        Return values at the given quantile 0.25 on x
        :param x: dataframe requested axis
        :return: quantile 0.25
        """
    return x.quantile(0.25)


def q75(x):
    """
        Return values at the given quantile 0.75 on x
        :param x: dataframe requested axis
        :return: quantile 0.75
        """
    return x.quantile(0.75)


def q90(x):
    """
        Return values at the given quantile 0.9 on x
        :param x: dataframe requested axis
        :return: quantile 0.9
        """
    return x.quantile(0.9)


def calc_comparison_data(df, list_var=[]):
    """
    Return the dataframe that will be used in the dashboard as comparison data.
    Cut the variables in 5 with quantile(0.2, 0.4, 0.6, 0.8). Then for all possible combinations and
    for each variable selected the function calculated ["mean", "min", q10, q25, "median", q75, q90, "max"]
    :param df: original dataframe
    :param list_var: list of variables selected
    :return: The dataframe used in the final dashboard as comparison data.
    """

    df_t0 = df[df["TARGET"] == 0]
    # Select only numeric
    df_t0 = df_t0.select_dtypes(exclude=['object'])

    # Select var for the filter
    list_list_value = []

    for var in list_var:
        df_t0[var + "_qcut"] = pd.qcut(df_t0[var], 5)
        list_list_value.append(list(df_t0[var + "_qcut"].dropna().unique()))

    list_combination = list(itertools.product(*list_list_value))

    # CREATE DATAFRAME OF THE AGG
    df_t0_agg = pd.DataFrame(columns=df_t0.columns)

    for combination in list_combination:
        df_combination = df_t0.copy()
        # filter
        for i in range(len(list_var)):
            df_combination = df_combination[df_combination[list_var[i] + "_qcut"] == combination[i]]
        # AGG
        df_combination_agg = df_combination.agg(["mean", "min", q10, q25, "median", q75, q90, "max"])

        # Fill combination value
        for i in range(len(list_var)):
            df_combination_agg[list_var[i] + "_qcut"] = combination[i]
        df_t0_agg = df_t0_agg.append(df_combination_agg)

    df_t0_agg.reset_index(inplace=True)
    df_t0_agg.rename(columns={"index": "AGG"}, inplace=True)

    return df_t0_agg
