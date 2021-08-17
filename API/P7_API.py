# -*- coding: utf-8 -*-
from flask import Flask, jsonify
import pandas as pd
# Suppress warnings
import warnings

warnings.filterwarnings('ignore')

from P7_functions_API import *

P7_API = Flask(__name__)


@P7_API.route("/")
def hello():
    hello_API = {"Title": "API P7 OpenClassrooms Data Science",
                 "By": "Aurelien Maurie",
                 "Description": "Welcome on the API of the Project 7 of the Data Science Master's Degree with "
                                 "OpenClassrooms",
                 "Route_1": "https://p7-oc-api.herokuapp.com/api/new_customer/index_list/",
                 "Route_2": "https://p7-oc-api.herokuapp.com/api/new_customer/<i>customer_id<i>/",
                 "Route_3": "https://p7-oc-api.herokuapp.com/api/new_customer/shap_values/<i>customer_id<i>/"}
    return jsonify(hello_API)


@P7_API.route("/api/new_customer/index_list/")
def get_new_customer_index():
    data_new_customer = pd.read_csv("data_new_customer.csv").drop(columns=["Unnamed: 0"])
    index_customer = list(data_new_customer["SK_ID_CURR"])
    return jsonify(index_customer)


@P7_API.route("/api/new_customer/<int:customer_id>/")
def get_data_new_customer_id(customer_id):
    data_new_customer = pd.read_csv("data_new_customer.csv").drop(columns=["Unnamed: 0"])
    df_customer_id = data_new_customer[data_new_customer["SK_ID_CURR"] == customer_id]
    df_scoring = lgbm_scoring_prediction(df_customer_id, name_model_file="model_lgbm.pkl")
    df_api = data_for_api(df_scoring)
    dict_api = df_api.to_dict('index')
    return jsonify(dict_api)


@P7_API.route("/api/new_customer/shap_values/<int:customer_id>/")
def get_shap_values_new_customer_id(customer_id):
    data_new_customer = pd.read_csv("data_new_customer.csv").drop(columns=["Unnamed: 0"])
    df_customer_id = data_new_customer[data_new_customer["SK_ID_CURR"] == customer_id]
    df_shap = shapley_values(df_customer_id, name_model_file="model_lgbm.pkl")
    dict_api = df_shap.to_dict('index')
    return jsonify(dict_api)


if __name__ == "__main__":
    P7_API.run(debug=True)
