import requests
import json
import pandas as pd
import random

# Dash
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# Plotly
import plotly.graph_objects as go
import plotly.express as px

# Tuto
# https://dash.plotly.com/basic-callbacks

########################################################################
url_heroku = "https://p7-oc-api.herokuapp.com"


# Import data from API
def import_api(api_url=url_heroku,
               api_route="/api/new_customer/",
               customer_id=None):
    """
    Import data from an API
    :param api_url: (str) main url of the API
    :param api_route: (str) route to the data
    :param customer_id: (int) customer_id
    :return: dataframe of the data downloaded from the API
    """
    if customer_id is None:
        url = api_url + api_route
    else:
        url = api_url + api_route + str(customer_id)
    data_json = requests.get(url).json()

    if type(data_json) is list:
        data = data_json.copy()
    else:
        data = pd.DataFrame.from_dict(data_json, orient="index")
        data.index = data.index.astype(int)

    return data


comparison_data = pd.read_csv("comparison_data.csv").drop(columns=["Unnamed: 0"])


# print(comparison_data["DAYS_BIRTH_qcut"].unique())


########################################################################
# REUSABLE COMPONENTS
def generate_table(df, max_rows=10):
    """
    Generate a table in html from a dataframe
    :param df: dataframe to display
    :param max_rows: (int) number of rows maximum to display
    :return:
    """
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in df.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(df.iloc[i][col]) for col in df.columns
            ]) for i in range(min(len(df), max_rows))
        ])
    ])


# list of dict of SK_ID_CURR
list_index = import_api(api_url=url_heroku,
                        api_route="/api/new_customer/index_list/")
list_dict_id = [{"label": str(cust_id), "value": int(cust_id)} for cust_id in list_index]


# Calculate statistics for comparison
def filter_comparison(df, filter="ALL", list_var=None):
    """
    filter comparison data
    :param df: dataframe comparison data
    :param filter: list of the intervals filter
    :param list_var: list of the variables names
    :return: dataframe of the comparison data filtered
    """
    df_filter = df.copy()
    if filter != "ALL":
        for var_i in range(len(list_var)):
            name_var = list_var[var_i]
            intervals = filter[var_i]
            df_filter = df_filter[df_filter[name_var].isin(intervals)]
    df_filter_agg = df_filter.groupby(["AGG"]).mean()
    return df_filter_agg


list_var = ["AMT_CREDIT_qcut", "ANNUITY_INCOME_PERCENT_qcut", "DAYS_BIRTH_qcut"]
data_comparison_agg = filter_comparison(comparison_data, filter="ALL", list_var=list_var)


def bullet_indicator(name_var=None, title_var=None, customer_id=None, data_value=None,
                     domain_y=None, data_agg=data_comparison_agg, inverse=False):
    """
    display go.Indicator
    :param name_var: (str) name of the variable
    :param title_var: (str) name of the variable to display
    :param customer_id: (int) customer id
    :param data_value: dataframe of the values
    :param domain_y: domain between 0 and 1 for the height of the plot
    :param data_agg: dataframe of the comparison data
    :param inverse: if inverse is True the bullet indicator show inversed indicator
    :return:
    """
    value_bullet = data_value.loc[customer_id, name_var]
    mean_bullet = data_agg.loc["mean", name_var]
    min_bullet = data_agg.loc["min", name_var]
    max_bullet = data_agg.loc["max", name_var]
    if inverse is False:
        reference_bullet = data_agg.loc["mean", name_var]
        step0_bullet = data_agg.loc["min", name_var]
        step10_bullet = data_agg.loc["q10", name_var]
        step25_bullet = data_agg.loc["q25", name_var]
        color_step1 = "gray"
        color_step2 = "lightgray"
    elif inverse is True:
        reference_bullet = value_bullet + (value_bullet - mean_bullet)
        step0_bullet = data_agg.loc["q75", name_var]
        step10_bullet = data_agg.loc["q90", name_var]
        step25_bullet = data_agg.loc["max", name_var]
        color_step1 = "lightgray"
        color_step2 = "gray"
    return go.Indicator(
        title={'text': title_var},
        mode="number+gauge+delta",
        domain={'x': [0.25, 1], 'y': domain_y},
        value=value_bullet,
        delta={'reference': reference_bullet},
        gauge={'shape': "bullet",
               'axis': {'range': [min_bullet,
                                  max_bullet]},
               'threshold': {'line': {'color': "red", 'width': 2},
                             'thickness': 0.75,
                             'value': mean_bullet},
               'steps': [{'range': [step0_bullet,
                                    step10_bullet],
                          'color': color_step1},
                         {'range': [step10_bullet,
                                    step25_bullet],
                          'color': color_step2}],
               'bar': {'color': "green"}
               })


########################################################################
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

# colors = {
#     'background': 'white',
#     'text': 'black'
# }
#
# fig = px.bar(data_former_customer, x="CODE_GENDER", y="TARGET", color='TARGET', barmode="group")
#
# fig.update_layout(
#     plot_bgcolor=colors['background'],
#     paper_bgcolor=colors['background'],
#     font_color=colors['text']
# )

app.layout = html.Div(children=[

    html.Div(children=[
        html.H1(children="Prêt à dépenser",
                style={'textAlign': 'center'},
                className="header-title"
                ),
        html.H2(children="Probabilité de paiement d'un prêt par un client",
                style={'textAlign': 'center'},
                className="header-description"
                )
    ],
        className="header", style={'backgroundColor': '#F5F5F5'}, ),

    html.Div(children="Renseignez l'identifiant client dans l'encadré ci-dessous",
             style={'textAlign': 'center'}
             ),
    dcc.Dropdown(id='customer_id-dropdown',
                 options=list_dict_id,
                 value=100001,
                 placeholder="Select a customer id",
                 style={"width": "50%", "margin": "auto"}
                 ),

    html.H3(children="Informations générales du client et son score",
            style={'textAlign': 'center'}),

    html.Div(children=[
        dcc.Graph(id="table_description",
                  style={"width": "50%", 'display': 'inline-block', 'vertical-align': 'middle'}),
        dcc.Graph(id='scoring-gauge',
                  style={"width": "50%", 'display': 'inline-block', 'vertical-align': 'middle'}),
    ]),

    html.H3(children="Variables augmentant ou diminuant le score",
            style={'textAlign': 'center'}),
    dcc.Graph(id='shap-force-plot'),

    html.H3(children="Comparaison avec les autres clients",
            style={'textAlign': 'center'}),
    html.Div([
        html.Div(children=[
            html.Div(children="Filtre sur la valeur du crédit ($)",
                     style={'textAlign': 'center', 'font-weight': 'bold'}
                     ),
            dcc.Checklist(id="amt_credit-checklist",
                          options=[{'label': '$44,999 - $254,700', 'value': '(44999.999, 254700.0]'},
                                   {'label': '$254,700 - $432,567', 'value': '(254700.0, 432567.0]'},
                                   {'label': '$432,567 - $616,500', 'value': '(432567.0, 616500.0]'},
                                   {'label': '$616,500 - $900,000', 'value': '(616500.0, 900000.0]'},
                                   {'label': '$900,000 - $4,050,000', 'value': '(900000.0, 4050000.0]'}
                                   ],
                          value=['(44999.999, 254700.0]',
                                 '(254700.0, 432567.0]',
                                 '(432567.0, 616500.0]',
                                 '(616500.0, 900000.0]',
                                 '(900000.0, 4050000.0]'],
                          style={"width": "50%", "margin": "auto"}),

            html.Div(children="Filtre sur le pourcentage Annuity/Income",
                     style={'textAlign': 'center', 'font-weight': 'bold'}
                     ),
            dcc.Checklist(id="annuity_income_percent-checklist",
                          options=[{'label': 'Moins de 10%', 'value': '(-0.001, 0.1]'},
                                   {'label': '10% - 14%', 'value': '(0.1, 0.14]'},
                                   {'label': '14% - 19%', 'value': '(0.14, 0.19]'},
                                   {'label': '19% - 25%', 'value': '(0.19, 0.25]'},
                                   {'label': 'Plus de 25%', 'value': '(0.25, 2.0]'}
                                   ],
                          value=['(-0.001, 0.1]',
                                 '(0.1, 0.14]',
                                 '(0.14, 0.19]',
                                 '(0.19, 0.25]',
                                 '(0.25, 2.0]'],
                          style={"width": "50%", "margin": "auto"}),

            html.Div(children="Filtre sur l'age du client",
                     style={'textAlign': 'center', 'font-weight': 'bold'}
                     ),
            dcc.Checklist(id="days_birth-checklist",
                          options=[{'label': 'Moins de 32 ans', 'value': '(7488.999, 11794.0]'},
                                   {'label': '32 ans - 39 ans', 'value': '(11794.0, 14545.0]'},
                                   {'label': '39 ans - 47 ans', 'value': '(14545.0, 17366.0]'},
                                   {'label': '47 ans - 56 ans', 'value': '(17366.0, 20579.0]'},
                                   {'label': 'Plus de 56 ans', 'value': '(20579.0, 25229.0]'}
                                   ],
                          value=['(7488.999, 11794.0]',
                                 '(11794.0, 14545.0]',
                                 '(14545.0, 17366.0]',
                                 '(17366.0, 20579.0]',
                                 '(20579.0, 25229.0]'],
                          style={"width": "50%", "margin": "auto"})
        ],
            style={"width": "19%", 'float': 'left', 'backgroundColor': 'lightgray', "margin-bottom": "15px"}),

        html.Div(children=[
            dcc.Graph(id='indicator-gauge'),
        ],
            style={"width": "79%", 'float': 'right'})
    ],
        style={'backgroundColor': 'white'}),

    html.Div(children=[
        dcc.Markdown(children='''
    **NOTE** : Ce dashboard permet de visualiser la probabilité qu'un client rembourse sont prêt. 
    Cette probabilité se retrouve sous la forme d'un score allant de 0 pour une probabilité de remboursement nulle
    et 100 pour une probabilité de remboursement maximale.
    
    1) Dans un premier temps l'identifiant du client doit être renseigné à partir de la liste déroulante du haut de 
    page. Attention, la mise à jour de la page peut prendre plusieurs secondes.     
    
    2) Les informations générales du client sont affichées dans un tableau.
    
    3) Le score du client est affiché dans un Bullet chart. Un Bullet chart permet de visualier une valeur en fonction 
    d'autres valeurs de référence. Ici le score est affiché en vert, la moyenne des scores des autres clients est 
    affichée en rouge. Les 10 % des valeurs les plus mauvaises sont affichés en gris foncé et en gris clair pour 
    celles allant de 10 à 25 %. Enfin il est possible de lire à droite la valeur du client ainsi que la différence entre
     cette valeur et la moyenne.
     
     4) Sous le tableau et le Bullet chart il est possible de visualiser les variables augmentant ou diminuant le score.
      Ces informations sont calculées à partir des shapley values. Les shapley values permettent de savoir pour 
      un client donné, quelles variables ont plus ou moins impactés l'algorithmes de calcul du score. Il est également 
      possible de savoir si l'impact de la variable à était positif ou négatif sur le calcul du score. 
      
      5) Le dernier graphique est une combinaison de Bullet chart sur différentes variables. Le groupe de clients par 
      rapport auquel les valeurs sont comparés, peut être défini à l'aide des filtres se situant dans l'encart gris. 
      ''')],
        style={'float': 'left', 'backgroundColor': '#F5F5F5'})
])


@app.callback(
    Output('table_description', 'figure'),
    Input('customer_id-dropdown', 'value')
)
def update_table_description_id(customer_id):
    data_new_customer_id = import_api(api_url=url_heroku,
                                      api_route="/api/new_customer/",
                                      customer_id=customer_id)

    data_new_customer_id_table = data_new_customer_id[["CODE_GENDER",
                                                       "DAYS_BIRTH",
                                                       "OCCUPATION_TYPE",
                                                       "ORGANIZATION_TYPE",
                                                       "DAYS_EMPLOYED",
                                                       "NAME_FAMILY_STATUS",
                                                       "CNT_CHILDREN",
                                                       "AMT_GOODS_PRICE",
                                                       "CREDIT_TERM"]]
    data_new_customer_id_table = data_new_customer_id_table.T

    table = go.Figure(data=[go.Table(
        header=dict(values=["<b>customer id<b>", "<b>" + str(customer_id) + "<b>"],
                    fill_color='paleturquoise',
                    align='center',
                    font_size=20,
                    height=40),
        cells=dict(values=[data_new_customer_id_table.index,
                           data_new_customer_id_table],
                   fill_color='lavender',
                   align='center',
                   font_size=15,
                   height=30))
    ])

    table.update_layout(transition_duration=500, height=400, margin={'t': 40, 'b': 20, 'l': 10, 'r': 0})

    return table


@app.callback(
    Output('scoring-gauge', 'figure'),
    Input('customer_id-dropdown', 'value'),
    Input('amt_credit-checklist', 'value'),
    Input('annuity_income_percent-checklist', 'value'),
    Input('days_birth-checklist', 'value')
)
def update_scoring_gauge(customer_id, credit_range, annuity_income_range, days_birth_range):
    data_new_customer_id = import_api(api_url=url_heroku,
                                      api_route="/api/new_customer/",
                                      customer_id=customer_id)

    score_id = data_new_customer_id.loc[customer_id, "SCORING_PREDICT"]

    list_filter = [credit_range, annuity_income_range, days_birth_range]
    data_comparison_agg_up = filter_comparison(comparison_data, filter=list_filter, list_var=list_var)

    fig = go.Figure(go.Indicator(
        title={'text': "<b>Score<b>"},
        mode="number+gauge+delta",
        domain={'x': [0.05, 1], 'y': [0.1, 0.9]},
        value=score_id,
        delta={'reference': data_comparison_agg_up.loc["mean", "SCORING_PREDICT"]},
        gauge={'shape': "bullet",
               'axis': {'range': [None, 100]},
               'threshold': {'line': {'color': "red", 'width': 2},
                             'thickness': 0.75,
                             'value': data_comparison_agg_up.loc["mean", "SCORING_PREDICT"]},
               'steps': [{'range': [0, data_comparison_agg_up.loc["q10", "SCORING_PREDICT"]],
                          'color': "gray"},
                         {'range': [data_comparison_agg_up.loc["q10", "SCORING_PREDICT"],
                                    data_comparison_agg_up.loc["q25", "SCORING_PREDICT"]],
                          'color': "lightgray"}]
               },

    ))

    fig.update_layout(transition_duration=500, height=400, margin={'t': 0, 'b': 20, 'l': 80, 'r': 0})

    return fig


@app.callback(
    Output('indicator-gauge', 'figure'),
    Input('customer_id-dropdown', 'value'),
    Input('amt_credit-checklist', 'value'),
    Input('annuity_income_percent-checklist', 'value'),
    Input('days_birth-checklist', 'value')
)
def update_indicator_gauge(customer_id, credit_range, annuity_income_range, days_birth_range):
    data_new_customer_id = import_api(api_url=url_heroku,
                                      api_route="/api/new_customer/",
                                      customer_id=customer_id)

    list_filter = [credit_range, annuity_income_range, days_birth_range]
    data_comparison_agg_up = filter_comparison(comparison_data, filter=list_filter, list_var=list_var)

    fig = go.Figure()

    fig.add_trace(bullet_indicator(name_var="ANNUITY_INCOME_PERCENT", title_var="Ratio Annuity/Income",
                                   data_value=data_new_customer_id, customer_id=customer_id,
                                   domain_y=[0.1, 0.25], data_agg=data_comparison_agg_up, inverse=True))
    fig.add_trace(bullet_indicator(name_var="DAYS_EMPLOYED_PERCENT", title_var="Ratio jours embauché",
                                   data_value=data_new_customer_id, customer_id=customer_id,
                                   domain_y=[0.35, 0.5], data_agg=data_comparison_agg_up))
    fig.add_trace(bullet_indicator(name_var="AMT_ANNUITY", title_var="Annuity ($)",
                                   data_value=data_new_customer_id, customer_id=customer_id,
                                   domain_y=[0.6, 0.75], data_agg=data_comparison_agg_up, inverse=True))
    fig.add_trace(bullet_indicator(name_var="AMT_GOODS_PRICE", title_var="Valeur du bien ($)",
                                   data_value=data_new_customer_id, customer_id=customer_id,
                                   domain_y=[0.85, 1], data_agg=data_comparison_agg_up))

    fig.update_layout(transition_duration=500, height=400, margin={'t': 10, 'b': 0, 'l': 0, 'r': 50})

    return fig


@app.callback(
    Output('shap-force-plot', 'figure'),
    Input('customer_id-dropdown', 'value')
)
def update_shap_force_plot(customer_id):
    data_new_customer_id = import_api(api_url=url_heroku,
                                      api_route="/api/new_customer/",
                                      customer_id=customer_id)
    app_new_customer_shap = data_new_customer_id.drop(columns=["SCORING_PREDICT"])

    data_shap_values = import_api(api_url=url_heroku,
                                  api_route="/api/new_customer/shap_values/",
                                  customer_id=customer_id)
    shap_sorted = data_shap_values.loc[customer_id, :].sort_values()
    n_features = 5
    shap_sorted_min = shap_sorted[:n_features]
    shap_sorted_max = shap_sorted[-n_features:]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=shap_sorted_min.values,
        y=shap_sorted_min.index,
        text=app_new_customer_shap.loc[customer_id, shap_sorted_min.index].values,
        textposition="inside",
        name='Valeurs diminuant le score',
        orientation='h',
        marker=dict(color='rgba(232, 2, 2, 0.6)',
                    line=dict(color='rgba(232, 2, 2, 1.0)', width=3))
    ))

    fig.add_trace(go.Bar(
        x=shap_sorted_max.values,
        y=shap_sorted_max.index,
        text=app_new_customer_shap.loc[customer_id, shap_sorted_max.index].values,
        textposition="inside",
        name='Valeurs augmentant le score',
        orientation='h',
        marker=dict(color='rgba(2, 200, 2, 0.6)',
                    line=dict(color='rgba(2, 200, 2, 1.0)', width=3))
    ))

    fig.update_layout(transition_duration=500, height=400, margin={'t': 0, 'b': 20, 'l': 20, 'r': 20})

    return fig


if __name__ == '__main__':
    app.run_server()
