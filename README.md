# Projet_7_OpenClassrooms

## Summary
This project is part of the Data Scientist Master Degree with OpenClassrooms. The aim of the project is to create an 
interactive dashboard for the company "Prêt à dépenser" which offers consumer credit.

The company wants to develop a scoring model of the customer's probability of default to support the decision on whether
 or not to lend to a potential customer based on a variety of data sources. In addition, customer relationship managers 
 have pointed out that there is a growing demand from customers for transparency in credit decisions. It therefore 
 decided to develop an interactive dashboard so that customer relationship managers could both explain credit decisions 
 as transparently as possible, but also allow their customers to have their personal information at their disposal and 
 to explore it easily.
 
## README contents
1. Data
2. Preprocessing
3. Machine Learning
4. API
5. Dashboard
6. Project Architecture
7. How to use
 
## 1. Data
[Data can be downloaded here!](https://www.kaggle.com/c/home-credit-default-risk/data)
The data is provided by Home Credit, a service dedicated to provided lines of credit (loans) to the unbanked population.

There are 7 different sources of data but only the first one is used in this project:
* **application_train/application_test**: the main training and testing data with information about each loan 
application at Home Credit. Every loan has its own row and is identified by the feature SK_ID_CURR. 
The training application data comes with the TARGET indicating 0: the loan was repaid or 1: the loan was not repaid. 

    * `application_train.csv` are the data used to train the model. After preprocessing these data are called 
    `former_customer` and there are also used as comparison values in the dashboard.

    * `application_test.csv` are the data on which we want to calculate the score. After preprocessing the data are 
    called `new_customer`, these are the customers that we can see the results of in the dashboard. In the presence of 
    even more new customers the `application_test.csv` file needs to be updated.

## 2. Preprocessing
[Preprocessing was done using this kernel.](https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction)

The `P7_preprocessing.py` Python file (in the main folder) has to be runned before to deploy API and Dashboard with 
heroku. Data calculated with this code are automatically assigned to the good folder.

## 3. Machine Learning
[Machine Learning Model was done using this kernel.](https://www.kaggle.com/willkoehrsen/intro-to-model-tuning-grid-and-random-search)

The Machine Learning Model is a 
[Light Gradient Boosting Machine](https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/)
(LGBM) which was hyperparameter tuned with random search. This process was done in an external notebook and the model 
was exported as a pickel file named `model_lgbm.pkl`.

## 4. API
The API created for the project can be visited at this URL --> https://p7-oc-api.herokuapp.com/ 

The API is made of 3 routes:
* `/api/new_customer/index_list/`: That returns the list of all unique id of the new customers.
* `/api/new_customer/<int:customer_id>/`: That returns data for a specific customer and the score calculated with 
`model_lgbm.pkl`. `<int:customer_id>` corresponding to the unique id of the customer.
* `/api/new_customer/shap_values/<int:customer_id>/`: That returns 
[shapley values](https://towardsdatascience.com/explain-your-model-with-the-shap-values-bc36aac4de3d) calculated on 
`model_lgbm.pkl` for a specific customer. `<int:customer_id>` corresponding to the unique id of the customer.

## 5. Dashboard 
The dashboard created for the project can be visited at this URL --> https://p7-oc-dashboard.herokuapp.com/. Please not 
than this dashboard is deployed on a free licence Heroku servor and may take several seconds to load.

The dashboard is made of 5 parts:
* Selection of the customer via his id.
* Main information of the customer and his score.
* Shapley values
* Comparison between the customer and former customers on 4 features. Filter can be applyed on the group of former 
customers.
* Notes explaining the different visualisations.

## 6. Project architecture
* **P7_API**: *Files used to deployed the API with Heroku*
    * data_new_customer.csv
    * model_lgbm.pkl
    * P7_API.py
    * P7_functions_API.py
    * Procfile
    * Procfile.windows
    * requirement.txt
* **P7_Dashboard**: *Files used to deployed the dashboard with Heroku*
    * comparison_data.csv
    * P7_dashboard_pyplot_dash.py
    * Procfile
    * Procfile.windows
    * requirement.txt
* **P7_data**: *Raw data of the project*
    * application_test.csv
    * application_train.csv
* **P7_functions**: *Functions used in P7_preprocessing.py*
    * P7_functions_comparison_data.py
    * P7_functions_data_for_API.py
    * P7_functions_ML.py
    * P7_functions_preprocessing.py
* **model_lgbm.pkl**: *Machine Learning used in this project*
* **P7_preprocessing.py**: *Python file to run before to deploy API and Dashboard with heroku. Data calculated with this 
code are automatically assigned to the good folder.*
* **README.md**

## 7. How to use
1. The folder P7_data contains raw data `application_train.csv` and `application_test.csv`. If this project need to be 
applied on new customer the `application_test.csv` need to be upgrade following same template as it is possible to find 
[here.](https://www.kaggle.com/c/home-credit-default-risk/data)

2. In the main folder: `P7_preprocessing.py` Python file has to be runned before to deploy API and Dashboard with 
heroku. Data calculated with this code are automatically assigned to the good folder. Functions used in this python file
are stored in the folder `P7_functions`.

3. The folders `P7_API` and `P7_Dashboard` contains files to deploy respectively the API and the dashboard with Heroku, 
 [an Heroku tutorial is disponible here.](https://devcenter.heroku.com/articles/getting-started-with-python)
 The versions of packages used are in `requirement.txt`. 
 
 4. Heroku - run app locally: to run the API or the dashboard locally with windows it is possible to use the command 
 `$ heroku local web -f Procfile.windows` in the Terminal. The command has to be run in the folder which contains the 
 app (`P7_API` or `P7_Dashboard`).
 
 5. Heroku - deploy the app: to deploy the API or the dashboard locally after create a new app via 
 [heroku.com](https://dashboard.heroku.com/apps), the following commands have to be runned in the terminal:
 
>`$ cd name_of_the_folder_where_is_your_app`
>
>`$ git init`
>
>`$ heroku git:remote -a name_of_your_app`
>
>`$ git add .`
>
>`$ git commit -am "your_commentary_here"`
>
>`$ git push heroku master`