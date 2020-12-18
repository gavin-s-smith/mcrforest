import pandas as pd
import numpy as np
import psycopg2
from mcrforest.forest import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

input_datatable = 'r.filtered_input_data_delivery'
with psycopg2.connect("dbname='dtree2' user='gavin' host='128.243.99.138' password=''") as conn:
    # Include all records
    df = pd.read_sql("SELECT * FROM {}".format(input_datatable), con = conn)


df = df.drop(columns = 'client')
variables_for_prediction = df
variables_for_prediction = pd.get_dummies(variables_for_prediction,
                                          columns=[ "district",
                                                  "previous_delivery_location",
                                                  "drinking_water", "education",
                                                  "electricity", "floor", "roof", "shehia", "status" ])

variables_for_prediction = variables_for_prediction.dropna()
variables_for_prediction = variables_for_prediction.reset_index(drop=True)

variables_for_prediction[ "delivery_day" ] = ( variables_for_prediction["dateofdelivery"] - 
                                              variables_for_prediction["edd"] ).dt.days
variables_for_prediction[ "early_delivery" ] = variables_for_prediction[ "delivery_day" ] < -21


## Create training and test sets
x = variables_for_prediction.drop( columns=[ "delivery_day", "early_delivery", "edd", "dateofdelivery", "childdeath", "first_visit_wk" ] )
y = variables_for_prediction[ "early_delivery" ]
X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2 )


## Parameters found from grid search
params = { 'bootstrap': False, 'max_depth': None, 'max_features': 'auto',
          'min_samples_leaf': 4, 'min_samples_split': 2, 'random_state':42 }

model = RandomForestClassifier( **params )


model.fit( X_train.values, y_train )