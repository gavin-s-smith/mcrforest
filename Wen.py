import pandas as pd
from mcrforest.forest import RandomForestRegressor,RandomForestClassifier
import numpy as np
from tqdm import tqdm


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore') 

"""**Load the dataset**


---
I have uploaded the preprocessed data to Google Drive. There are four datasets. "X_smote_train","y_smote_train"( meaning X_train and y_train, which have been one-hot encoded and over-resampled), "X_ohe_test"(X_test has been one-hot encoded) and "y_test".
"""


def plot_mcr(df_in):
    df_in = df_in.copy()
    df_in.columns = [ x.replace('MCR+', 'MCR- (lollypops) | MCR+ (bars)') for x in df_in.columns]
    ax = sns.barplot(x='MCR- (lollypops) | MCR+ (bars)',y='variable',data=df_in)

    plt.hlines(y=range(df_in.shape[0]), xmin=0, xmax=df_in['MCR-'], color='skyblue')
    plt.plot(df_in['MCR-'], range(df_in.shape[0]), "o", color = 'skyblue')
    plt.show()

    #3. Read file as panda dataframe
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
train_run = True

if train_run:

    X_smote_train = pd.read_csv("/home/gavin/Downloads/X_smote_train.csv")
    y_smote_train = pd.read_csv("/home/gavin/Downloads/y_smote_train.csv")

    X_ohe_test = pd.read_csv("/home/gavin/Downloads/X_ohe_test.csv")
    y_test = pd.read_csv("/home/gavin/Downloads/y_test.csv")

    # instruction:
    #(1)run the cell, 
    #(2)click the following link
    #(3)sign in a google account
    #(4)allow the goole cloud SDK
    #(5)copy the code
    #(6)paset the code to the rectangle

    X_smote_train.head(2)

    X_smote_train = X_smote_train.iloc[:,1:]
    X_smote_train.head(2)

    y_smote_train = y_smote_train.iloc[:,1:]
    y_smote_train.head(2)

    print(y_smote_train.shape)
    print(X_smote_train.shape)

    X_ohe_test.head(2)

    X_ohe_test = X_ohe_test.iloc[:,1:]
    X_ohe_test.head(2)

    X_ohe_test.shape

    y_test.head(2)

    y_test = y_test.iloc[:,1:]

    y_test.shape

   



    """**RF-MCR**

    ---
    use the best model selected from the other .ipython file directly.
    """

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # For the model class RF
    #search = {'n_estimators':[500],'min_impurity_decrease':[0.000001,0.00001,0.0001,0.001,0.01],'max_features':[1,'auto']}

    # {'bootstrap': False, 'ccp_alpha': 0.0, 'criterion': 'mse', 'max_depth': None, 'max_features': 1, 'max_leaf_nodes': None, 'max_samples': None, 'mcr_tree_equivilient_tol': 3.1713, 'min_impurity_decrease': 1e-06, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 500, 'n_jobs': None, 'oob_score': False, 'performance_equivilence': True, 'random_state': 13111985, 'spoof_as_sklearn': None, 'verbose': 0, 'warm_start': False}

    #rf_cv_model = GridSearchCV(RandomForestRegressor(mcr_tree_equivilient_tol=3.1713, bootstrap=False, criterion='mse', random_state=13111985), search, cv = kf, refit = True)
    #rf_best_model = rf_cv_model.fit(X_smote_train, y_smote_train.values.flatten()).best_estimator_
    #print(rf_best_model.get_params())
    

    rf_best_model = RandomForestClassifier(bootstrap=False,random_state=42,n_estimators = 150, max_depth = 9, min_samples_leaf = 18)
    rf_best_model.fit(X_smote_train, y_smote_train)

    # Commented out IPython magic to ensure Python compatibility.
    # # compute MCR on test data

    from tqdm import tqdm
    variables = []
    mcrp = []
    mcrm = []
    i = 0
    for c in tqdm(X_smote_train.columns):
        variables.append(c)
        mcr_p = rf_best_model.mcr( X_smote_train.values,y_smote_train.values.flatten(), np.asarray([i]), 
                            num_times = 100, debug = False, mcr_type = 1, mcr_as_ratio=False)

        mcr_m = rf_best_model.mcr( X_smote_train.values,y_smote_train.values.flatten(), np.asarray([i]), 
                            num_times = 100, debug = False, mcr_type = -1, mcr_as_ratio=False)

        mcrm.append(mcr_m)
        mcrp.append(mcr_p)
        i += 1

    rf_results = pd.DataFrame({'variable':variables, 'MCR+':mcrp, 'MCR-':mcrm})

    rf_results.sort_values("MCR-",ascending= False)


    rf_results.to_csv('RF-MCR3.csv') 

else:
    rf_results = pd.read_csv('RF-MCR3.csv')

    #plot of mcr on test set
    plt.rcParams['figure.figsize'] = (10,15.0)
    plot_mcr(rf_results)


