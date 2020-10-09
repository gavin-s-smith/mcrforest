# mcrforest

This repository contains the code from our 2020 NeurIPS paper (to appear) on Model Class Reliance. 

## Installation
Install for use via pip:
pip install git+https://github.com/gavin-s-smith/mcrforest

Install for developing / debugging from unzipped source:
python setup.py build_ext --inplace

## Usage

mcrforest is an extention to the sklearn RandomForestClassifier and RandomForestRegressor classes and can be used as a direct replacement.

mcrforest includes an additional method which can be called after training a model. In addition there are two restrictions on the model building that must be met.
1. bootstrap must be set to false
2. when using a RandomForestClassifier currently only binary classification is supported and the labels must be 0,1

```
mcr(X_in, y_in, indices_to_permute, num_times = 100, mcr_type = 1, seed = 13111985)
```

Computes and MCR+ or MCR- score of a variable or group of variables.

*Parameters:*
```
X_in (2D numpy array): The input features to use to compute the MCR.
y_in (1D numpy array): The output features to use to compute the MCR.
indices_to_permute (1D numpy array): A numpy array of the index/indices indicating the variable or group of variables to compute MCR for.
num_times (int): The number of times to permute the index/indices.
mcr_type (int): 1 for MCR+, -1 for MCR-.
seed (int): A seed to control the permutation randomness.
```


### Example Usage
```
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mcrforest.forest import RandomForestClassifier

# Load data
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')

# If we are going to use the training set for computing MCR then we MUST ensure 
# we have controlled the complexity of the fit.
# See: https://christophm.github.io/interpretable-ml-book/feature-importance.html

base_model = RandomForestRegressor(random_state = 13111985, bootstrap=False)

search = {'n_estimators':[500],'max_features':['auto'], 'max_depth':[5,10,15,20,30]}

rf_cv_model = GridSearchCV(base_model, search)
rf_cv_model.fit(X_train,y_train)

# Refit with best parameters
model = RandomForestRegressor( **rf_cv_model.best_params_ )

# Compute MCR+ for each variable
results = []
groups_of_indicies_to_permute = [[x] for x in range(len(X_train.columns))]

for gp in groups_of_indicies_to_permute:
    rn = model.mcr(X_train.values,y_train.values, np.asarray(gp) ,  num_times = 20, mcr_type = 1)
    results.append([','.join([list(X_train.columns)[x] for x in gp]), 'RF-MCR+', rn])


# Compute MCR- for each variable
for gp in groups_of_indicies_to_permute:
    rn = model.mcr(X_train.values,y_train.values, np.asarray(gp) ,  num_times = 20,  mcr_type = -1)
    results.append([','.join([list(X_train.columns)[x] for x in gp]), 'RF-MCR-', rn])


# Plot the results

lbl = [ x[0] for x in results if 'MCR+' in x[1] ]
mcrp = [ x[2] for x in results if 'MCR+' in x[1] ]
mcrm = [ x[2] for x in results if 'MCR-' in x[1] ]

rf_results = pd.DataFrame({'variable':lbl, 'MCR+':mcrp, 'MCR-':mcrm})


def plot_mcr(df_in, fig_size = (11.7, 8.27)):
    df_in = df_in.copy()
    df_in.columns = [ x.replace('MCR+', 'MCR- (lollypops) | MCR+ (bars)') for x in df_in.columns]
    ax = sns.barplot(x='MCR- (lollypops) | MCR+ (bars)',y='variable',data=df_in)
    plt.gcf().set_size_inches(fig_size)
    plt.hlines(y=range(df_in.shape[0]), xmin=0, xmax=df_in['MCR-'], color='skyblue')
    plt.plot(df_in['MCR-'], range(df_in.shape[0]), "o", color = 'skyblue')

plot_mcr(rf_results)
plt.show()
```