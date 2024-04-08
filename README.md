# Model Class Reliance for Random Forests (python package: mcrforest)

An implementation of **Model Class Reliance** for Random Forests (RF-MCR), including Group-MCR. 

**Variable Importance:** Explains the importance of a variable in a single, typically arbitrary, machine learnt model.  
**Model Class Reliance:** Explains the underpining phenomena (under mild assumptions) by considering all models with equally optimal performance.

See this [3 Minute Explainer Video](https://slideslive.com/38937760/model-class-reliance-mcr-for-random-forests).

**RF-MCR is introduced in:**  
*Smith, G., Mansilla, R. and Goulding, J. "Model Class Reliance for Random Forests". 34th Conference on Neural Information Processing Systems (NeurIPS 2020), Vancouver, Canada.*<br />[Paper](https://proceedings.neurips.cc/paper/2020/hash/fd512441a1a791770a6fa573d688bff5-Abstract.html) | [Supplementary Material](https://proceedings.neurips.cc/paper/2020/file/fd512441a1a791770a6fa573d688bff5-Supplemental.pdf) 

**Group-MCR for RF-MCR is introduced in:**  
*Ljevar, V., Goulding, J., Smith, G. and Spence, A. "Using Model Class Reliance to measure group effect on adherence to asthma medication". (IEEE International Conference on Big Data (Big Data). IEEE, 2021.).* <br /> [Pre-print](http://www.cs.nott.ac.uk/~pszgss/Using_Model_Class_Reliance_to_measure_group_effect_on_adherence_to_asthma_medication.pdf) | [Proceedings](https://ieeexplore.ieee.org/abstract/document/9671559) 

## Installation
Install for use via pip:

**NOTE:** Currently you MUST use `cython<3`. I.e. run `pip install cython<3` to either install or downgrade before pip installing mcrforest.

**NOTE:** If you get a `ValueError: Multi-dimensional indexing is not longer supported` error when plotting you may need to downgrade matplotlib to v3.6.0. 
```
pip install git+https://github.com/gavin-s-smith/mcrforest
```
Reinstall via pip:
```
pip install --upgrade --no-cache-dir git+https://github.com/gavin-s-smith/mcrforest
```

Install for developing / debugging from unzipped source:
```
python setup.py build_ext --inplace
```

## Usage

mcrforest is an extention to the sklearn RandomForestClassifier and RandomForestRegressor classes and can be used as a direct replacement.

mcrforest includes an additional method which can be called after training a model. In addition there are two restrictions on the model building that must be met.
1. bootstrap must be set to false
2. when using a RandomForestClassifier currently only binary classification is supported and the labels must be 0,1

```
mcr(X_in, y_in, indices_to_permute, num_times = 100, mcr_type = 1, seed = 13111985)
```

Computes and MCR+ or MCR- score of a variable or group of variables. Low level function to MCR. 
See plot_mcr(...) for high level function that is often more useful in practice.

*Parameters:*
```
X_in (2D numpy array): The input features to use to compute the MCR.
y_in (1D numpy array): The output features to use to compute the MCR.
indices_to_permute (1D numpy array): A numpy array of the index/indices indicating the variable or group of variables to compute MCR for.
num_times (int): The number of times to permute the index/indices.
mcr_type (int): 1 for MCR+, -1 for MCR-.
seed (int): A seed to control the permutation randomness.
```


### Example Usage A
```
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mcrforest.forest import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from mcrforest.Datasets import get_demo_dataset

# Load data
# If loading from csv, use something like
# X_train = pd.read_csv('X_train.csv')
# y_train = pd.read_csv('X_train.csv').values.ravel()
X_train, y_train = get_demo_dataset()

# If we are going to use the training set for computing MCR then we MUST ensure 
# we have controlled the complexity of the fit. This is equally true for traditional
# permutation importance.
# See: https://christophm.github.io/interpretable-ml-book/feature-importance.html

base_model = RandomForestRegressor(random_state = 13111985, bootstrap=False)

search = {'n_estimators':[500],'max_features':['auto'], 'max_depth':[5,10,15,20,30]}

rf_cv_model = GridSearchCV(base_model, search)
rf_cv_model.fit(X_train,y_train)

# Refit with best parameters
model = RandomForestRegressor( **rf_cv_model.best_params_ )
model.fit(X_train, y_train)
model.plot_mcr(X_train, y_train)

```

The documentation for plot_mcr is:
```
Method of an mcrforest.forest.RandomForestRegressor or  mcrforest.forest.RandomForestClassifier

plot_mcr(X_in, y_in, feature_names = None, feature_groups_of_interest = 'all individual features', num_times = 100, show_fig = True)

Compute the required information for an MCR plot and optionally display the MCR plot. 
Groups of variables may be specified in which the Group-MCR extention will be used.

Parameters
----------
X_in : {numpy array or Pandas DataFrame} of shape (n_samples, n_features)
    The input samples. 
y_in : {numpy array or Pandas DataFrame} of shape (n_samples)
    The output values.
feature_names : {array-like} of shape (n_features)
    A list or array of the feature names. If None and a DataFrame is passed the feature names will be taken from the DataFrame else features will be named using numbers.
feature_groups_of_interest : {str or numpy array of numpy arrays}
    Either:
    1. 'all individual features': compute the MCR+/- for all features individually. Equvilent to: [[x] for x in range(len(feature_names))]
    2. A numpy array where each element is a numpy array of variable indexes (group of variables) which will be jointly permuated (i.e. these indexes will be considered a single unit of analysis for MCR)
       A single MCR+ and single MCR- score (plotted as a single bar in the graph) will be computed for each sub-array.
num_times : int
        The number of permutations to use when computing the MCR.
show_fig : bool
        If True show the MCR graph. In either case a dataframe with the information that would have been shown in the graph is returned.
pdf_file: str
                If not None, a path to save a pdf of the graph to.
Returns
-------
rf_results2 : {pandas DataFrame} of shape (2*[number_of_features OR len(feature_groups_of_interest)], 3)
        A DataFrame with three columns: ['variable', 'MCR+', 'MCR-']
        Where the column variable contains the variable name and the columns MCR+ and MCR- contain the variable's MCR+ and MCR- scores respectively.
```


### Example Usage B
Example Usage B shows how to compute the MCR+/- values and then plot them without the help of `plot_mcr(.)`.
```
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mcrforest.forest import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from mcrforest.Datasets import get_demo_dataset

# Load data
# If loading from csv, use something like
# X_train = pd.read_csv('X_train.csv')
# y_train = pd.read_csv('X_train.csv').values.ravel()
X_train, y_train = get_demo_dataset()

# If we are going to use the training set for computing MCR then we MUST ensure 
# we have controlled the complexity of the fit. This is equally true for traditional
# permutation importance.
# See: https://christophm.github.io/interpretable-ml-book/feature-importance.html

base_model = RandomForestRegressor(random_state = 13111985, bootstrap=False)

search = {'n_estimators':[500],'max_features':['auto'], 'max_depth':[5,10,15,20,30]}

rf_cv_model = GridSearchCV(base_model, search)
rf_cv_model.fit(X_train,y_train)

# Refit with best parameters
model = RandomForestRegressor( **rf_cv_model.best_params_ )

model.fit(X_train,y_train)

# Compute MCR+ for each variable
results = []
groups_of_indicies_to_permute = [[x] for x in range(len(X_train.columns))]

for gp in groups_of_indicies_to_permute:
    rn = model.mcr(X_train,y_train, np.asarray(gp) ,  num_times = 20, mcr_type = 1)
    results.append([','.join([list(X_train.columns)[x] for x in gp]), 'RF-MCR+', rn])


# Compute MCR- for each variable
for gp in groups_of_indicies_to_permute:
    rn = model.mcr(X_train,y_train, np.asarray(gp) ,  num_times = 20,  mcr_type = -1)
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

### Example output plot if using the code above
![Example image](http://cs.nott.ac.uk/~pszgss/research/mcr/examplemcr.png)

## Common errors
`AttributeError: 'RandomForestRegressor' object has no attribute '_validate_data'`
mcrforest requires sklearn version >= 0.23. Ensure you reinstall mcrforest after upgrading.

### Windows Only
`Cannot open include file: 'basetsd.h': No such file or directory`
You need to ensure you have the correct compilers for Windows installed since the package uses Cython.
Install the correct compilers for your version of python from: https://wiki.python.org/moin/WindowsCompilers


### Replication of results from our 2020 NeurIPS paper

*Smith, G., Mansilla, R. and Goulding, J. "Model Class Reliance for Random Forests". 34th Conference on Neural Information Processing Systems (NeurIPS 2020), Vancouver, Canada.*

Synthetic Experiments:
https://colab.research.google.com/drive/1UuORvqSYW14eiBX3nzz2WWUrjQAXcvFw

COMPAS Experiments:
https://colab.research.google.com/drive/1-hWJ4DNOnvrLz4fxGd--NJjGHV26TIei

Breast Cancer Experiments
https://colab.research.google.com/drive/16HGlytaraR6Kn4EmqKk0_Q9Nl7O_hU5F

RF-MCR Analysis:
https://colab.research.google.com/drive/1AMDW9Ss69QEzgBkMgx8Tw_zIpcZnMcr4

The above code was run with the following version:
pip install git+https://github.com/gavin-s-smith/mcrforest@NeurIPS
