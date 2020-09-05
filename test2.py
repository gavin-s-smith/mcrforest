import pandas as pd
from mcrforest.forest import RandomForestRegressor,DecisionTreeRegressor
import numpy as np
from tqdm import tqdm

X_train = pd.read_csv('X_train.csv')#.head(3)
y_train = pd.read_csv('y_train.csv').values.ravel()#.head(3)



#rfd = DecisionTreeRegressor(random_state = 13111985)
rfd = RandomForestRegressor(n_estimators = 100,random_state = 13111985, bootstrap = False,max_features=1)


#rfd.fit(X_train.iloc[:,[111,38]],y_train)

rfd.fit(X_train,y_train)

# rfd.print_trees(X_train.columns[[111,38]])

# exit(-1)

groups_of_indicies_to_permute = np.asarray([[x] for x in list(range(len(X_train.columns)))])
results = []
for gp in tqdm(groups_of_indicies_to_permute):
    rn = rfd.mcr(X_train.values,y_train, gp ,  num_times = 10,  mcr_type = 1, mcr_as_ratio = True)
    results.append([','.join([list(X_train.columns)[x] for x in gp]), 'RF-MCR+', rn])
    rn = rfd.mcr(X_train.values,y_train, gp ,  num_times = 10,  mcr_type = -1, mcr_as_ratio = True)
    results.append([','.join([list(X_train.columns)[x] for x in gp]), 'RF-MCR-', rn])
lbl = [ x[0] for x in results if 'MCR+' in x[1] ]
mcrp = [ x[2] for x in results if 'MCR+' in x[1] ]
mcrm = [ x[2] for x in results if 'MCR-' in x[1] ]
rf_results2 = pd.DataFrame({'variable':lbl, 'MCR+':mcrp, 'MCR-':mcrm})
import seaborn as sns
import matplotlib.pyplot as plt

def plot_mcr(df_in, fig_size = (11.7, 8.27)):
    df_in = df_in.copy()
    df_in.columns = [ x.replace('MCR+', 'MCR- (lollypops) | MCR+ (bars)') for x in df_in.columns]
    ax = sns.barplot(x='MCR- (lollypops) | MCR+ (bars)',y='variable',data=df_in)
    plt.gcf().set_size_inches(fig_size)
    plt.hlines(y=range(df_in.shape[0]), xmin=0, xmax=df_in['MCR-'], color='skyblue')
    plt.plot(df_in['MCR-'], range(df_in.shape[0]), "o", color = 'skyblue')

rf_results2.to_csv('mcr.csv')

plot_mcr(rf_results2)

plt.show()