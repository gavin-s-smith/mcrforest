#============= Import libraries ==============#
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from IPython.display import clear_output
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.widgets import Button
from matplotlib.widgets import TextBox
import seaborn as sns
import matplotlib.pyplot as plt
from mcrforest.forest import RandomForestRegressor
from IPython.display import clear_output
import tkinter as tk

#============= Import Dataset ==============#
# This is compas_X.csv in the supplementary zip file
X = pd.read_csv('compas_X.csv') 
# This is compas_y.csv in the supplementary zip file
y = pd.read_csv('compas_y.csv',header=0,names=['Y']) 
# This is compas_train_indices.csv in the supplementary zip file
train_bool_mask = pd.read_csv('compas_train_indices.csv').values.flatten()

X_train = X[train_bool_mask]
y_train = y[train_bool_mask]
X_test = X[~train_bool_mask]
y_test = y[~train_bool_mask]

#============= Define MCR Random Forest Function ==============#
def MCR_Random_Forest(X,y):
    # For the model class RF
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # For the model class RF
    search = {'n_estimators':[10],'min_impurity_decrease':[0.001,0.01],'max_features':[1,'auto']}
    rf_cv_model = GridSearchCV(RandomForestRegressor(mcr_tree_equivilient_tol=3.1713, bootstrap=False, 
                                                criterion='mse', random_state=13111985), search, cv = kf, refit = True)
    rf_best_model = rf_cv_model.fit(X, y.values.flatten()).best_estimator_
    #============= Create Datframe with MCR- & MCR+ ==============#
    variables = []
    mcrp = []
    mcrm = []
    for i, c in enumerate(X.columns):
        variables.append(c)
        mcr_p = rf_best_model.mcr( X.values,y.values.flatten(), np.asarray([i]), 
                            num_times = 2, debug = False, mcr_type = 1, mcr_as_ratio=True)

        mcr_m = rf_best_model.mcr( X.values,y.values.flatten(), np.asarray([i]), 
                            num_times = 2, debug = False, mcr_type = -1, mcr_as_ratio=True)
        clear_output()
        mcrm.append(mcr_m)
        mcrp.append(mcr_p)

    rf_results = pd.DataFrame({'variable':variables, 'MCR+':mcrp, 'MCR-':mcrm})
    return rf_results

#============= Define MCR Plot function ==============#
def plot_mcr_test(df_in,fig_size = (7, 4)):
    plt.style.use('ggplot')
    df_in = df_in.copy() # create a copy of the dataframe
    df_in.columns = [ x.replace('MCR+', 'MCR- (lollypops) | MCR+ (bars)') for x in df_in.columns] # Change column MCR+ name for MCR- (lollypops) | MCR+ (bars)

    # Set bar graph parameters
    fig, ax = plt.subplots() 
    plt.gcf().set_size_inches(fig_size)
    ax.set_title('Click on variables to explore alternative explainations', fontsize=12)
    y_pos = np.arange(len(df_in))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_in['variable'])
    ax.set_xlabel('MCR- (lollypops) | MCR+ (bars)', fontsize=11)
    ax.set_ylabel('Variables')
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlim(0, mcr_results['MCR+'].max() + 1) #fix x limits between 0 and 6
    # Generate the bar graph
    ax.barh(y_pos, df_in['MCR- (lollypops) | MCR+ (bars)'], color=plt.get_cmap("Set3").colors, picker=True) # Picker=True to make bars pickable
    for label in ax.get_yticklabels():  # make the ytick labels pickable
        label.set_picker(True)
      
    # Generate lollypops graph
    plt.hlines(y=range(df_in.shape[0]), xmin=0, xmax=df_in['MCR-'], color='dimgrey')
    plt.plot(df_in['MCR-'], range(df_in.shape[0]), "o", color = 'dimgrey')

    # Call the event
    fig.canvas.mpl_connect('pick_event', on_click)
    
    # Display the graph
    plt.tight_layout()
    plt.show()

#============= List of list joinning function ==============#
def join_mcr_var(mcr,var):
  l=[]
  for x,y in zip(mcr, var):
    l.append([y,x])
  return l
#============= MCR function with built must_use_variable_ordering ==============#
def mcr():
    return np.random.rand(1)[0] #generate a random float between 0 and 1

#============= Event on_click function ==============#
history = [] 

def on_click(event):
    plt.gcf().canvas.draw_idle()

    if isinstance(event.artist, Text): # If what the user clicks is one of the y-label variables it execute the code below
        text = event.artist 
        variable = text.get_text()
        print('You click the variable:', text.get_text())
        
        #============== POP-UP WINDOW ====================================================#       
        root  = tk.Tk()
        root.geometry('500x280+850+150') #to give size of the pop-up windown (500x200) and the location in the sceen (600+300)
        root.title('MCR Alternative Explainations')
        root.configure(background='white')
        
        #============= INTERNAL FUNCTIONS ================================================#
        #=== USE function ===#
        def use():
            history.append([text.get_text(),'True'])
            print(history)
        #=== AVOID function ===# 
        def avoid():
            history.append([text.get_text(),'Fasle'])
            print(history)
        #=== CLEAR function ===#     
        def clear_history():
            history.clear() 
            print('History list is clear: ',history)    
        #=== SHOW HISTORY function ===#       
        def show_history(): 
            print(history) 
        #=== UNDO function ===# 
        def undo(): 
            history.pop()
            print(history) 
        #=== RUN MCR with must_use_variable_ordering ===# 
        def run_mcr():
            def compute_mcr_based_on_clicked_list(df, hist, mcr_plus):
                must_use_variable_ordering = []
                mcr_values = []
                variables = []
                
                for cidx, c in enumerate(df.columns):
                # all variable not in history or the current variable
                    all_other_vars_as_list = [ jidx for jidx, j in enumerate(df.columns) if jidx not in [cidx] and j not in [x[0] for x in hist] ] #esta es una lista con las variables que no estan en cada iteracion del for loop en la lista historica
                    
                    # For MCR-/+ Build must_use_variable_ordering for this variable and the history
                    for e in hist:
                        if e[1] == True: # must use
                            must_use_variable_ordering.append(e[0])

                    if mcr_plus == True:
                        if c not in [x[0] for x in hist]: # if the variable of interest is already in the history it has it's fixed place in the must_use_variable_ordering
                            must_use_variable_ordering.append(c)

                        must_use_variable_ordering += all_other_vars_as_list
                    else:
                        must_use_variable_ordering += all_other_vars_as_list

                        if c not in [x[0] for x in hist]: # if the variable of interest is already in the history it has it's fixed place in the must_use_variable_ordering
                            must_use_variable_ordering.append(c)

                    for e in history[::-1]:
                        if e[1] == False: 
                            must_use_variable_ordering.append(e[0]) # append variable you want to avoid using

                    # Call mcr function with built must_use_variable_ordering
                    r = mcr()
                    # Append variables and mcr values to their respective lists
                    mcr_values.append(r)
                    variables.append(c)

                return mcr_values,variables
            
            mcr_p,var_p = compute_mcr_based_on_clicked_list(df = X_train, hist = history, mcr_plus = True ) # mcr_p = [0.41, 0.10, 0.92, 0.82], var_p = ['A', 'B', 'C', 'D']
            mcr_m,var_m = compute_mcr_based_on_clicked_list(df = X_train, hist = history, mcr_plus = False ) # mcr_p = [0.22, 0.06, 0.80, 0.16], var_p = ['A', 'B', 'C', 'D']
            
            #=== Transform list of list into dataframe ===#
            new_mcr_results = pd.DataFrame({'variable':var_p, 'MCR+':mcr_p, 'MCR-':mcr_m})
            #=== Plot graph with new MCR+/- values ===#
            plot_mcr_test(new_mcr_results)
            #=== Close pop-up window after running mcr model ===#
            root.destroy()      
        
        #=== Instructions ===#
        tk.Label(root, text=f'Would you like to "USE" or "AVOID" variable "{variable}"', 
        font=('Arial Bold',10), bg='white',fg='darkslategrey').place(x=5,y=20)     
        #=== "USE" Button ===#
        tk.Button(root, text='USE',activebackground='silver',fg='darkslategrey', width = 12, height = 1 , command=use).place(x=70,y=60)

        #=== "AVOID" Button ===#
        tk.Button(root, text='AVOID',activebackground='silver',fg='darkslategrey', width = 12, height = 1 , command=avoid).place(x=310,y=60)
        #=== "SELECT OTHER VARIABLE" Button ===#
        tk.Button(root, text='Select Another Variable',activebackground='silver',fg='darkslategrey', width = 19, height = 1 , command=root.destroy).place(x=160,y=110)
        #=== "SHOW HISTORY" Button ===#
        tk.Button(root, text='Show History',activebackground='silver',fg='darkslategrey', width = 14, height = 1 , command=show_history).place(x=40,y=170)
        #=== "UNDO" Button ===#
        tk.Button(root, text='Undo',activebackground='silver',fg='darkslategrey', width = 10, height = 1 , command=undo).place(x=195,y=170)
        #=== "CLEAR HISTORY" Button ===#
        tk.Button(root, text='Clear History',activebackground='silver',fg='darkslategrey', width = 14, height = 1 , command=clear_history).place(x=320,y=170)
        #=== "RUM MCR" Button ===#
        tk.Button(root, text='Run MCR',activebackground='silver',fg='darkslategrey', width = 12, height = 1 , command=run_mcr).place(x=187,y=230)
        #=== Main loop ===#
        root.resizable(0,0)
        root.mainloop()

# Testing MCR Random Forest
mcr_results = MCR_Random_Forest(X_train,y_train) # Input data in a dataframe type

# Plot MCR graph
plot_mcr_test(mcr_results)