#============= Import libraries ==============#
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from matplotlib.text import Text
from matplotlib.widgets import Button
from matplotlib.widgets import TextBox
import seaborn as sns
import matplotlib.pyplot as plt
from mcrforest.forest import RandomForestClassifier
from pgmpy.sampling.Sampling import BayesianModelSampling
from pgmpy.models import BayesianModel
import tkinter as tk
from pgmpy.factors.discrete import TabularCPD

#============= Import Dataset ==============#
# Defining the model structure. We can define the network by just passing a list of edges.
model_structure = [ ('A','Y'), ('B','Y'), ('B','C')]
model = BayesianModel( model_structure )

# Defining individual CPDs.
cpd_a = TabularCPD(variable='A', variable_card=2, values=[[0.5], [0.5]])
cpd_b = TabularCPD(variable='B', variable_card=2, values=[[0.5], [0.5]])
cpd_c = TabularCPD(variable='C', variable_card=2, values=[[0.5], [0.5]])


cpd_bc = TabularCPD(variable='C', variable_card=2, values=[[1,0], [0,1]], evidence=['B'], evidence_card=[2])

cpd_y = TabularCPD(variable='Y', variable_card=2, values=[[1,0,0,1], [0,1,1,0]], evidence=['A','B'], evidence_card=[2,2])

# Associating the CPDs with the network
model.add_cpds(cpd_a, cpd_b, cpd_bc, cpd_y)

# check_model checks for the network structure and CPDs and verifies that the CPDs are correctly 
# defined and sum to 1.
model.check_model()

# Define data (randomly)
sampler = BayesianModelSampling(model)

out = sampler.forward_sample(size=1000, return_type='dataframe') 

y_train = pd.DataFrame(out.Y)
y = y_train.copy()
X_train = pd.DataFrame(out.drop(columns = ['Y']))
X = X_train.copy()

debug_call = False

#============= Define MCR Random Forest Function ==============#
def MCR_Random_Forest(X,y):
    params = {'bootstrap': False, 'ccp_alpha': 0.0, 'max_depth': None, 'criterion':'gini', 'max_features': 1, 'max_leaf_nodes': None, 'max_samples': None, 'mcr_tree_equivilient_tol': 0.1, 'min_impurity_decrease': 0.01, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 500, 'n_jobs': None, 'oob_score': False, 'performance_equivilence': True, 'random_state': 13111985, 'spoof_as_sklearn': False, 'verbose': 0, 'warm_start': False}
    rf_best_model = RandomForestClassifier(**params)
    rf_best_model.fit(X.values, y.values.flatten())
     
    #============= Create Datframe with MCR- & MCR+ ==============#
    variables = []
    mcrp = []
    mcrm = []
    for i, c in enumerate(X.columns):
        variables.append(c)
        mcr_p = rf_best_model.mcr( X.values,y.values.flatten(), np.asarray([i]), 
                            num_times = 20, debug_call = debug_call, mcr_type = 1, mcr_as_ratio=False)
        mcr_m = rf_best_model.mcr( X.values,y.values.flatten(), np.asarray([i]), 
                            num_times = 20, debug_call = debug_call, mcr_type = -1, mcr_as_ratio=False)
     
        mcrm.append(mcr_m)
        mcrp.append(mcr_p)
    rf_results = pd.DataFrame({'variable':variables, 'MCR+':mcrp, 'MCR-':mcrm})
    return rf_results, rf_best_model


#============= Define MCR Plot function ==============#
def plot_mcr_test(df_in, fig_size = (7, 4)):
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
        root.geometry('500x280+80+150') #to give size of the pop-up windown (500x200) and the location in the sceen (600+300)
        root.title('MCR Alternative Explainations')
        root.configure(background='white')
        
        #============= INTERNAL FUNCTIONS ================================================#
        #=== USE function ===#
        def use():
            history.append([text.get_text(),True])
            model.set_estimators(True, X.columns.tolist().index(text.get_text()), debug = True )
            print(history)
        #=== AVOID function ===# 
        def avoid():
            history.append([text.get_text(),False])
            model.set_estimators(False, X.columns.tolist().index(text.get_text()), debug = True )
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
            def compute_mcr_based_on_clicked_list(model, X,y, hist, mcr_plus):
                
                mcr_values = []
                variables = []
                new_estimators = []
                
                for cidx, c in enumerate(X.columns):
                    
                    must_use_variable_ordering = []
                # all variable not in history or the current variable
                    #all_other_vars_as_list = [ jidx for jidx, j in enumerate(X.columns) if jidx not in [cidx] and j not in [x[0] for x in hist] ] #esta es una lista con las variables que no estan en cada iteracion del for loop en la lista historica

                    all_other_vars_as_list = []
                    for jidx, j in enumerate(X.columns):
                        if jidx not in [cidx] and j not in [x[0] for x in hist]:
                            all_other_vars_as_list.append( jidx )

                    # For MCR-/+ Build must_use_variable_ordering for this variable and the history
                    for e in hist:
                        if e[1] == True: # must use
                            index = list(X.columns).index(e[0])
                            must_use_variable_ordering.append(index)

                    if mcr_plus == 1:
                        if c not in [x[0] for x in hist]: # if the variable of interest is already in the history it has it's fixed place in the must_use_variable_ordering
                            index = list(X.columns).index(c)
                            must_use_variable_ordering.append(index)

                        must_use_variable_ordering += all_other_vars_as_list
                    else:
                        must_use_variable_ordering += all_other_vars_as_list

                        if c not in [x[0] for x in hist]: # if the variable of interest is already in the history it has it's fixed place in the must_use_variable_ordering
                            index = list(X.columns).index(c)
                            must_use_variable_ordering.append(index)

                    for e in hist[::-1]:
                        if e[1] == False: 
                            index = list(X.columns).index(e[0])
                            must_use_variable_ordering.append(index) # append variable you want to avoid using

                    # convert must_use_variable_ordering into a 1D numpy array to run mcr_ordering
                    must_use_variable_ordering = np.array(must_use_variable_ordering)
                    
                    #===================== Run MCR with built must_use_variable_ordering ========#
                    r = model.mcr(X.values,y.values.flatten(), np.asarray([cidx]), mcr_ordering = must_use_variable_ordering, num_times = 20, debug_call = debug_call, mcr_type = mcr_plus, mcr_as_ratio=False)

                    
                    # Convert must_use_variable_ordering into a list to continue appending values
                    must_use_variable_ordering = list(must_use_variable_ordering)
                    # Append variables and mcr values to their respective lists
                    mcr_values.append(r)
                    
                    variables.append(c)   
                print(f'MCR ({mcr_plus}): ',mcr_values)     
                    
                return mcr_values,variables
            
            mcr_p, var_p = compute_mcr_based_on_clicked_list(model = model, X = X_train, y=y_train, hist = history, mcr_plus = 1 ) # mcr_p = [0.41, 0.10, 0.92, 0.82], var_p = ['A', 'B', 'C', 'D']
            mcr_m, var_m = compute_mcr_based_on_clicked_list(model = model, X = X_train, y=y_train, hist = history, mcr_plus = -1 ) # mcr_p = [0.22, 0.06, 0.80, 0.16], var_p = ['A', 'B', 'C', 'D']
            
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
mcr_results, model = MCR_Random_Forest(X_train,y_train) # Input data in a dataframe type

# Plot MCR graph
print(mcr_results)
plot_mcr_test(mcr_results)
print(list(mcr_results['MCR+']))
print(list(mcr_results['MCR-']))