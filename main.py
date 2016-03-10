#!/usr/bin/python

from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import learning_curve
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor

import random
import numpy as np
import os,sys

import ML_SK.parser
import ML_SK.plotting.plots
import ML_SK.plotting.plotter




data_file      = '../DS/data_charges.dat'
parsed_folder  = '../DS/PARSED/Charges/'

target_element = 'O'
train_size     = 300       #in Molecules.
larger_mol     = 29       #Largest molecule in the dataset.
atomic_cut_off = 19.      #Cut_off used to parse the dataset.
cut_off_extra  = 9       #Applies a cut_off to the linear CM.
descriptor     = 'RadialCoulombMatrix'
algo           = 'KernelRidge'
kernel         = 'laplacian'
shuffle        = False

load_parsed    = True
plot           = 'mini'

# View data
fast_histo     = False    #Plot a Histogram of the dataset and exit.
plot_CM        = False    #Plot the Coulomb matrix ParseLength and exit.

# algo:
# KernelRidge, Ridge, LinearRegression, BayesianRidge
# SGDRegressor, RandomForestRegressor









cut_off_extra=ML_SK.parser.atomic_cut_off2cut_off(cut_off_extra)


if target_element == 'O': train_size= 2*train_size; nspec= 2
if target_element == 'C': train_size= 7*train_size; nspec= 7
if target_element == 'H': train_size=10*train_size; nspec=10

############################################################################
########################### Loading Dataset ################################
############################################################################

if plot_CM == True:
  ML_SK.plotting.plotter.plot_CM(data_file,target_element,2.9,6000)

parsed_data=ML_SK.parser.parse_data(data_file,parsed_folder,target_element,descriptor,\
                                 atomic_cut_off,larger_mol,load_parsed)

X=parsed_data[0][:,:cut_off_extra]
y=parsed_data[1]

if load_parsed == True:
  print '\n(*) Coulomb matrix loaded with',X.shape[1],'elements.'

# Random shuffle of the data maintaining the zero as first item of the array
if shuffle==True:
  idx=range(len(y))
  idx.remove(0)

  idm=np.array(idx).reshape(6095,nspec)
  np.random.shuffle(idm)

  idx=idm.reshape(6095*nspec).tolist()
  idx.insert(0,0)

  X=X[idx]
  y=np.array(y)[idx]
else:
  X=X
  y=np.array(y)



#Histogram of data
if fast_histo==True:
  print '\nHistogram and exit'
  ML_SK.plotting.plots.plot_histo(y[1:],'All dataset',\
                               y[1:train_size+1],'Training sets',\
                               train_size)
  exit()


############################################################################
########################## Training and Predicting #########################
############################################################################


######################## Algorithms ########################################
KRR = KernelRidge(kernel=kernel, gamma=1e-5, alpha=1e-9)
RR  = linear_model.Ridge(alpha = 0.5,solver='auto')
OLS = linear_model.LinearRegression()
BR  = linear_model.BayesianRidge()
SGD = linear_model.SGDRegressor()
RFR = RandomForestRegressor(n_estimators=100, max_depth=30)

algorithm_list  = [KRR, RR, OLS, BR, SGD, RFR]

algorithm_names = ["KernelRidge",   "Ridge",       "LinearRegression",\
                   "BayesianRidge", "SGDRegressor","RandomForestRegressor"]

model=algorithm_list[algorithm_names.index(algo)]

train_size=train_size+1 #Because the zeros on the first row of X

# Fitting
print '\nTraining set', train_size-1, target_element, 'atoms'
model.fit(X[1:train_size,:cut_off_extra],y[1:train_size])

# Predicting
print 'Predicting results for', X.shape[0]-1-train_size-1,target_element, 'atoms\n'
ypred=model.predict(X[train_size:,:cut_off_extra])

# MAE Prediction
print 'MAE       = %12.6f '  % (mean_absolute_error(y[train_size:],ypred))   ,\
      '- out of sample'




############################################################################
########################### Plotting results ###############################
############################################################################

ML_SK.plotting.plotter.plot_results(y,ypred,model,train_size,plot)

############################################################################
####################### Save predictions to file ###########################
############################################################################

ML_SK.parser.save_prediction(y,ypred,model,train_size)
