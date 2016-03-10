#!/usr/bin/python

from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import learning_curve
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor

import random
import matplotlib.pyplot as plt
import numpy as np
import os,sys

import ML.parser
from ML.plots import plotter




data_file      = 'data_GDB9_atomicenergies.dat'
parsed_folder  = 'DATASETS/GDB9_partial_energies/'


target_element = 'O'
train_size     = 100 #Molecules
atomic_cut_off = 10.
cut_off        = 190
algo           = 'KernelRidge'
kernel         = 'laplacian'
shuffle        = False

load_parsed    = False
DevMode        = False
ParseLength    = 6000
save_results   = True
plot           = 'mini'
plot_pred      = True
plot_large     = True
plot_coeff     = True
plot_histo     = True
fast_histo     = False     #Plot a Histogram of the dataset and exit
plot_CM        = False 

# algo:
# KernelRidge, Ridge, LinearRegression, BayesianRidge
# SGDRegressor, RandomForestRegressor







# This overrides cut_off with atomic_cut_off
#if atomic_cut_off != None:
  #cut_off=ML.parser.atomic_cut_off2cut_off(atomic_cut_off)
#else:
  #atomic_cut_off=cut_off

if isinstance(atomic_cut_off, int):
  cut_off=ML.parser.atomic_cut_off2cut_off(atomic_cut_off)
else:
  cut_off=ML.parser.atomic_cut_off2cut_off(19)


if target_element == 'O': train_size= 2*train_size; nspec= 2
if target_element == 'C': train_size= 7*train_size; nspec= 7
if target_element == 'H': train_size=10*train_size; nspec=10

############################################################################
########################### Loading Dataset ################################
############################################################################

if plot_CM == True: plotter.plot_CM(data_file,target_element,2.9,ParseLength)

parsed_data=ML.parser.parse_data(data_file,parsed_folder,target_element,
                       atomic_cut_off,load_parsed,DevMode,ParseLength)

X=parsed_data[0][:,:cut_off]
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


#Histogram of data
if fast_histo==True:
  print '\nHistogram and exit'
  bins = np.linspace(y[1:].min(), y[1:].max(), 140)
  plt.figure(1)
  plt.subplot(211); plt.title('Full dataset'); plt.hist(y[1:],bins)

  plt.subplot(212); plt.title('Training set'); plt.hist(y[1:train_size+1],bins)

  plt.show()

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

algorithm_names = ["KernelRidge", "Ridge", "LinearRegression",\
                   "BayesianRidge", "SGDRegressor","RandomForestRegressor"]

model=algorithm_list[algorithm_names.index(algo)]

train_size=train_size+1 #Because the zeros on the first row of X

# Fitting
print '\nTraining set', train_size-1, target_element, 'atoms'
model.fit(X[1:train_size,:cut_off],y[1:train_size])

# Predicting
print 'Predicting results for', X.shape[0]-1-train_size-1,target_element, 'atoms\n'
ypred=model.predict(X[train_size:,:cut_off])


# MAE Prediction
print 'MAE       = %12.6f '  % (mean_absolute_error(y[train_size:],ypred))        ,\
      '- out of sample'




############################################################################
########################### Plotting results ###############################
############################################################################

if plot=='off':
  plot_pred = False; plot_large = False; plot_coeff = False; plot_histo = False
if plot=='mini':
  plot_pred = True; plot_large = False; plot_coeff = False; plot_histo = False


if plot_pred==True:
  diagonal=np.linspace(y[1:].min(),y[1:].max(), 10)
  plt.plot(y[train_size:],ypred,'bo',diagonal,diagonal,'r')
  plt.xlim([y[1:].min(),y[1:].max()])
  plt.ylim([y[1:].min(),y[1:].max()])
  plt.show()


if plot_large==True:
  diagonal=np.linspace(y[1:].min(),y[1:].max(), 10)
  plt.figure(1)
  plt.subplot(211)
  #plt.plot(y[train_size:],ypred,'bo',pyt,pyp,'go',diagonal,diagonal,'r')
  plt.plot(y[train_size:],ypred,'bo',diagonal,diagonal,'r')
  plt.xlim([y[1:].min(),y[1:].max()])
  plt.ylim([y[1:].min(),y[1:].max()])

  plt.subplot(212)
  bins = np.linspace(y[1:].min(), y[1:].max(), 140)
  plt.xlim([y[1:].min(),y[1:].max()])
  plt.hist(y[1:train_size],bins)
  plt.show()


if plot_coeff==True:
  if hasattr(model, 'dual_coef_'):
    coeffs=model.dual_coef_
  else:
    coeffs=model.coef_
  bins = np.linspace(coeffs.min(), coeffs.max(), 140)
  plt.title('Weights')
  #plt.hist(coeffs,bins)
  plt.plot(sorted(coeffs),'bo-')
  plt.show()



if plot_histo==True:
  bins = np.linspace(y[1:].min(), y[1:].max(), 140)
  plt.figure(1)
  plt.subplot(211)
  plt.title('Full dataset')
  plt.hist(y[1:],bins)

  plt.subplot(212)
  plt.title('Training set')
  plt.hist(y[1:train_size],bins)
  plt.show()



############################################################################
####################### Save predictions to file ###########################
############################################################################
if save_results==True:
  print '\nSaving results to file'
  s=open('result.dat','w')
  print >>s, '# Training set',train_size,'\n'\
             '# Predictions ',len(ypred)
  for i,j in enumerate(y[train_size:]): print >>s, j, ypred[i]
  s.close()
