
import numpy as np
import os,sys
from ML_SK.descriptors import CoulombMatrix as cm
import ML_SK.parser
import ML_SK.plotting


############################################################################
########################## Some predefined plots ###########################
############################################################################


#def plot_results(y,ypred,model,train_size,plot):
#  if plot=='off':
#    plot_pred = False; plot_coeff = False; plot_histo = False
#  if plot=='mini':
#    plot_pred = True;  plot_coeff = False; plot_histo = False
#  if plot=='all':
#    plot_pred = True;  plot_coeff = True; plot_histo = True
#
#  if plot_pred==True:
#    ML_SK.plotting.plots.common_plot(y[train_size:],ypred,'Title',\
#                                  diag='diag',prediction=True)
#
#  if plot_coeff==True:
#    if hasattr(model, 'dual_coef_'):
#      coeffs=np.array(model.dual_coef_)
#    else:
#      coeffs=np.array(model.coef_)
#    coeffs.sort()
#    ML_SK.plotting.plots.common_plot(np.arange(len(coeffs)),coeffs,'Weights')
#
#  if plot_histo==True:
#    ML_SK.plotting.plots.plot_histo(y[1:],'Full dataset',\
#                                 y[1:train_size],'Training set',\
#                                 train_size)
