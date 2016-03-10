
import numpy as np
import os,sys
from ML.descriptors import CoulombMatrix as cm
import ML.parser
from ML.plotting import plots


def plot_CM(data_file,target_element,cut_off,ParseLength):
  """Parse data until the line line 'ParseLength' and show the
     Coulomb matrix of the corresponding molecule at this point
     of the file. Then exit.
  """
  #Open data file to look for a molecule
  f=open(data_file,'r')
  lf=f.readlines()
  f.close()

  line=0; nmol=0
  while(line<ParseLength):
    nat=int(lf[line])
    atomic_data_str=np.array([lf[line+i+2].split() for i in range(nat)])

    line+=nat+2
    nmol+=1

  print nat
  print 'Molecule',nmol
  for i in atomic_data_str[:,:4]:
    print "%s %12.6f %12.6f %12.6f" % (i[0],float(i[1]),float(i[2]),float(i[3]))

  pcm=cm.cmatrix(atomic_data_str,target_element,cut_off)
  plots.plot_matrix(pcm[0])
  exit()




def plot_results(y,ypred,model,train_size,plot):
  if plot=='off':
    plot_pred = False; plot_coeff = False; plot_histo = False
  if plot=='mini':
    plot_pred = True;  plot_coeff = False; plot_histo = False
  if plot=='all':  
    plot_pred = True;  plot_coeff = True; plot_histo = True
  
  if plot_pred==True:
    ML.plotting.plots.common_plot(y[train_size:],ypred,'Title',\
                                  diag='diag',prediction=True)
  
  if plot_coeff==True:
    if hasattr(model, 'dual_coef_'):
      coeffs=np.array(model.dual_coef_)
    else:
      coeffs=np.array(model.coef_)
    coeffs.sort()
    ML.plotting.plots.common_plot(np.arange(len(coeffs)),coeffs,'Weights')
 
  if plot_histo==True:
    ML.plotting.plots.plot_histo(y[1:],'Full dataset',\
                                 y[1:train_size],'Training set',\
                                 train_size)

