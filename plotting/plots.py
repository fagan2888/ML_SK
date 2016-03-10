
import numpy as np
import matplotlib.pyplot as plt

def plot_matrix(data,label=None):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  cax = ax.matshow(data)
  fig.colorbar(cax)

  #ax.set_xticklabels(label)
  #ax.set_yticklabels(label)
  plt.show()


def plot_histo(x,str_x,y,str_y,train_size):
  '''Plots a two-panel histogram'''
  bins = np.linspace(x.min(), x.max(), 140)
  plt.figure(1)
  plt.subplot(211); plt.title(str_x); plt.hist(x,bins); plt.xlim(([x.min(),x.max()]))
  plt.subplot(212); plt.title(str_y); plt.hist(y,bins); plt.xlim(([x.min(),x.max()]))
  plt.show()


def common_plot(x,y,str_x,diag=None,prediction=False):
  plt.title(str_x)

  plt.plot(x,y,'bo')#,diagonal,diagonal,'r')

  if diag=='diag':
    diagonal=np.linspace(x.min(),x.max(), 10)
    plt.plot(diagonal,diagonal,'r')

  plt.xlim([x.min(),x.max()])
  if prediction==False:
    plt.ylim([y.min(),y.max()])
  else:
    plt.ylim([x.min(),x.max()])
  plt.show()
