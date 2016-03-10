
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
