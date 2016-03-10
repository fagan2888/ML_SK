
import numpy as np
import os,sys
from ML.descriptors import CoulombMatrix as cm
import ML.parser
from ML.plots import plots


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

