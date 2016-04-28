#import numpy as np
#import os,sys
#from ML_SK.descriptors import CoulombMatrix as cm
#import re
#
#from ML_SK.descriptors import Obj_CM
#
#
#def grep_elem(elem,datafile):
#  """Tells how many atoms of element 'elem' are in datafile"""
#  c=0
#  for line in datafile:
#    if re.search(elem,line):
#      c+=1
#  return c
#
## TODO save as pandas?
#def save_prediction(y,ypred,model,train_size):
#  print '\nSaving results to file'
#  s=open('result.dat','w')
#  print >>s, '# Training set',train_size,'\n'\
#             '# Predictions ',len(ypred)
#  for i,j in enumerate(y[train_size:]): print >>s, j, ypred[i]
#  s.close()
