import numpy as np
import os,sys
from ML_SK.descriptors import CoulombMatrix as cm
import re

from ML_SK.descriptors import Obj_CM


def grep_elem(elem,datafile):
  """Tells how many atoms of element 'elem' are in datafile"""
  c=0
  for line in datafile:
    if re.search(elem,line):
      c+=1
  return c


def atomic_cut_off2cut_off(atomic_cut_off):
  """xxx"""
  cut_off=atomic_cut_off*atomic_cut_off+atomic_cut_off
  return cut_off/2


def parse_data(data_file,parsed_folder,target_element,descriptor,\
               atomic_cut_off,larger_mol,load_parsed):
  if os.path.isfile(parsed_folder+'parsed_data_'+target_element+'.X') and load_parsed==True:
    Xload=open(parsed_folder+'parsed_data_'+target_element+'.X','r')
    X=np.load(Xload)
    Xload.close()

    yload=open(parsed_folder+'parsed_data_'+target_element+'.y','r')
    y=np.load(yload)
    yload.close()

    print 'Loading a parsed dataset\n'
    print 'Found', X.shape[0]-1, target_element, 'atoms'
    print 'Coulomb matrix with',X.shape[1],'elements.'
  else:
    print 'Parsing data'
    f=open(data_file,'r')
    lf=f.readlines()
    f.close()
    
    Ndata=grep_elem(target_element,lf)
    print 'Found', Ndata, target_element, 'atoms'


    

    # This overrides cut_off with atomic_cut_off
    if isinstance(atomic_cut_off, int): cut_off=atomic_cut_off2cut_off(atomic_cut_off)
    else:                               cut_off=atomic_cut_off2cut_off(larger_mol)

    #Initializing X with one 1D array of zeros
    X=np.zeros([Ndata+1,cut_off])
    y=np.zeros( Ndata+1 )
    #y=[0]
    print '\n(*) Creating Coulomb matrices with',X.shape[1],'elements.\n'
    
    if descriptor=='RadialCoulombMatrix': DESCR=Obj_CM.CM_Radial()
    if descriptor=='AtomicCoulombMatrix': DESCR=Obj_CM.CM_Atomic()
    
    line=0; nmol=0;
    while(line<len(lf)):
      nat=int(lf[line])
      atomic_data_str=np.array([lf[line+i+2].split() for i in range(nat)])

      cmatrix_of_selected_element=DESCR.f(atomic_data_str,target_element,atomic_cut_off)
      #cmatrix_of_selected_element=cm.cmatrix_radial(atomic_data_str,target_element,atomic_cut_off)
      #cmatrix_of_selected_element=cm.cmatrix_atomic(atomic_data_str,target_element,int(atomic_cut_off))
      
      for i,j in enumerate(cm.elem_index(atomic_data_str,target_element)):
        y[nmol+i+1]=float(atomic_data_str[j][4])
      
      cmshape=cmatrix_of_selected_element.shape
      for i in range(cmshape[0]):
	X[nmol+1,:cmatrix_of_selected_element[i].shape[0]]=cmatrix_of_selected_element[i]
	#Printing update of the parsing
	sys.stdout.write("\r[%d] " % (nmol+1))
        sys.stdout.flush()
	nmol+=1
        #Note that X[0] keeps being [0,0,0,0,...] always

      #for j in cm.elem_index(atomic_data_str,target_element):
      #  y.append(float(atomic_data_str[j][4]))
      line+=nat+2

    print "- Parced",len(y[1:]),target_element, "atoms"

    if load_parsed==False: #and dev_mode==False:
      Xsave=open(parsed_folder+'parsed_data_'+target_element+'.X','w')
      np.save(Xsave,X)
      Xsave.close()

      ysave=open(parsed_folder+'parsed_data_'+target_element+'.y','w')
      np.save(ysave,y)
      ysave.close()

      print '\nThe parsed data have been saved.'
    else:
      print '(*) parced_data file is not saved\n'

  return (X,y)





def save_prediction(y,ypred,model,train_size):
  print '\nSaving results to file'
  s=open('result.dat','w')
  print >>s, '# Training set',train_size,'\n'\
             '# Predictions ',len(ypred)
  for i,j in enumerate(y[train_size:]): print >>s, j, ypred[i]
  s.close()
