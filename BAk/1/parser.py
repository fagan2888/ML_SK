import numpy as np
import os,sys
from ML.descriptors import CoulombMatrix as cm



def parse_data(data_file,parsed_folder,target_element,\
               cut_off,load_parsed,DevMode,ParseLength):
  # If there is parsed data, load it, otherwise parse it.
  if os.path.isfile(parsed_folder+'parsed_data_'+target_element+'.X') \
    and load_parsed==True:
    Xload=open(parsed_folder+'parsed_data_'+target_element+'.X','r')
    X=np.load(Xload)
    Xload.close()

    yload=open(parsed_folder+'parsed_data_'+target_element+'.y','r')
    y=np.load(yload)
    yload.close()

    print 'Loading a parsed dataset\n'
    print 'Found', X.shape[0]-1, target_element, 'atoms'
  else: 
    print 'Parsing data'
    f=open(data_file,'r')
    lf=f.readlines()
    f.close()


    nat=int(lf[0])

    #Initializing X with one 1D array of zeros
    X=np.zeros([1,cut_off])
    y=[0]

    if DevMode==False: ParseLength=len(lf)
    
    line=0
    while(line<ParseLength):
        nat=int(lf[line])
        atomic_data_str=np.array([lf[line+i+2].split() for i in range(nat)])

        cmatrix_of_selected_element= cm.cmatrix(atomic_data_str,target_element,cut_off)
        #print 'CM:', cmatrix_of_selected_element.shape
        #print 'X: ', X.shape
        X=np.concatenate((X,cmatrix_of_selected_element))
        # Note that X[0] keeps being [0,0,0,0,...] always
        
        for i,j in enumerate(cm.elem_index(atomic_data_str,target_element)):
            y.append(float(atomic_data_str[j][4]))
            
        line+=nat+2
        
    print "Parced",len(y[1:]),target_element, "atoms"
    
    #Save the parsed data only if DevMode==False
    if load_parsed==False and DevMode==False:
      Xsave=open(parsed_folder+'parsed_data_'+target_element+'.X','w')
      np.save(Xsave,X)
      Xsave.close()
      
      ysave=open(parsed_folder+'parsed_data_'+target_element+'.y','w')
      np.save(ysave,y)
      ysave.close()
      
      print "Parsed dataset saved."
    else:
      if DevMode==True:
	print '(*) parced_data file is not saved in developer mode\n'

  return (X,y)
