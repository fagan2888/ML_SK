import numpy as np
import os,sys
import re

import ML_SK.parser.common

# TODO update this for molecular properties

def parse_data(molec_file,property_file,parsed_folder,descriptor,\
               atomic_cut_off,larger_mol,load_parsed):
  """xxx."""
  if os.path.isfile(parsed_folder+'parsed_data_'+'Global.X') \
                    and load_parsed==True:
     return ML_SK.parser.common.load_parsed_data(parsed_folder,'Global')
  else:
    print 'Parsing data'
    X,y = ML_SK.parser.common.parse_molecular_data(molec_file, property_file,\
                        descriptor, atomic_cut_off, larger_mol)

    print "- Parced",len(y[1:]), "molecules"

    if load_parsed==False: #and dev_mode==False:
      ML_SK.parser.common.save_parsed_data(X,y,parsed_folder,'Global')
      print '\nThe parsed data have been saved.'
    else:
      print '(*) parced_data file is not saved\n'

  return (X,y)
