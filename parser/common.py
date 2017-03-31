import numpy as np
from ML_SK.descriptors import Obj_CM
# TODO test if dill should be used instead of cPickle
import cPickle as pickle
import time
import sys

def grep_elem(elem,datafile):
  """Tells how many atoms of element 'elem' are in datafile"""
  c = 0
  for line in datafile:
    if re.search(elem,line):
      c += 1
  return c

def load_parsed_data(out_path):
    with open(out_path) as f:
        x = pickle.load(f)
    return x


def save_parsed_data(x, out_path):
    with open(out_path, "w") as f:
        pickle.dump(x,f, protocol=-1)


##TODO update
#def save_prediction(y,ypred,model,train_size, verb=True): #TODO what is
#  if verb==True:
#    print '\nSaving results to file'
#  s=open('result.dat','w')
#  print >>s, '# Training set',train_size,'\n'\
#             '# Predictions ',len(ypred)
#  for i,j in enumerate(y[train_size:]): print >>s, j, ypred[i]
#  s.close()
#
#
#def parse_molecular_data(molec_file, property_file, descriptor,\
#                         atomic_cut_off, larger_mol): #TODO what is
#  """xxx."""
#  f=open(molec_file,'r');    lf=f.readlines(); f.close()
#  t=open(property_file,'r'); lt=t.readlines(); t.close()
#
#  Ndata=len(lt)
#  print 'Found', Ndata, 'molecules'
#  cut_off=atomic_cut_off2cut_off(larger_mol)
#
#  #Initializing X with one 1D array of zeros
#  X=np.zeros([Ndata+1,cut_off])
#  y=np.zeros( Ndata+1 )
#  N=np.zeros( Ndata+1,dtype=int)
#  print '\n(*) Creating Coulomb matrices with',X.shape[1],'elements.\n'
#
#  if descriptor=='MolecularCoulombMatrix': DESCR=Obj_CM.CM_Molecular()
#
#  line=0;
#  while(line<len(lt)):
#    y[line+1]=float(lt[line])
#    line+=1
#
#  line=0; nmol=0;
#  while(line<len(lf)):
#    nat=int(lf[line])
#    N[nmol+1]=nat
#    atomic_data_str=np.array([lf[line+i+2].split() for i in range(nat)])
#    atomic_data_str=atomic_data_str[atomic_data_str.argsort(axis=0)[:,0]]
#
#    cmatrix_of_selected_element=DESCR.f(atomic_data_str)
#    #DESCR.plot(atomic_data_str)
#
#    X[nmol+1,:cmatrix_of_selected_element.shape[0]]=cmatrix_of_selected_element
#    #Note that X[0] keeps being [0,0,0,0,...] always.
#    sys.stdout.write("\r[%d] " % (nmol+1))
#    sys.stdout.flush()
#    nmol+=1
#    line+=nat+2
#
#  return (X,y,N)


def parse_atomic_data(data_filenames, settings):
    '''
    Parses xyz files with atomic properties in the 5th through the last column and constructs the descriptor.
    Returns a tuple a,b,c where a is a 2d ndarray of size (N_elements,descriptor_length),
    b is a 1d array of the property and c is a dictionary that maps each molecule with the atoms in it.
    '''
    if settings.descriptor == 'CoulombMatrix': descr = Obj_CM.CM_Atomic()
    elif settings.descriptor == 'SortedCoulombMatrix': descr = Obj_CM.CM_AtomicSorted()
    elif settings.descriptor == 'RandomSortedCoulombMatrix': descr = Obj_CM.CM_SortedRandom()
    else: quit("Descriptor: %s not recognized" % descriptor)

    comb_descriptor = []
    comb_observables = []
    molecule_indices = dict([(x,[]) for x in range(len(data_filenames))])
    t = np.zeros(1,dtype=np.float128)
    c = 0 # counts the current number of elements
    t1 = time.time()
    for i, fname in enumerate(data_filenames):
        tokens = fname.split("/")[-1].split("_")[:2][::-1]
        # keep track of which indices a new molecule starts at,
        # so we make sure that we don't have contamination
        # between training and test dataset
        with open(fname) as f:
            lines = f.readlines()
            if sys.argv[5] == "CM":
                data = [line.split() for line in lines[2:]]
                descriptor, observables = descr.f(data, settings)
                comb_descriptor.extend([descriptor[int(sys.argv[1])]-1])
            else:
                count = 0
                coords = []
                for line in lines:
                    tokens = line.split()
                    if tokens[-1] == "CA":
                        coords.append(np.asarray(tokens[1:4],dtype=float))
                coords = np.asarray(coords)
                dist_vector = np.sum((coords-coords[int(sys.argv[1])-1])**2,axis=1)**0.5
                comb_descriptor.append(dist_vector)
                observables = [["CA"]*56]
            #comb_descriptor.extend(descriptor)
            if (~np.isfinite(comb_descriptor[-1])).any():
                quit(fname)
            comb_observables.extend(observables)
        molecule_indices[i] = range(c,len(comb_descriptor))
        c = len(comb_descriptor)
    descriptor = np.asarray(comb_descriptor)
    observables = np.asarray(comb_observables)

    return descriptor, observables, molecule_indices

