import numpy as np




z={'H':1.,'C':6.,'N':7.,'O':8.,'F':9.}


def vector_abs(nodes):
  """
  Absolute value of the vector 'nodes'.
  """
  return np.power(np.einsum('ij,ij->i',nodes,nodes),0.5)

def elem_index(atomic_data_str,elem):
  """
  Returns a list with the positions (row in the .xyz file) of the
  element 'elem' in the molecular data block:
  For example:
  >>>elem_index(mol_str,'O')
  [0, 2]
  means that oxygen is in the first (0) and third (2) rows.
  """
  return [i for i,j in enumerate(atomic_data_str[:,:1]) if j[0]==elem]


def cmatrix(atomic_data_str,elem,cut_off):
  """
  Returns the 2D Coulomb Matrix.
  """ 
  nat=len(atomic_data_str)
  atomic_coord=atomic_data_str[:,1:4].astype(np.float)
  ei=elem_index(atomic_data_str,elem)
  cm=np.zeros([len(ei),nat,len(atomic_data_str)])
 
  for cc,i in enumerate(ei):
      #sorting atoms with respect to the head atom
      dl=vector_abs(atomic_coord-atomic_coord[i])
      cmlen=len(dl[np.where(dl<cut_off)])
      print cmlen
      sorted_order=dl.argsort()
      zz=np.array([z[atomic_data_str[:,:1][at][0]] for at in sorted_order])
 
      for cci,ii in enumerate(sorted_order):
          dl=vector_abs(atomic_coord[sorted_order]-atomic_coord[ii])
          dl[cci]=1.
          invdl=z[atomic_data_str[:,:1][ii][0]]/dl
          #invdl[np.where(dl>cut_off)]=0.
          cm[cc][cci]=invdl*zz
          cm[cc][cci][cci]=\
                   np.array([0.5*np.power(z[atomic_data_str[:,:1][ii][0]],2.4)])
  return cm[:,:cmlen,:cmlen]


def cmatrix_atomic(atomic_data_str,elem,atomic_cut_off):
    """
    Returns the Coulomb matrices (CM) of all atoms of species 'elem'
    in the molecule.

    The output is an (N_elem,cut_off) numpy array with 
    each row corresponding to the reshaped upper triangle of 
    side 'atomic_cut_off' of the CM with the 'elem' as head element.

    Here the molecule is ordered by distance to the head atom.
    """

    nat = len(atomic_data_str)
    ei=elem_index(atomic_data_str,elem)
    
    cm  = cmatrix(atomic_data_str,elem,1000.)

    return np.array([cm[i][np.triu_indices(atomic_cut_off)] \
                     for i in range(len(ei))])




def cmatrix_radial(atomic_data_str,elem,radial_cut_off):
    """
    Returns the Coulomb matrices (CM) of all atoms of element 'elem'
    in the molecule.

    The output is an (N_elem,Nat*Nat) numpy array with the reshaped 
    upper triangle of the CM with the 'elem' as head element.
    The matrix elements with distances larger than 'radial_cut_off'
    are replaced by zero.

    Here the molecule is ordered by distance to the head atom.
    """

    nat=len(atomic_data_str)
    ei=elem_index(atomic_data_str,elem)
    
    cm  = cmatrix(atomic_data_str,elem,radial_cut_off)
    

    return np.array([cm[i][np.triu_indices(cm[i].shape[0])] \
                     for i in range(len(ei))])


    


def cmatrix_eigen(atomic_data_str,elem,cut_off):
    """
    Returns the Coulomb matrices (CM) of all atoms of element 'elem'
    in the molecule.

    The output is an 1D numpy array with the reshaped lower triangle
    of the CM with the 'elem' as head element.

    Here the molecule is ordered by distance to the head atom.
    """
    nat=len(atomic_data_str)
    ei=elem_index(atomic_data_str,elem)

    cm  = cmatrix(atomic_data_str,elem,radial_cut_off)

    return np.array([np.linalg.eigh(cm[i])[0] for i in range(len(ei))])
