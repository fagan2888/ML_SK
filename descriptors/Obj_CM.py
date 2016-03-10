import numpy as np
from ML.descriptors import CoulombMatrix as cmf



class CM_Radial(object):
  """xxx"""
  def f(self,atomic_data_str,elem,radial_cut_off):
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
    ei=cmf.elem_index(atomic_data_str,elem)

    cm  = cmf.cmatrix(atomic_data_str,elem,radial_cut_off)

    return np.array([cm[i][np.triu_indices(cm[i].shape[0])] \
                     for i in range(len(ei))])



class CM_Atomic(object):
  """xxx"""
  def f(self,atomic_data_str,elem,atomic_cut_off):
    """
    Returns the Coulomb matrices (CM) of all atoms of species 'elem'
    in the molecule.
    The output is an (N_elem,cut_off) numpy array with
    each row corresponding to the reshaped upper triangle of
    side 'atomic_cut_off' of the CM with the 'elem' as head element.
    Here the molecule is ordered by distance to the head atom.
    """
    nat = len(atomic_data_str)
    ei=cmf.elem_index(atomic_data_str,elem)

    cm  = cmf.cmatrix(atomic_data_str,elem,1000.)

    return np.array([cm[i][np.triu_indices(atomic_cut_off)] \
                     for i in range(len(ei))])

