import numpy as np

from ML_SK.descriptors import CoulombMatrix as cmf


class CM_Atomic(object):

    def f(self, data, settings,t):
        """
        Returns the Coulomb matrices (CM) of all atoms of element 'elem'
        in the molecule.
        The output is a 1d array descriptor with size depending on the options given
        """
        descriptor, properties = cmf.cmatrix_sorted_distance(data, settings,t)

        return descriptor, properties

class CM_AtomicSorted(object):

    def f(self, data, settings):
        """
        Returns the Coulomb matrices (CM) of all atoms of element 'elem'
        in the molecule.
        The output is a 1d array descriptor with size depending on the options given
        """
        descriptor, properties = cmf.cmatrix_sorted_rows(data, settings)
        return descriptor, properties

class CM_AtomicSortedRandom(object):

    def f(self, data, settings):
        """
        Returns the Coulomb matrices (CM) of all atoms of element 'elem'
        in the molecule.
        The output is a 1d array descriptor with size depending on the options given
        """
        descriptor, properties = cmf.cmatrix_sorted_rows(settings)
        return descriptor, properties


#class BoBH(object):
#
#    def f(self, data, elem, cut_off, sparse = False, dtype = np.float): # TODO smoothed cutoff
#        # TODO update this
#        """
#        Returns the Coulomb matrices (CM) of all atoms of element 'elem'
#        in the molecule.
#        The output is an (N_elem * (N_atom + 1) / 2 ) numpy 1d array with the reshaped
#        upper triangle of the CM with the 'elem' as head element.
#        The matrix elements with distances larger than 'cut_off'
#        are replaced by zero.
#        Here the molecule is ordered by distance to the head atom.
#        """
#        cm  = cmf.cmatrix(data, elem, cut_off, sparse = sparse, dtype = dtype) # TODO sparse
#        # TODO here
#
#        return np.array([cm[i][np.triu_indices(cm[i].shape[0])] for i in range(cm.shape[0])])

#TODO molecular
