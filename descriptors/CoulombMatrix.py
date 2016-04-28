import numpy as np
from scipy.spatial.distance import cdist
from itertools import permutations, product
from utils import Settings

# TODO finish bobh #M
# TODO quippy descriptors #L

# define atom weights
atom_weights = {'H':1,'C':6,'N':7,'O':8,'F':9,'S':16}

def distance(nodes):
    """
    calculate the euclidian pair distances squared between two equal sized sets of coordinates
    """
    return np.sqrt(np.einsum('ij,ij->i',nodes,nodes))

def elem_index(data, elem):
    """
    Returns a list with the positions (row in the .xyz file) of the
    element 'elem' in the molecular data block:
    For example:
    >>>elem_index(mol_str,'O')
    [0, 2]
    means that oxygen is in the first (0) and third (2) rows.
    """
    return np.where(data[:,0] == elem)[0]


def cmatrix_sorted_rows(data, settings):
    """
    Returns the local sorted Coulomb matrix (CM).

    The output is an (N_elem x (N_neighbours*(N_neighbours+1)/2)) shaped matrix.

    Here the molecule is ordered by the row 2-norm
    """
    # get CMs sorted by distance to center atom
    CMs, observables = cmatrix_sorted_distance(data, settings)

    # Not efficient, but not ratelimiting step - convert the 1d arays back to
    # triangular matrices
    full_CMs = np.zeros((CMs.shape[0],settings.max_neighbours,settings.max_neighbours))
    ind = np.triu_indices(settings.max_neighbours)
    full_CMs[:,ind[0],ind[1]] = CMs
    full_CMs[:,ind[1],ind[0]] = CMs

    # resort by 2-norm (and 1-norm on ties)
    l2init = np.linalg.norm(full_CMs,axis=-1)
    l1 = np.linalg.norm(full_CMs,axis=-1,ord=1)

    if settings.random > 0:
        # add random elements to 2-norm
        l2init = np.random.normal(l2init,settings.random_sigma,((settings.random,)+l2init.shape))
        CMs = np.tile(CMs,(random,1))

    else:
        l2init = np.expand_dims(l2init,0)

    for l, l2 in enumerate(l2init):
        sort_indices = np.lexsort((l2,l1))[:,::-1]
        # make sure the central atom is first in the sorting
        if settings.force_central_first:
            sort_indices = np.insert(sort_indices[sort_indices > 0].reshape((-1,settings.max_neighbours-1)),0,0,axis=1)
        nonsort_indices = range(settings.max_neighbours)
        for i,CM in enumerate(full_CMs):
            full_CMs[i,:,[sort_indices[i],nonsort_indices]] = full_CMs[i,:,[nonsort_indices,sort_indices[i]]]
            full_CMs[i,[sort_indices[i],nonsort_indices],:] = full_CMs[i,[nonsort_indices,sort_indices[i]],:]
            CMs[i] = full_CMs[i][np.triu_indices(settings.max_neighbours)]

    return CMs, observables

### begin functions for dampening ###
def cos_damp(r,r_cut,*args):
    return np.piecewise(r,[r<r_cut, r>=r_cut], [0.5 + 0.5*np.cos(np.pi*r/r_cut),0])

def damp(r,r_cut,alpha, dr):
    return np.piecewise(r,[r<r_cut-dr, (r>=r_cut-dr) & (r<r_cut), r>=r_cut], [1,lambda r:2*((r-r_cut)/dr)**3 + 3*((r-r_cut)/dr)**2,0])

def normal_damp(r,r_cut,alpha,*args):
    return np.exp(-alpha*r**2)

def laplace_damp(r,r_cut,alpha,*args):
    return np.exp(-alpha*r)

def no_damp(*args):
    return 1

def smooth_norm_damp(r,r_cut,alpha,dr):
    return normal_damp(r,r_cut,alpha)*damp(r,r_cut,dr)

def smooth_laplace_damp(r,r_cut,alpha,dr):
    return laplace_damp(r,r_cut,alpha)*damp(r,r_cut,dr)

def set_fc(dampening):
    fc = None
    if dampening == 'smooth':
        fc = damp
    elif dampening == None:
        fc = no_damp
    elif dampening == 'cos':
        fc = cos_damp
    elif dampening == 'norm':
        fc = normal_damp
    elif dampening == 'laplace':
        fc = laplace_damp
    elif dampening == 'smooth_norm':
        fc = smooth_norm_damp
    elif dampening == 'smooth_laplace':
        fc = smooth_laplace_damp
    return fc
### end functions for dampening ###

# TODO molecular observables
# TODO add more cutoffs

## bag of bonds histogram proposed in EDIC-ru/05.05.2009
## TODO finish # M
## TODO add triplets # L
#def bobh(data, elem, cut_off, atoms = ["C","H","N","S","O"],dtype = float, binsize = 0.25, sncf = True, triplet = True):
#    """
#    Returns the bag of bonds histogram
#    """
#    # coordinates
#    coords = data[:,1:4].astype(dtype)
#    # indices of atoms of type elem
#    elem_indices = elem_index(data, elem)
#
#    # get observables for selected element
#    observables = data[elem_indices,4:]
#    CMs = np.zeros((len(elem_indices), len(atoms) + ((len(atoms)*(len(atoms)+1))/2)*round(cut_off/binsize+0.99)))
#    for i, ei in enumerate(elem_indices):
#
#        # distances between atoms
#        # coordinates relative to the head atom
#        relative_coords = coords - coords[ei]
#        dist = distance(relative_coords)
#
#        # number of neighbours
#        N_neighbours = sum(dist <= cut_off)
#
#        # get indices for sorting by distance
#        sort_indices = dist.argsort()[:N_neighbours]
#
#        # only include interactions between atoms within cutoff of center atom
#        relative_coords = relative_coords[sort_indices]
#
#        if sncf:
#            # create sncf distance vector
#            R = np.zeros((N_neighbours,N_neighbours))
#            for i, Ri in enumerate(coords[sort_indices]):
#                for j, Rj in enumerate(coords[sort_indices][i:]):
#                    R[i,j+i] = (np.mean(relative_coords[i]**2)**0.5 \
#                             + np.mean(relative_coords[i+j]**2)**0.5 \
#                             + np.mean((Ri-Rj)**2)**0.5) ** exponent
#                    R[i,j+i] /= (fc(relative_coords[i],cut_off,damp_const1,damp_const2)*
#                                 fc(relative_coords[i+j],cut_off,damp_const1,damp_const2)*
#                                 fc(Ri-Rj,cut_off,damp_const1,damp_const2))
#                    R[i+j,i] = R[i,i+j]
#        else:
#            # create euclidian distance vector
#            R = cdist(relative_coords, relative_coords)**exponent
#            # dampening
#            R /= fc(relative_coords,cut_off,damp_const1,damp_const2)
#
#        bob = []
#        atom_types = data[:,0][sort_indices]
#        # create all bond types
#        bonds = []
#        for i, ai in enumerate(atoms):
#            for j, aj in enumerate(atoms[i:]):
#                bonds.append((i,j))
#
#        # TODO add number of atoms
#        # TODO add kernel
#        # TODO finish
#        for j,k in bonds:
#            indicesj = np.where(atom_types == j)[0]
#            indicesk = np.where(atom_types == k)[0]
#            index_pairs = product(indicesj, indicesk)
#            # only use pairs that are unique
#            if reduce == 'bob':
#                list(index_pairs) # the x[0] != x[1] check fails if this isn't put before
#                uniq_pairs = np.array(list(set(tuple(sorted(x)) for x in index_pairs if x[0] != x[1])))
#            elif reduce == 'bobd':
#                uniq_pairs = np.array(list(set(tuple(sorted(x)) for x in index_pairs)))
#            if uniq_pairs.shape[0] > 1:
#                bob.append(np.sort(np.pad(CM[uniq_pairs[:,0],uniq_pairs[:,1]],
#                                         (0,max_neighbours_dict[(j,k)]-len(uniq_pairs)),
#                                         mode='constant'))[::-1])
#            elif uniq_pairs.shape[0] == 1:
#                bob.append(np.zeros(max_neighbours_dict[(j,k)]))
#                bob[-1][0] = CM[uniq_pairs[0][0],uniq_pairs[0][1]]
#            else:
#                bob.append(np.zeros(max_neighbours_dict[(j,k)]))
#
#        CMs[i] = np.concatenate(bob)
#
#    return CMs, observables


def cmatrix_sorted_distance(data, settings):
    """
    Returns variants of the local sorted Coulomb matrix (CM).
    """
    dtype = np.float
    #define dampening function
    fc = set_fc(settings.dampening)
    fc2 = set_fc(settings.dampening2)
    # coordinates
    data = np.asarray(data)
    coords = data[:,1:4].astype(dtype)
    if settings.mulliken:
        # use mulliken charges
        z = np.matrix([atom_weights[x] for x in data[:,0]])
        charges = np.asmatrix(data[:,5],dtype=dtype)
        z2 = z.T * z - 0.5*(z.T*charges + charges.T*z) + charges.T*charges
        
    else:
        # weights / charges in the coulomb matrix
        z = np.asarray([atom_weights[x] for x in data[:,0]])
        z2 = np.asmatrix(z).T * np.asmatrix(z)
    # indices of atoms of type elem
    elem_indices = elem_index(data, settings.target_element)

    # get observables for selected element
    observables = data[elem_indices,4:]

    # optional use of the reduced coulomb matrix
    # See later for further details
    if settings.reduce == 'red':
        CMs = np.zeros((len(elem_indices),2*settings.max_neighbours-1))
    elif settings.reduce == 'eig':
        CMs = np.zeros((len(elem_indices),settings.max_neighbours))
    elif settings.reduce in ['bob','bobd']:
        CMs = np.zeros((len(elem_indices), sum((i for i in settings.max_neighbours_dict.values()))))
    else:
        CMs = np.zeros((len(elem_indices),(settings.max_neighbours*(settings.max_neighbours+1))/2))
    for e, ei in enumerate(elem_indices):

        # coordinates relative to the head atom
        relative_coords = coords - coords[ei]
        # distances between atoms
        dist = distance(relative_coords)

        # number of neighbours
        N_neighbours = sum(dist <= settings.cut_off)

        # get indices for sorting by distance
        sort_indices = dist.argsort()[:N_neighbours]

        # only include interactions between atoms within cutoff of center atom
        relative_coords = relative_coords[sort_indices]

        if settings.sncf:
            # create sncf distance vector
            R = np.zeros((N_neighbours,N_neighbours))
            for i, Ri in enumerate(coords[sort_indices]):
                for j, Rj in enumerate(coords[sort_indices][i:]):
                    R[i,j+i] = (np.mean(relative_coords[i]**2)**0.5 \
                             + np.mean(relative_coords[i+j]**2)**0.5 \
                             + np.mean((Ri-Rj)**2)**0.5) ** settings.exponent
                    with np.errstate(divide='ignore'): # ignore divide by zero warnings
                        R[i,j+i] /= (fc(relative_coords[i],settings.cut_off,settings.damp_const1,settings.damp_const2)*
                                     fc(relative_coords[i+j],settings.cut_off,settings.damp_const1,settings.damp_const2)*
                                     fc2(Ri-Rj,settings.cut_off2,settings.damp2_const1,settings.damp2_const2))
                    R[i+j,i] = R[i,i+j]
        else:
            # create euclidian distance vector
            R = cdist(relative_coords, relative_coords)**settings.exponent
            # dampening
            for i, Ri in enumerate(coords[sort_indices]):
                for j, Rj in enumerate(coords[sort_indices][i:]):
                    with np.errstate(divide='ignore'): # ignore divide by zero warnings
                        R[i,j+i] /= (fc(relative_coords[i],settings.cut_off,settings.damp_const1,settings.damp_const2)*
                                     fc(relative_coords[i+j],settings.cut_off,settings.damp_const1,settings.damp_const2)*
                                     fc2(Ri-Rj,settings.cut_off2,settings.damp2_const1,settings.damp2_const2))
                    R[i+j,i] = R[i,i+j]
        # inverse distance
        with np.errstate(divide='ignore'): # ignore divide by zero warnings
            invR = 1./R

        # calculate coulomb matrix
        s = sort_indices[np.mgrid[0:N_neighbours,0:N_neighbours]]
        weights = z2[s[0],s[1]]
        CM = np.einsum('ij,ij->ij',invR,weights)

        diagonal_weights = settings.self_energy_fn(np.sqrt(np.asarray(weights.diagonal())))
        if settings.double_diagonal:
            # set diagonal elements to Z^(2.4)
            np.fill_diagonal(CM,diagonal_weights)
        else:
            # set diagonal elements to 0.5*Z^(2.4)
            # note that weights.diagonal corresponds to z^2
            np.fill_diagonal(CM,0.5*diagonal_weights)

        # cut_off low values in the matrix
        CM[CM < settings.cm_cut] = 0

        if reduce == 'red':
            # pad to dimensions max_neighbours and return
            # the reduced coulomb matrix which is only the first row and diagonal
            # TODO make sorted version #L
            CMs[e] = np.concatenate([np.pad(CM[0],(0,max_neighbours-N_neighbours),mode='constant'),
                                     np.pad(CM.diagonal()[1:],(0,max_neighbours-N_neighbours),mode='constant')])
        if reduce == 'eig':
            # calculate eigenvalues and pad to max_neighbours
            CMs[e] = np.pad(np.sort(np.linalg.eigvalsh(CM))[::-1],(0,max_neighbours-N_neighbours),mode='constant')
        elif reduce in ['bob','bobd']:
            bob = []
            atom_types = data[:,0][sort_indices]
            # create all bond types
            for j,k in max_neighbours_dict.keys():
                indicesj = np.where(atom_types == j)[0]
                indicesk = np.where(atom_types == k)[0]
                index_pairs = product(indicesj, indicesk)
                # only use pairs that are unique
                if reduce == 'bob': # The original implementation without diagonal elements
                    list(index_pairs) # the x[0] != x[1] check fails if this isn't put before
                    uniq_pairs = np.array(list(set(tuple(sorted(x)) for x in index_pairs if x[0] != x[1])))
                elif reduce == 'bobd': # with diagonal elements
                    uniq_pairs = np.array(list(set(tuple(sorted(x)) for x in index_pairs)))
                if uniq_pairs.shape[0] > 1:
                    bob.append(np.sort(np.pad(CM[uniq_pairs[:,0],uniq_pairs[:,1]],
                                             (0,max_neighbours_dict[(j,k)]-len(uniq_pairs)),
                                             mode='constant'))[::-1])
                elif uniq_pairs.shape[0] == 1:
                    bob.append(np.zeros(max_neighbours_dict[(j,k)]))
                    bob[-1][0] = CM[uniq_pairs[0][0],uniq_pairs[0][1]]
                else:
                    bob.append(np.zeros(max_neighbours_dict[(j,k)]))

            CMs[e] = np.concatenate(bob)
        else:
            # pad to dimensions (max_neighbours,max_neighbours) and return
            # the upper triangular matrix
            CMs[e] = np.pad(CM,(0,settings.max_neighbours-N_neighbours),mode='constant')[np.triu_indices(settings.max_neighbours)]

    return CMs, observables


#def cmatrix_eigen(atomic_data_str,elem,cut_off):
#    """
#    Returns the Coulomb matrices (CM) of all atoms of element 'elem'
#    in the molecule.
#
#    The output is an 1D numpy array with the reshaped lower triangle
#    of the CM with the 'elem' as head element.
#
#    Here the molecule is ordered by distance to the head atom.
#    """
#    ei=elem_index(atomic_data_str,elem)
#
#    cm  = cmatrix(atomic_data_str,elem,radial_cut_off)
#
#    return np.array([np.linalg.eigh(cm[i])[0] for i in range(len(ei))])

if __name__ == "__main__":
    # for internal testing
    data = np.array([['C', '7.56873937876', '3.68060597047', '6.33261784125','123','5.8'],
                     ['C', '2.87980609339', '6.93282733005', '2.13562892849','153','6.2'],
                     ['H', '7.93736430134', '1.20064913896', '6.02568213786','3','1.1'],
                     ['C', '5.69443624018', '6.72826694866', '3.59996220496','4','5.7'],
                     ['N', '5.26265824714', '1.37885896213', '4.6860703957','120','7.5'],
                     ['C', '3.29543742382', '0.371825475271', '3.11646008871','143','6.0']],
                    dtype='|S32')
    settings = Settings()
    settings.target_element = "C"
    settings.cut_off = 5.0
    settings.max_neighbours = 6
    #settings.max_neighbours_dict = dict([(('C','C'),6),(('C','H'),3),(('C','N'),3),(('N','N'),3),(('N','H'),3),(('H','H'),3)]),
    settings.mulliken = True
    cmatrix_sorted_distance(data, settings)
