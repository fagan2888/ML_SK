import numpy as np
import copy

class Settings:
    # just initialize most to None for now
    def __init__(self):
        self.data_folder = None # folder containing xyz files
        self.output_folder = None # folder to save data pickles
        self.target_element = None # atomtype to model for the atomic case
        #self.train_size = None
        #self.test_size = None
        self.cut_off = 4.0 # cut-off (or in the case of a smooth cut-off, the upper limit)
        self.cut_off2 = 1000. # second cut_off that works on atoms on opposite sides of the central atom
        self.max_neighbours = 56 # The matrix size is preallocated at this size, but excess dimensions are removed after parsing.
        self.max_neighbours_dict = {} # different max_neighbours for each atom type for bob etc.
        self.atomic = True # If atomic property of molecular
        self.descriptor = 'CoulombMatrix' # The base descriptor (CoulombMatrix, SortedCoulombMatrix, RandomSortedCoulombMatrix)
        self.regressor = 'KernelRidge' # Algorithm to the the predicitons (KernelRidge, ...)
        self.kernel = 'laplacian' # Kernel in KernelRidge (laplacian, ...)
        self.exponent = 1 # exponent of 1/r in the coulomb matrix. Should probably be betweet 1 and 12
        self.sncf = False # Use of the british railroad metric. For central atom j: 1/Rijk -> 1/(Rij + Rjk + Rik)
        self.double_diagonal = False # extra factor 2 in the diagonal (testing)
        self.reduce = False # Dimensionality reduction during parsing (bob, bobd, eig, red, False)
        self.mulliken = None # Use mulliken charges instead of static charges if available. (None, mulliken_charge_index)
        self.dampening = None # functions for dampening (...)
        self.damp_const1 = 1 # parameter in dampening. Typically the width of the dempening potential.
        self.damp_const2 = 1 # parameter in dampening. Typically the distance over where the potential is dampened. lower_cut_off = cut_off - dr
        self.dampening2 = None # functions for dampening (...)
        self.damp2_const1 = 1 # parameter in dampening. Typically the width of the dempening potential.
        self.damp2_const2 = 1 # parameter in dampening. Typically the distance over where the potential is dampened. lower_cut_off = cut_off - dr
        self.cm_cut = 0 # set values in the matrix lower than this to zero
        self.force_central_first = True # Force the central atom to appear first even in the sorted matrices
        self.random = 0 # number of random matrices to create
        self.random_sigma = 1 # stdev of the random variation in the l2norm of the rnadom matrices
        self.self_energy_param = (0.5, 2.4, 0, 1)# functional form of the diagonal elements of the coulomb matrix

        self.verbose = False # print out extra information. # TODO change to integer to set verbosity level
        self.observable_index = 0 # index for the observable to be predicted.
        self.label_index = None # index for atom/molecule labels if they exist
        self.testK = 5 # Kfold cross validation for test set
        self.valK = 5 # Kfold cross validation for validation set
        self.metric = "mae" # use mae or rmsd for optimizing parameters
        self.optalg = "neldermead"


    def self_energy_fn(self, x):
        A, a, B, b = self.self_energy_param
        return A*x**a + B*x**b

class Data:
    def __init__(self, x, y, mol_idx, settings):
        self.x = x.copy()
        self.y = y.copy()
        self.d_idx = self.split(mol_idx, settings)
        self.XYlabels = None
        self.labels = None

        # preprocessing transformation parameters
        self.sigma = None
        self.mu = None

    # selts mu, sigma and labels
    def make_labels(self, labels, idx, center = "labels", scale = "all"):
        self.XYlabels = labels
        self.labels = np.sort(np.unique(labels))
        this_y = self.y[:,idx].astype(np.float)

        # median
        if center == "labels":
            print this_y.shape
            self.mu = [np.median(this_y[labels == i]) for i in self.labels]
        elif center == "all":
            self.mu = [np.median(this_y) for i in self.labels]
        else:
            self.mu = [0 for i in self.labels]
        # IQR
        if scale == "labels":
            self.sigma = [np.subtract(*np.percentile(this_y[labels == i], [75, 25])) for i in self.labels]
        elif scale == "all":
            self.sigma = [np.subtract(*np.percentile(this_y, [75, 25])) for i in self.labels]
        else:
            self.sigma = [1 for i in self.labels]



    # Split in train/test/validation sets. Stratify if labels exist
    # It was painful to make sure the test and training set did not
    # contain atoms from the same molecule :S
    # Could probably return as something smarter than a dict
    def split(self, mol_idx, settings):
        d = {}
        idx = np.asarray(mol_idx.keys())
        np.random.shuffle(idx)

        N = idx.size
        if settings.label_index != None: # stratify. This only works with my specific data format right now
            # ok, so what happens is that we're gonna get the amino acid from the label column.
            # this is done by grapping the first atom of each molecule
            amino_acids = [self.y[mol_idx[i], settings.label_index][0] for i in idx]
            # then we find the indices that sort this list
            aa_sort = np.argsort(amino_acids)
            # idx is then sorted by amino acid type
            idx = idx[aa_sort]
            # another resort is done such that each amino acid type is split evenly in idx
            idx = [i for j in range(settings.testK) for i in idx[j::settings.testK]]

        for n in range(settings.testK):
            d[n] = {}
            test_idx = idx[(n*N)/settings.testK:((n+1)*N)/settings.testK]
            not_test_idx = np.extract(~np.in1d(idx,test_idx), idx)
            np.random.shuffle(not_test_idx)
            M = not_test_idx.size
            for m in range(settings.valK):
                d[n][m] = {}
                val_idx = not_test_idx[(m*M)/settings.valK:((m+1)*M)/settings.valK]
                train_idx = not_test_idx[~np.in1d(not_test_idx,val_idx)].copy()

                # d[n][m] is tuple of val, test and train indices for a given n, m
                d[n][m] = [i for j in test_idx for i in mol_idx[j]],  \
                          [i for j in val_idx for i in mol_idx[j]], \
                          [i for j in train_idx for i in mol_idx[j]]

        return d

    def get(self, t, n = 0, m = 0):
        idx = self.d_idx[n][m]
        try:
            float(t[idx[0][0]])
            return [np.asarray((t[i]), dtype=np.float) for i in idx] # return val/test/train for t
        except (ValueError, TypeError):
            return [np.asarray(t[i]) for i in idx] # return val/test/train for t

    def transform(self, y, settings):
        if self.XYlabels != None:
            y = np.asarray([(y[i]-self.mu[j])/self.sigma[j] for i, j in enumerate(self.XYlabels)])
        return y

    def inverse_transform(self, y, settings):
        if self.XYlabels != None:
            y = np.asarray([y[i]*sigma[j]+mu[j] for i, j in enumerate(self.XYlabels)])
        return y

def create_settings(d, settings):
    this_settings = copy.deepcopy(settings)
    for k, v in zip(d.keys(), d.values()):
        this_settings.__dict__[k] = v
    return this_settings

