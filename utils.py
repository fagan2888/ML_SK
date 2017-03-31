import numpy as np
import warnings
import copy
import sklearn.feature_selection
import sklearn.cluster
import glob
import scipy.optimize
import sys
import cPickle as pickle
from ML_SK.parser import atomic
import pandas
import hashlib
import os

class Settings:
    # just initialize most to None for now
    def __init__(self):
        self.data_folder = None # folder containing xyz files
        self.data_pickle_folder = None # folder to save data pickles
        self.output_folder = None # folder to save results as pandas DataFrames
        self.save_data_pickle = False # Save parsed data as pickle
        self.save_prediction = False # Save prediction
        self.target_element = None # atomtype to model for the atomic case (eg. 'C', 'H', ...)
        self.train_size = np.inf # Size of training set
        self.test_size = np.inf # Size of test set
        self.cv_train_size = np.inf # Size of training set for fitting kernel parameters
        self.cv_val_size = np.inf # Size of validation set for fitting kernel parameters
        self.cut_off = 4.0 # cut-off (or in the case of a smooth cut-off, the upper limit)
        self.cut_off2 = 1000. # second cut_off that works on atoms on opposite sides of the central atom
        self.max_neighbours = 56 # The matrix size is preallocated at this size, but excess dimensions are removed after parsing.
        # TODO merge this with max_neighbours
        self.max_neighbours_dict = {} # different max_neighbours for each atom type for bob etc.
        self.atomic = True # If atomic property of molecular
        self.descriptor = 'CoulombMatrix' # The base descriptor (CoulombMatrix, SortedCoulombMatrix, RandomSortedCoulombMatrix)
        self.regressor = 'KernelRidge' # Algorithm to the the predicitons (KernelRidge, ...)
        self.kernel = 'laplacian' # Kernel in KernelRidge (laplacian, ...)
        self.exponent = 1 # exponent of 1/r in the coulomb matrix. Should probably be betweet 1 and 12
        self.sncf = False # Use of the british railroad metric. For central atom j: 1/Rijk -> 1/(Rij + Rjk + Rik)
        self.reduce = False # Dimensionality reduction during parsing (bob, bobd, eig, red, False)
        self.mulliken = None # Use mulliken charges instead of static charges if available. (None, mulliken_charge_index) # NOTE: the index can't be 0
        self.dampening = None # functions for dampening (...)
        self.damp_const1 = 1 # parameter in dampening. Typically the width of the dempening potential.
        self.damp_const2 = 1 # parameter in dampening. Typically the distance over where the potential is dampened. lower_cut_off = cut_off - dr
        self.dampening2 = "hard" # functions for dampening (...)
        self.damp2_const1 = 1 # parameter in dampening. Typically the width of the dempening potential.
        self.damp2_const2 = 1 # parameter in dampening. Typically the distance over where the potential is dampened. lower_cut_off = cut_off - dr
        self.cm_cut = 0 # set values in the matrix lower than this to zero
        self.force_central_first = True # Force the central atom to appear first even in the sorted matrices
        self.random = 0 # number of random matrices to create
        self.random_sigma = 1 # stdev of the random variation in the l2norm of the rnadom matrices
        self.self_energy_param = (0.5, 2.4, 0, 1)# functional form of the diagonal elements of the coulomb matrix
        self.kernel_parameters_folder = None
        self.save_kernel_parameters = 1

        self.verbose = 0 # print out extra information.
        self.observable_index = 0 # index for the observable to be predicted.
        self.label_index = None # index for atom/molecule labels if they exist
        self.testK = 5 # Kfold cross validation for test set
        self.valK = 5 # Kfold cross validation for validation set
        self.metric = "mae" # use mae or rmsd for optimizing parameters
        #self.optalg = "neldermead"
        self.preprocess_nclusters  = None # None, or number of clusters
        self.baseline_index = self.observable_index # index of data to be clustered
        self.remove_labels = None

        self.md5_data = self.get_md5_data()

    # used to avoid calculating kernel parameters every time.
    # n is the index corresponding to the partition of data used in the cross validation
    def get_md5_kernel_parameters(self, n):
        string = self.md5_data
        string += str(self.regressor)
        string += str(self.kernel)
        string += str(self.metric)
        string += str(self.preprocess_nclusters)
        string += str(self.baseline_index)
        # TODO remove this as it's only in here to avoid recalculating alot of parameters
        if self.remove_labels == None:
            pass
        else:
            string += str(self.remove_labels)
        string += str(n)

        return hashlib.md5(string).hexdigest()

    def get_md5_data(self):
        """
        Converts a string based on settings to md5
        This way we can avoid parsing the same data
        over and over again.
        """
        string = str(self.data_folder)
        string += str(self.target_element)
        string += str(self.cut_off)
        string += str(self.cut_off2)
        string += str(self.atomic)
        string += str(self.descriptor)
        string += str(self.exponent)
        string += str(self.sncf)
        string += str(self.reduce)
        string += str(self.mulliken)
        string += str(self.dampening)
        string += str(self.damp_const1)
        string += str(self.damp_const2)
        string += str(self.dampening2)
        string += str(self.damp2_const1)
        string += str(self.damp2_const2)
        string += str(self.cm_cut)
        string += str(self.force_central_first)
        string += str(self.random)
        string += str(self.random_sigma)
        string += str(self.self_energy_param)
        string += str(self.observable_index)
        string += str(self.label_index)

        return hashlib.md5(string).hexdigest()

    def self_energy_fn(self, x):
        A, a, B, b = self.self_energy_param
        return A*x**a + B*x**b


class Data:
    def  __init__(self, x, y, mol_idx, settings):
        # Once initialized do not allow x and y to be changed.
        self._x_init = x
        #import sklearn.manifold
        #import time
        #t = time.time()
        #f = sklearn.manifold.SpectralEmbedding(n_components=20, affinity="rbf")#, eigen_solver="amg")
        #f.fit(x[:4000])
        #print x.shape, f.embedding_.shape,time.time()-t
        self._y_init = y
        self.mol_idx = mol_idx
        if settings.remove_labels != None:
            self.remove_labels(settings)
        self.d_idx = self.split(settings)
        # labels can be merged and stored here
        # for preprocessing (e.g. for methyle hydrogens)
        self.preprocess_labels = None
        self.classification = [None for _ in range(8)]
        self._y_preprocessed = self.preprocess_y(settings)

    def x(self):
        return self._x_init.copy()

    # if n is int, preprocess data with n clusters
    def y(self, k = None):
        if k == None:
            return self._y_init.copy()
        # just do this every time this is called, since it's hardly
        # a limiting factor speedwise
        return self._y_preprocessed[k-1]

    def preprocess_y(self,settings):
        '''
        Returns preprocessed observables
        '''
        if self.preprocess_labels == None:
            labels = self.y()[:,settings.label_index]
        else:
            labels = self.preprocess_labels

        Y = []
        # Add a bias to the data by clustering the baseline observables
        # try up to 6 clusters
        for k in range(1,9):
            # cluster the centroids of each label
            #x = self.y()[:,[settings.baseline_index,4,7,9]].astype(np.float)
            x = self.y()[:,settings.baseline_index].astype(np.float)
            y = self.y()[:,settings.observable_index].astype(np.float)

            classification = cluster(k,x, labels)
            self.classification[k-1] = classification.copy()

            y = self.y()[:,settings.observable_index].astype(np.float)

            # preprocess the observables
            y[classification == k-1] = y[classification == k-1] \
                                       - np.median(y[classification == k-1]) \
                                       + np.median(x[classification == k-1])

            Y.append(y.copy())
        return Y

    # Split in train/test/validation sets. Stratify if labels exist
    # It was painful to make sure the test and training set did not
    # contain atoms from the same molecule :S
    # Could probably return as something smarter than a dict
    # TODO move this to main as it's dataset specific
    def split(self, settings):
        d = {}
        idx = np.asarray(self.mol_idx.keys()).copy()
        np.random.shuffle(idx)

        N = idx.size
        if settings.label_index != None: # stratify. This only works with my specific data format right now
            # ok, so what happens is that we're gonna get the amino acid from the label column.
            # this is done by grabbing the first atomname of each molecule
            amino_acids = [self.y()[self.mol_idx[i], settings.label_index][0][0] for i in idx]
            # then we find the indices that sort this list
            aa_sort = np.argsort(amino_acids)
            # idx is then sorted by amino acid type
            idx = idx[aa_sort]
            # another resort is done such that each amino acid type is split evenly in idx
            idx = [i for j in range(settings.testK) for i in idx[j::settings.testK]]

        for n in range(settings.testK):
            d[n] = {}
            test_idx = idx[(n*N)/settings.testK:((n+1)*N)/settings.testK]
            not_test_idx = np.extract(~np.in1d(idx,test_idx), idx).copy()
            np.random.shuffle(not_test_idx)
            M = not_test_idx.size
            for m in range(settings.valK):
                val_idx = not_test_idx[(m*M)/settings.valK:((m+1)*M)/settings.valK]
                train_idx = not_test_idx[~np.in1d(not_test_idx,val_idx)]

                # d[n][m] is tuple of test, val and train indices for a given n, m
                d[n][m] = [i for j in test_idx for i in self.mol_idx[j]],  \
                          [i for j in val_idx for i in self.mol_idx[j]], \
                          [i for j in train_idx for i in self.mol_idx[j]]

        return d

    def get(self, t, n = 0, m = 0):
        idx = self.d_idx[n][m]
        try:
            float(t[idx[0][0]])
            return [np.asarray(t[i], dtype=np.float).copy() for i in idx] # return val/test/train for t
        except (ValueError, TypeError):
            return [np.asarray(t[i]).copy() for i in idx] # return val/test/train for t

    def remove_labels(self, settings):
        labels = self._y_init[:,settings.label_index]
        idx = np.in1d(labels, settings.remove_labels, invert=True)
        new_idx = np.cumsum(idx) - 1

        for k, v in self.mol_idx.items():
            v = np.asarray(v)
            self.mol_idx[k] = new_idx[v[idx[v]]]
        self._x_init = self._x_init[idx]
        self._y_init = self._y_init[idx]

        if settings.verbose >= 2:
            print "Removed specific labeled data and reduced data set size to", new_idx[-1] + 1


def save_dataframe(settings, y_test, y_pred, labels, parameters):
    N = y_test.size
    mae = np.mean(abs(y_test - y_pred))
    rmsd = np.mean((y_test - y_pred)**2)

    d = {
    "data_folder"         : [settings.data_folder for _ in range(N)],
    "target_element"      : [settings.target_element for _ in range(N)],
    "train_size"          : [settings.train_size for _ in range(N)],
    "cut_off"             : [settings.cut_off for _ in range(N)],
    "cut_off2"            : [settings.cut_off2 for _ in range(N)],
    "atomic"              : [settings.atomic for _ in range(N)],
    "descriptor"          : [settings.descriptor for _ in range(N)],
    "regressor"           : [settings.regressor for _ in range(N)],
    "kernel"              : [settings.kernel for _ in range(N)],
    "exponent"            : [settings.exponent for _ in range(N)],
    "sncf"                : [settings.sncf for _ in range(N)],
    "reduce"              : [settings.reduce for _ in range(N)],
    "mulliken"            : [settings.mulliken for _ in range(N)],
    "dampening"           : [settings.dampening for _ in range(N)],
    "damp_const1"         : [settings.damp_const1 for _ in range(N)],
    "damp_const2"         : [settings.damp_const2 for _ in range(N)],
    "dampening2"          : [settings.dampening2 for _ in range(N)],
    "damp2_const1"        : [settings.damp2_const1 for _ in range(N)],
    "damp2_const2"        : [settings.damp2_const2 for _ in range(N)],
    "cm_cut"              : [settings.cm_cut for _ in range(N)],
    "force_central_first" : [settings.force_central_first for _ in range(N)],
    "random"              : [settings.random for _ in range(N)],
    "random_sigma"        : [settings.random_sigma for _ in range(N)],
    "self_energy_param"   : [settings.self_energy_param for _ in range(N)],
    "observable_index"    : [settings.observable_index for _ in range(N)],
    "label_index"         : [settings.label_index for _ in range(N)],
    "mae"                 : [mae for _ in range(N)],
    "rmsd"                : [rmsd for _ in range(N)],
    "metric"              : [settings.metric for _ in range(N)],
    "preprocess_nclusters": [settings.preprocess_nclusters for _ in range(N)],
    "remove_labels"       : [settings.remove_labels for _ in range(N)],
    "baseline_index"      : [settings.baseline_index for _ in range(N)]
    }

    d["kernel_parameters"] = [parameters for _ in range(N)]
    d["y_test"] = y_test
    d["y_pred"] = y_pred
    d["labels"] = labels

    while True:
        # It makes a few things easier if we just use a random number as the pickle name
        dataframe_path = settings.output_folder + "/" + str(np.random.randint(0,1e18)) + ".pickle"
        if not os.path.isfile(dataframe_path):
            break
    df = pandas.DataFrame.from_dict(d)
    df.to_pickle(dataframe_path)


def create_settings(d, settings):
    this_settings = copy.deepcopy(settings)
    for k, v in zip(d.keys(), d.values()):
        this_settings.__dict__[k] = v
        #exec('this_settings.' + k + ' = v')
    this_settings.md5_data = this_settings.get_md5_data()
    return this_settings

def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def make_data(settings):
    data_filenames = glob.glob(settings.data_folder + "/*.xyz")

    if settings.atomic:
        # Redundant dimensions are removed further down
        parsed_data = atomic.parse_data(data_filenames, settings)
    else:
        quit("Molecular version not implemented")

    # the descriptors
    X = parsed_data[0]
    # Remove all features that does not vary in the dataset.
    x = sklearn.feature_selection.VarianceThreshold(threshold=0.0).fit_transform(X)
    # all observables
    Y = parsed_data[1]
    # Mapping from molecule to index in X/Y
    molecule_indices = parsed_data[2]

    return Data(x, Y, molecule_indices, settings)

def fit_params(settings, *args):

    if settings.regressor == "KernelRidge":
        return fit_KRRparams(settings, args)

def opt(opt_func, x0, settings, args, lb = [], ub = []):
    # Done to make settings available for the opt function
    args = (settings,) + args
    #if settings.optalg == "neldermead":#5m
    xopt = scipy.optimize.minimize(opt_func, x0, args=args,
            method='Nelder-Mead')#,tol=1e-4)#options={'ftol':settings.tol,'xtol':settings.tol})
    #elif settings.optalg == "cobyla":#10m
    #    xopt = scipy.optimize.minimize(opt_func, x0, args=args,
    #                 method='COBYLA',tol=1e-4, options={'rhobeg':(1e-5,1e-5)})
    #elif settings.optalg == "cg":#7m
    #    xopt = scipy.optimize.minimize(opt_func, x0, args=args,
    #                 method='CG',tol=1e-4)
    #elif settings.optalg == "ncg":#5m
    #    xopt = scipy.optimize.minimize(opt_func, x0, args=args,
    #                 method='Newton-CG',tol=1e-4)
    #elif settings.optalg == "bfgs":# 6m
    #    xopt = scipy.optimize.minimize(opt_func, x0, args=args,
    #                 method='BFGS',tol=1e-4)
    #elif settings.optalg == "l-bfgs-b":#10m
    #xopt = scipy.optimize.minimize(opt_func, x0, bounds=[(0,None) for _ in x0],
    #        args=args, method='TNC',options={"ftol":1e-4})
    #elif settings.optalg == "tnc":#7m
    #    xopt = scipy.optimize.minimize(opt_func, x0, args=args,
    #                 method='TNC',tol=1e-4)
    #elif settings.optalg == "slsqp":#6m
    #    xopt = scipy.optimize.minimize(opt_func, x0, args=args,
    #                 method='SLSQP',tol=1e-4)
    #elif settings.optalg == "dogleg":#6m
    #    xopt = scipy.optimize.minimize(opt_func, x0, args=args,
    #                 method='dogleg',tol=1e-4)
    #elif settings.optalg == "trust-ncg":#6m
    #    xopt = scipy.optimize.minimize(opt_func, x0, args=args,
    #                 method='trust-ncg',tol=1e-4)

    # check that the optimizer exited gracefully
    if xopt.success == False:
        quit("optimization of parameters failed:\n", xopt.message)

    return xopt.x

def fit_KRRparams(settings, args):

    if settings.kernel == "laplacian":
        # Normal KRR with laplacian kernel
        if settings.preprocess_nclusters == None:
            lb = [1e-14,1e-14]
            ub = [1e-10,1e-10]
            x0 = [np.random.choice(np.logspace(np.log10(lb[0]),np.log10(ub[0]))),
                  np.random.choice(np.logspace(np.log10(lb[0]),np.log10(ub[0])))]
        # else add a bias from clustering
        else:
            lb = [1e-14,1e-14] + [-np.inf for _ in range(settings.preprocess_nclusters)]
            ub = [1e-10,1e-10]+ [np.inf for _ in range(settings.preprocess_nclusters)]
            x0 = [np.random.choice(np.logspace(np.log10(lb[0]),np.log10(ub[0]))),
                  np.random.choice(np.logspace(np.log10(lb[0]),np.log10(ub[0])))]
            x0 += [0 for _ in range(settings.preprocess_nclusters)]
        params = opt(KRR, x0, settings, args, lb=lb, ub=ub)
    else:
        quit("Only Laplacian Kernel at the moment")

    return params

def KRR(x0, *args):
    if (x0 < 0).any():
        # TODO should probably return inf instead but need to test stability
        return 100
    settings, x_train, x_test, y_train, y_test  = args
    model = sklearn.kernel_ridge.KernelRidge(alpha = x0[0], gamma = x0[1], kernel = settings.kernel)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # return an infinite mae/rmsd if the fit does not converge
        try:
            model.fit(x_train,y_train)
            y_pred = model.predict(x_test)

            if settings.metric == "mae":
                return sklearn.metrics.mean_absolute_error(y_pred, y_test)
            elif settings.metric == "rmsd":
                return sklearn.metrics.mean_squared_error(y_pred, y_test)
            elif settings.metric == "full":
                return y_test, y_pred
        except np.linalg.linalg.LinAlgError:
            if settings.metric in ["mae", "rmsd"]:
                return np.inf
            else:
                quit("LinAlgError in KernelRidge")

def KRR_cluster(x0, *args):
    if (x0 < 0).any():
        # TODO stability?
        return np.inf
    settings, x_train, x_test, y_train, y_test  = args
    model = sklearn.kernel_ridge.KernelRidge(alpha = x0[0], gamma = x0[1], kernel = settings.kernel)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # return an infinite mae/rmsd if the fit does not converge
        try:
            model.fit(x_train,y_train)
            y_pred = model.predict(x_test)

            if settings.metric == "mae":
                return sklearn.metrics.mean_absolute_error(y_pred, y_test)
            elif settings.metric == "rmsd":
                return sklearn.metrics.mean_squared_error(y_pred, y_test)
            elif settings.metric == "full":
                return y_test, y_pred
        except np.linalg.linalg.LinAlgError:
            if settings.metric in ["mae", "rmsd"]:
                return np.inf
            else:
                quit("LinAlgError in KernelRidge")

def cluster(k, x, labels):
    #x = x.reshape(-1,1)

    uniq_labels = np.sort(np.unique(labels))
    X = []
    # only use medians for each class
    for l in uniq_labels:
        X.append(np.median(x[labels == l].astype(float),axis=0))
    xdim = 1
    if len(x.shape) == 2:
        xdim = x.shape[1]
    X = np.asarray(X).reshape(-1,xdim)
   
    #model = sklearn.cluster.KMeans(n_clusters=k, n_jobs=6)
    model = sklearn.cluster.AgglomerativeClustering(n_clusters=k,linkage="average")
    classification = model.fit_predict(X)

    # make cluster 0 correpond to the lowest mean etc.
    sort_idx = np.argsort([np.mean(X[classification==i,0]) for i in np.unique(classification)])
    tmp_class = classification.copy()
    for i in np.unique(classification):
        tmp_class[classification == sort_idx[i]] = i
    classification = tmp_class

    # return labels for each datapoint
    return np.asarray([classification[ np.where(uniq_labels == labels[i]) ] for i in range(x.shape[0])]).ravel()
