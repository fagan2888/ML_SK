#!/usr/bin/env python2

import glob
import time
import warnings
from itertools import product
import copy
import operator

import sklearn
import sklearn.metrics
import sklearn.feature_selection
import sklearn.linear_model
import sklearn.mixture
import sklearn.cluster
import numpy as np
import scipy.optimize

import utils
import parser.atomic

import pp_parallel

# initialize settings object
settings = utils.Settings()

# TODO split settings in descriptor and prediction part - at least for saving pickles
settings.data_folder      = 'data/delta'
settings.output_folder    = 'output/'
settings.target_element   = 'C'
settings.cut_off          = 4.       #Cut_off used to parse the dataset.
settings.verbose          = True
settings.observable_index = 0
settings.label_index      = 3
settings.max_neighbours = min(int(10*(settings.cut_off-0.9)+2.49),56)


# TODO clustering of input for label specific prediction #L
# TODO biclustering for selecting bayesian linear regression clusters
# and for selecting partitions of labels for optimal prediction #L
# TODO fit diagonal coulomb matrix #L
# TODO quippy #L
# TODO dampening for bobh #L
# TODO residue pair specific weights/dampening #L

#preprocess
# linear model

# dimesionality reduction #L
# VarianceThreshold
# mutual_info_classif
# SpectralEmbedding
# MDS
# TSNE
# TruncatedSVD
# PLSRegression
# PLSCanonical
# CCA
# PLSSVD
# GaussianRandomProjection
# SparseRandomProjection
# PCA
# IncrementalPCA
# ProjectedGradientNMF
# RandomizedPCA
# KernelPCA
# FactorAnalysis
# FastICA
# TruncatedSVD
# NMF
# SparsePCA
# MiniBatchSparsePCA
# fastica

# regressor
# GaussianProcessRegressor #M
# KernelRidge #H
# linear models #L
# neighbours models #L
# neural network #L
# SVR #L
# LinearSVR #L
# NuSVR #L
# DecisionTreeRegressor #L
# ExtraTreeRegressor #L
# AdaBoostRegressor #M
# BaggingRegressor #M
# ExtraTreesRegressor #M
# GradientBoostingRegressor #M
# RandomForestRegressor #M

# kernel
# kernel.Sum #L
# kernel.Product #L
# kernel.Exponentiation #L
# ConstantKernel #L
# RBF #H
# Matern #M
# RationalQuadratic #M
# ChiSquaredKernel (histogram) #L
# CosineSimilarity #L

# kernelapproximation
# AdditiveChi2Sampler (histogram) #L
# SkewedChi2Sampler (histogram) #L
# Nystroem (all) #L
# RBFSampler #L

############################################################################
########################### Loading Dataset ################################
############################################################################

# TODO move away from main
def make_data(settings):
    data_filenames = glob.glob(settings.data_folder + "/*.xyz")

    if settings.atomic:
        # (fitted) Largest number of neighbours within cutoff in the dataset.
        # Redundant dimensions are removed further down
        if settings.cut_off == None:
            quit("Cut-off needs to be set")

        parsed_data = parser.atomic.parse_data(data_filenames, settings)
    else:
        # Create molecular version #M
        quit("Molecular version not implemented")

    # the descriptors
    X = parsed_data[0]
    # Remove all features that are constant in the dataset.
    x = sklearn.feature_selection.VarianceThreshold(threshold=0.0).fit_transform(X)
    # all observables
    Y = parsed_data[1]
    # mapping from molecule to index in x/Y
    molecule_indices = parsed_data[2]

    return utils.Data(x, Y, molecule_indices, settings)


############################################################################
########################## Training and Predicting #########################
############################################################################

def predict(model, settings, x_train, x_test, y_train, y_test, return_full = False):

    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    if return_full:
        return y_pred

    mae  = sklearn.metrics.mean_absolute_error(y_pred, y_test)
    rmsd = sklearn.metrics.mean_squared_error(y_pred, y_test)

    return mae, rmsd

def do_prediction(settings, x0, *args):

    if settings.regressor == "KernelRidge":
        return do_KRR(settings, x0, args)

def do_KRR(settings, x0, args):
    if settings.kernel == "laplacian":
        # not the most beautiful syntax, but puts settings in args for the optimization function
        metric = KRR_laplace(x0, settings, *args)
    else:
        quit("Only Laplacian Kernel at the moment")

    return metric

    
def fit_params(settings, *args):

    if settings.regressor == "KernelRidge":
        return fit_KRRparams(settings, args)

def opt(opt_func, x0, settings, args, lb = [], ub = []):
    args = (settings,) + args
    # TODO revert change
    if settings.optalg == "neldermead":
        xopt = scipy.optimize.minimize(opt_func, x0, args=args,
                     method='Nelder-Mead',options={'ftol':1e-5})
        #xopt = scipy.optimize.minimize(opt_func, x0, args=args,
        #             method='COBYLA',tol=1e-6, options={'rhobeg':(1e-5,1e-5)})
    elif settings.optalg == "cobyla":
        xopt = scipy.optimize.minimize(opt_func, x0, args=args,
                     method='COBYLA',tol=1e-6, options={'rhobeg':(1e-5,1e-5)})

    return xopt.x
    #xopt = scipy.optimize.least_squares(opt_func, [1e-4,0.0001], bounds = [lb,ub], ftol = 1e-6, xtol=1e-10)
    #print "l_sq",t1 - t0, "\n", xopt
    #xopt = scipy.optimize.fmin_slsqp(opt_func, [1e-4,0.0001], bounds = zip(lb,ub), acc=5e-7, full_output=1)
    #print "slsqp",t1 - t0, "\n", xopt
    #xopt = scipy.optimize.fmin_l_bfgs_b(opt_func, [1e-4,0.0001], bounds = zip(lb,ub), m=50, maxls=100,approx_grad=1, epsilon=1e-8,factr=1e8,pgtol=1e-3)
    #print "l-bfgs-b",t1 - t0, "\n", xopt
    #xopt = scipy.optimize.fmin_powell(opt_func, [1e-4,0.0001], direc=[-1e-6,-1e-6], full_output=1, xtol=0.00001, ftol=0.00001)
    #print "powell",t1 - t0, "\n", xopt
    #xopt = scipy.optimize.minimize(opt_func, [1e-4,0.0001], method='SLSQP',bounds = zip(lb,ub),options={'ftol':5e-7})
    #print "slsqp",t1 - t0, "\n", xopt

def KRR_laplace(x0,*args):
    model = sklearn.kernel_ridge.KernelRidge(alpha = x0[0], gamma = x0[1], kernel = "laplacian")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        settings = args[0]
        if settings.metric == "mae":
            return predict(model, *args)[0]
        else:
            return predict(model, *args)[1]



def fit_KRRparams(settings, args):
    lb = [1e-10,-1e-5]
    ub = [1e-4,1e-5]

    x0 = [np.random.choice(np.logspace(-10,-4)), np.random.choice(np.logspace(-9,-5))*np.sign(np.random.random()-0.5)]

    if settings.kernel == "laplacian":
        params = opt(KRR_laplace, x0, settings, args, lb=lb, ub=ub)
        params[0] = abs(params[0])
    else:
        quit("Only Laplacian Kernel at the moment")

    return params

# TODO if only conformational changes, sort CM by actual interaction #M
# TODO shiftx2/camshift etc. #H
# TODO remove filtered structures #H
# TODO get error per label #H
# TODO HF delta #L

def do_initial_plots(data, settings, job_server):
    import numpy as np
    from plotting import plots
    preproc = kpreprocess(1,data).flatten()
    data.make_labels(preproc,5, scale=None)
    y1 = data.y[:,5].astype(np.float)
    #y1 = data.y[:,5].astype(np.float)
    y1 = data.transform(y1, settings)
    y2 = data.y[:,6].astype(np.float)
    labels = data.y[:,3]
    uniq_labels = np.unique(labels)

    modules =  ("numpy as np",
                "matplotlib.pyplot as plt",
                "scipy.stats as ss",
                "matplotlib.cm as cm")
    #job_server.submit(plots.plot_labels, args=(y1,y2,labels,"opbe_vs_lmp2_labels"), group="plots", modules = modules)
    #job_server.submit(plots.plot_scatter, args=(y1,y2,"opbe_vs_lmp2"), group="plots", modules = modules)
    plots.plot_kde(y1,"opbe_KDE_K1")
    #job_server.submit(plots.plot_kde, args = (y1-y2,"delta_KDE K=2"), group="plots", modules=modules)

def kpreprocess(n_clusters, data):
    x = data.y[:,5].astype(np.float).reshape(-1,1)
    y = data.y[:,6].astype(np.float).reshape(-1,1)
    labels = data.y[:,3]
    uniq_labels = np.sort(np.unique(labels))
    X = []
    # only use medians for each class
    for l in uniq_labels:
        X.append(np.median(x[labels == l]))
    X = np.asarray(X).reshape(-1,1)

    model = sklearn.cluster.KMeans(n_clusters=n_clusters)
    classification = model.fit_predict(X)

    # return labels for each datapoint
    return np.asarray([classification[ np.where(uniq_labels == labels[i]) ] for i in range(x.size)])

def process_all(d, settings, job_server):
    input = [dict(zip(d, v)) for v in product(*d.values())]
    jobs1 = {}
    jobs2 = {}
    data = {}
    results = []

    inp_settings = []
    for i, inp in enumerate(input):
        # TODO remove bad combinations

        this_settings = utils.create_settings(inp, settings)
        inp_settings.append(copy.deepcopy(this_settings))

        data[i] = job_server.submit(make_data, args=(this_settings,), 
                modules=("glob","parser.atomic","sklearn.feature_selection", "utils"))



    for i, this_settings in enumerate(inp_settings):

        preproc = kpreprocess(5,data[i]()).flatten()
        data[i]().make_labels(preproc,0, scale="all")
        data[i]().y[:,0] = data[i]().transform(data[i]().y[:,0].astype(float), this_settings)

        # submit everything
        jobs1[i] = process_cv(data[i](), this_settings, job_server)

    # next round of jobs
    # check if jobs needed for next step is done
    flag = True
    c = 0
    while flag == True:
        c += 1
        if c % 10 == 0:
            print job_server.print_stats()
        flag = False
        for i in jobs1.keys():
            flag = True
            if len(jobs1[i].keys()) == 0:
                print "deleting jobs1",i
                del jobs1[i]
                continue
            for n in jobs1[i].keys():
                # if all jobs for i, n is finished, continue next step
                #if reduce(operator.mul, [job.finished for job in jobs1[i][n].values()]):
                #    params = [job() for job in jobs1[i][n].values()]
                #    opt_params = np.median(params, axis=0)
                #    print "deleting jobs1",i,n
                del jobs1[i][n]
                #    print "opt_params", i, n, opt_params
                x_test, x_val, x_train = data[i]().get(data[i]().x,n)
                y_test, y_val, y_train = data[i]().get(data[i]().y[:,settings.observable_index],n)
                #l_test, l_val, l_train = data.get(data.y[:,settings.label_index],n,m)
                x_train = np.concatenate([x_train[:],x_val[:200]])
                y_train = np.concatenate([y_train[:],y_val[:200]])
                x_test = x_test[:100]
                y_test = y_test[:100]

                if i not in jobs2.keys(): jobs2[i] = {}

                jobs2[i][n] = job_server.submit(do_prediction, 
                        args=(inp_settings[i], [3e-8,3e-9], x_train, x_test, y_train, y_test),
                        depfuncs=(do_KRR,KRR_laplace,predict),
                    modules = ("numpy as np","scipy.optimize", "sklearn.kernel_ridge","warnings"))
                    #print jobs2[i][n]()

        for i in jobs2.keys():
            flag = True
            # if all jobs for i is finished, continue next step
            if reduce(operator.mul, [job.finished for job in jobs2[i].values()]):
                metric = [job() for job in jobs2[i].values()]
                del jobs2[i]
                median_metric = np.median(metric)*data[i]().sigma[0]
                results.append((inp_settings[i], median_metric))

        if flag == False:
            break

        time.sleep(5)

    return results



def process_job_cv_val_step(settings, data, n, m):
    #x_test, x_val, x_train = data.get(data.x,n,m)
    #y_test, y_val, y_train = data.get(data.y[:,settings.observable_index],n,m)
    ##l_test, l_val, l_train = data.get(data.y[:,settings.label_index],n,m)
    #x_train = x_train[:1000]
    #y_train = y_train[:1000]
    #x_val = x_val[:500]
    #y_val = y_val[:500]

    #return fit_params(settings, x_train, x_val, y_train, y_val)
    return {0:0, 1:1, 2:2, 3:3, 4:4}


def process_cv(data, settings, job_server):

    jobs = {}

    # run
    # limit data set size
    # preprocess
    # CV cont params (or not)


    # Each job should take the same amount of time. Roughly.
    # So it should be fine not to group it
    for n in range(settings.testK):
        jobs[n] = {}
        for m in range(settings.valK):
            jobs[n][m] = job_server.submit(process_job_cv_val_step, args = (settings, data, n, m),
                    depfuncs = (fit_params,fit_KRRparams, opt, KRR_laplace, predict), 
                    modules = ("numpy as np","scipy.optimize", "sklearn.kernel_ridge","warnings"))

    # lest hope that the memory can handle this
    return jobs


if __name__ == '__main__':


    #jobid = pp_parallel.start_workers(50)
    #time.sleep(5)
    job_server = pp_parallel.start_server(8)


    data = make_data(settings)
    #do_initial_plots(data, settings, job_server)
    #quit()
    d = {}
    ### pipeline
    #### descriptor
    d["target_element"] = ["C"]#,"H","N"]
        #self.train_size = #TODO
        #self.test_size = #TODO
    d['cut_off'] = [1.5,2.0]#,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0]
    #d['cut_off2'] = [1000., 8.0, 7.0, 6.0, 5.0]
    d['descriptor'] = ['CoulombMatrix']#, 'SortedCoulombMatrix', 'RandomSortedCoulombMatrix']
    d['regressor'] = ['KernelRidge'] # optparam
    d['kernel'] = ['laplacian']#, 'generalized_normal', 'matern1', 'matern3', 'matern5'] #optparam
    #d['exponent'] = range(1,13)
    #d['sncf'] = [False, True]
    #d['double_diagonal'] = [False, True]
    #d['reduce'] = ["bob", "bobd", "eig", "red", False]
    #d['mulliken'] = [None, 4]
    #d['dampening'] = [None, "smooth", "cos", "norm", "laplace", "smooth_norm", "smooth_laplace"]
    #d['dampening2'] = [None, "smooth", "cos", "norm", "laplace", "smooth_norm", "smooth_laplace"]
    #d['damp_const1'] = [1, 1e-1, 1e-2, 1e-3, 1e-4]
    #d['damp2_const1'] = [1, 1e-1, 1e-2, 1e-3, 1e-4]
    #d['damp_const1'] = [2., 1.5, 1., 0.5]
    #d['damp2_const1'] = [2., 1.5, 1., 0.5]
    #d['cm_cut'] = [0, 1e-2, 1e-1, 1, 1e2]
    #d['force_central_first'] = True, False
    #d['random'] = [0, 2, 4, 8]
    #d['random_sigma'] = [1e-3, 1e-2, 1e-1, 1, 10]
    #d['self_energy_param'] = someiter
    d['metric'] = ["mae"]

    #d['centering'] = ["all", 1, 4, 5, None]
    #d['scale'] = ["all", 1, 4, 5, None]

    results = process_all(d, settings, job_server)
    for i, (s, j) in enumerate(results):
        print i, j, s.__dict__['cut_off']

    #### crossval
    ##### regressor

    ### Different preprocessing
    #kpreprocessing(data,job_server)

    print job_server.print_stats()
    # kill slurm jobid
    pp_parallel.kill_workers(jobid)
    # kill workers that the server can find
    job_server.destroy()

#print do_param_search(N = 5, algo = algo, verbose = True, scaling_method = None, kernel = kernel)
#print do_prediction(cut_off = float(sys.argv[1]))
    #model = sklearn.linear_model.RidgeCV(alphas=[10**i for i in range(-8,5)])
    #model.fit(y1[labels==uniq_labels[0]].reshape(1,-1),y2[labels==uniq_labels[0]].reshape(1,-1))

#labels = np.unique(xylabels)
#assignments = np.random.randint(0,2,labels.size)
#print ML_SK.clustering.loglikelihood(x, y, xylabels, assignments, labels, 2)
#
