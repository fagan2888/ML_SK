#!/usr/bin/env python2

import time
from itertools import product
import copy
import operator
import numpy as np
import sys
import cPickle as pickle
import os

import sklearn.cluster

from ML_SK import utils
from ML_SK.utils import Data,cluster
from ML_SK import parallel
from ML_SK import plotting


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

# TODO if only conformational changes, sort CM by actual interaction #M
# TODO shiftx2/camshift etc. #H
# TODO remove filtered structures #H
# TODO get error per label #H
# TODO HF delta #L
# TODO remove (most) ALA cap atoms in training

# Initialize settings object
def init_settings():
    settings = utils.Settings()
    settings.data_folder              = '/home/lab/dev/ML_SK/data/delta'
    settings.data_pickle_folder       = '/home/lab/dev/ML_SK/data/data_pickles/'
    settings.kernel_parameters_folder = '/home/lab/dev/ML_SK/data/kernel_parameters/'
    #settings.plots_folder             = '/home/lab/dev/ML_SK/output/plots'
    settings.save_data_pickle         = 1
    settings.save_prediction          = 1
    settings.save_kernel_parameters   = 1
    settings.output_folder            = '/home/lab/dev/ML_SK/output/'
    settings.max_neighbours           = 56
    settings.label_index              = 3
    settings.verbose                  = 2
    settings.testK                    = 15
    settings.valK                     = 5
    settings.observable_index         = 0
    settings.baseline_index           = 5

    return settings


#TODO do this with the shieldings and add sqrt(2) weight in the KRR
class merge_labels:
    def __init__(self,settings):
        self.label_index = settings.label_index

    # merge labels for methyle and amine/amide hydrogens
    def do_merge(self, data):
        labels = data.y()[:,self.label_index]
        for l, i in enumerate(labels):
            if l == "A5":
                labels[i] = "A4"
            elif l == "A6":
                labels[i] = "A4"
            elif l == "A14":
                labels[i] = "A13"
            elif l == "A15":
                labels[i] = "A13"
            elif l == "A24":
                labels[i] = "A23"
            elif l == "A25":
                labels[i] = "A23"
            elif l == "A34":
                labels[i] = "A33"
            elif l == "A35":
                labels[i] = "A33"
            elif l == "A41":
                labels[i] = "A40"
            elif l == "A42":
                labels[i] = "A40"
            elif l == "C5":
                labels[i] = "C4"
            elif l == "C6":
                labels[i] = "C4"
            elif l == "C14":
                labels[i] = "C13"
            elif l == "C15":
                labels[i] = "C13"
            elif l == "C35":
                labels[i] = "C34"
            elif l == "C36":
                labels[i] = "C34"
            elif l == "C42":
                labels[i] = "C41"
            elif l == "C43":
                labels[i] = "C41"
            elif l == "D5":
                labels[i] = "D4"
            elif l == "D6":
                labels[i] = "D4"
            elif l == "D14":
                labels[i] = "D13"
            elif l == "D15":
                labels[i] = "D13"
            elif l == "D36":
                labels[i] = "D35"
            elif l == "D37":
                labels[i] = "D35"
            elif l == "D43":
                labels[i] = "D42"
            elif l == "D44":
                labels[i] = "D42"
            elif l == "E5":
                labels[i] = "E4"
            elif l == "E6":
                labels[i] = "E4"
            elif l == "E14":
                labels[i] = "E13"
            elif l == "E15":
                labels[i] = "E13"
            elif l == "E39":
                labels[i] = "E38"
            elif l == "E40":
                labels[i] = "E38"
            elif l == "E46":
                labels[i] = "E45"
            elif l == "E47":
                labels[i] = "E45"
            elif l == "F5":
                labels[i] = "F4"
            elif l == "F6":
                labels[i] = "F4"
            elif l == "F14":
                labels[i] = "F13"
            elif l == "F15":
                labels[i] = "F13"
            elif l == "F44":
                labels[i] = "F43"
            elif l == "F45":
                labels[i] = "F43"
            elif l == "F51":
                labels[i] = "F50"
            elif l == "F52":
                labels[i] = "F50"
            elif l == "G5":
                labels[i] = "G4"
            elif l == "G6":
                labels[i] = "G4"
            elif l == "G14":
                labels[i] = "G13"
            elif l == "G15":
                labels[i] = "G13"
            elif l == "G31":
                labels[i] = "G30"
            elif l == "G32":
                labels[i] = "G30"
            elif l == "G38":
                labels[i] = "G37"
            elif l == "G39":
                labels[i] = "G37"
            elif l == "H5":
                labels[i] = "H4"
            elif l == "H6":
                labels[i] = "H4"
            elif l == "H14":
                labels[i] = "H13"
            elif l == "H15":
                labels[i] = "H13"
            elif l == "H42":
                labels[i] = "H41"
            elif l == "H43":
                labels[i] = "H41"
            elif l == "H49":
                labels[i] = "H48"
            elif l == "H50":
                labels[i] = "H48"
            elif l == "I5":
                labels[i] = "I4"
            elif l == "I6":
                labels[i] = "I4"
            elif l == "I14":
                labels[i] = "I13"
            elif l == "I15":
                labels[i] = "I13"
            elif l == "I28":
                labels[i] = "I27"
            elif l == "I29":
                labels[i] = "I27"
            elif l == "I34":
                labels[i] = "I33"
            elif l == "I35":
                labels[i] = "I33"
            elif l == "I43":
                labels[i] = "I42"
            elif l == "I44":
                labels[i] = "I42"
            elif l == "I50":
                labels[i] = "I49"
            elif l == "I51":
                labels[i] = "I49"
            elif l == "K5":
                labels[i] = "K4"
            elif l == "K6":
                labels[i] = "K4"
            elif l == "K14":
                labels[i] = "K13"
            elif l == "K15":
                labels[i] = "K13"
            elif l == "K37":
                labels[i] = "K36"
            elif l == "K38":
                labels[i] = "K36"
            elif l == "K46":
                labels[i] = "K45"
            elif l == "K47":
                labels[i] = "K45"
            elif l == "K53":
                labels[i] = "K52"
            elif l == "K54":
                labels[i] = "K52"
            elif l == "L5":
                labels[i] = "L4"
            elif l == "L6":
                labels[i] = "L4"
            elif l == "L14":
                labels[i] = "L13"
            elif l == "L15":
                labels[i] = "L13"
            elif l == "L31":
                labels[i] = "L30"
            elif l == "L32":
                labels[i] = "L30"
            elif l == "L34":
                labels[i] = "L33"
            elif l == "L35":
                labels[i] = "L33"
            elif l == "L43":
                labels[i] = "L42"
            elif l == "L44":
                labels[i] = "L42"
            elif l == "L50":
                labels[i] = "L49"
            elif l == "L51":
                labels[i] = "L49"
            elif l == "M5":
                labels[i] = "M4"
            elif l == "M6":
                labels[i] = "M4"
            elif l == "M14":
                labels[i] = "M13"
            elif l == "M15":
                labels[i] = "M13"
            elif l == "M32":
                labels[i] = "M31"
            elif l == "M33":
                labels[i] = "M31"
            elif l == "M41":
                labels[i] = "M40"
            elif l == "M42":
                labels[i] = "M40"
            elif l == "M48":
                labels[i] = "M47"
            elif l == "M49":
                labels[i] = "M47"
            elif l == "N5":
                labels[i] = "N4"
            elif l == "N6":
                labels[i] = "N4"
            elif l == "N14":
                labels[i] = "N13"
            elif l == "N15":
                labels[i] = "N13"
            elif l == "N30":
                labels[i] = "N29"
            elif l == "N38":
                labels[i] = "N37"
            elif l == "N39":
                labels[i] = "N37"
            elif l == "N45":
                labels[i] = "N44"
            elif l == "N46":
                labels[i] = "N44"
            elif l == "P5":
                labels[i] = "P4"
            elif l == "P6":
                labels[i] = "P4"
            elif l == "P14":
                labels[i] = "P13"
            elif l == "P15":
                labels[i] = "P13"
            elif l == "P38":
                labels[i] = "P37"
            elif l == "P39":
                labels[i] = "P37"
            elif l == "P42":
                labels[i] = "P41"
            elif l == "P43":
                labels[i] = "P41"
            elif l == "Q5":
                labels[i] = "Q4"
            elif l == "Q6":
                labels[i] = "Q4"
            elif l == "Q14":
                labels[i] = "Q13"
            elif l == "Q15":
                labels[i] = "Q13"
            elif l == "Q33":
                labels[i] = "Q32"
            elif l == "Q41":
                labels[i] = "Q40"
            elif l == "Q42":
                labels[i] = "Q40"
            elif l == "Q48":
                labels[i] = "Q47"
            elif l == "Q49":
                labels[i] = "Q47"
            elif l == "R5":
                labels[i] = "R4"
            elif l == "R6":
                labels[i] = "R4"
            elif l == "R14":
                labels[i] = "R13"
            elif l == "R15":
                labels[i] = "R13"
            elif l == "R38":
                labels[i] = "R37"
            elif l == "R39":
                labels[i] = "R37"
            elif l == "R40":
                labels[i] = "R37"
            elif l == "R41":
                labels[i] = "R37"
            elif l == "R48":
                labels[i] = "R47"
            elif l == "R49":
                labels[i] = "R47"
            elif l == "R55":
                labels[i] = "R54"
            elif l == "R56":
                labels[i] = "R54"
            elif l == "S5":
                labels[i] = "S4"
            elif l == "S6":
                labels[i] = "S4"
            elif l == "S14":
                labels[i] = "S13"
            elif l == "S15":
                labels[i] = "S13"
            elif l == "S35":
                labels[i] = "S34"
            elif l == "S36":
                labels[i] = "S34"
            elif l == "S42":
                labels[i] = "S41"
            elif l == "S43":
                labels[i] = "S41"
            elif l == "T5":
                labels[i] = "T4"
            elif l == "T6":
                labels[i] = "T4"
            elif l == "T14":
                labels[i] = "T13"
            elif l == "T15":
                labels[i] = "T13"
            elif l == "T28":
                labels[i] = "T27"
            elif l == "T29":
                labels[i] = "T27"
            elif l == "T38":
                labels[i] = "T37"
            elif l == "T39":
                labels[i] = "T37"
            elif l == "T45":
                labels[i] = "T44"
            elif l == "T46":
                labels[i] = "T44"
            elif l == "V5":
                labels[i] = "V4"
            elif l == "V6":
                labels[i] = "V4"
            elif l == "V14":
                labels[i] = "V13"
            elif l == "V15":
                labels[i] = "V13"
            elif l == "V28":
                labels[i] = "V27"
            elif l == "V29":
                labels[i] = "V27"
            elif l == "V31":
                labels[i] = "V30"
            elif l == "V32":
                labels[i] = "V30"
            elif l == "V40":
                labels[i] = "V39"
            elif l == "V41":
                labels[i] = "V39"
            elif l == "V47":
                labels[i] = "V46"
            elif l == "V48":
                labels[i] = "V46"
            elif l == "W5":
                labels[i] = "W4"
            elif l == "W6":
                labels[i] = "W4"
            elif l == "W14":
                labels[i] = "W13"
            elif l == "W15":
                labels[i] = "W13"
            elif l == "W48":
                labels[i] = "W47"
            elif l == "W49":
                labels[i] = "W47"
            elif l == "W55":
                labels[i] = "W54"
            elif l == "W56":
                labels[i] = "W54"
            elif l == "Y5":
                labels[i] = "Y4"
            elif l == "Y6":
                labels[i] = "Y4"
            elif l == "Y14":
                labels[i] = "Y13"
            elif l == "Y15":
                labels[i] = "Y13"
            elif l == "Y45":
                labels[i] = "Y44"
            elif l == "Y46":
                labels[i] = "Y44"
            elif l == "Y52":
                labels[i] = "Y51"
            elif l == "Y53":
                labels[i] = "Y51"
        data.labels = labels.copy()

def add_to_queue(input,inp_settings,settings, data, jobs1, jobs2, skip, job_server, opt_params_dict):
    # Limit more that N_max crossvalidation runs of kernel parameters at a time.
    # Otherwise memory is easily filled.
    # Setting this requirement won't hinder cpu utilization
    # unless a large number of nodes is used.
    # N_max is set to 2.0*max_cpus here
    N_max = sum([i for i in job_server.get_active_nodes().values()])
    N_current = len([i for j in jobs2 for i in jobs2[j]])
    try:
        N_current += len([i for j in jobs1 for k in jobs1[j] for i in jobs1[j][k]])
    except KeyError:
        N_current += len([i for j in jobs1 for i in jobs1[j]])
    if settings.verbose >= 2:
        print "jobs queued: %d.   max cpus: %d" % (N_current, N_max)
    if N_current > N_max*2.0:
        return
    # skip tracks all jobs that has been queued
    for i, inp in (tup for tup in enumerate(input) if tup[0] not in skip):
        inp_settings[i] = utils.create_settings(inp, settings)
        this_setting = copy.deepcopy(inp_settings[i])

        # submit creation of data with given settings
        # use that the data might be the same as a previous calculation
        md5_data = this_setting.md5_data
        if md5_data in data.keys():
            skip.append(i)
            pass
        else:
            if settings.verbose >= 2:
                print "Reading data with md5 = %s" % md5_data
            merge = merge_labels(this_setting)
            data[md5_data] = job_server.submit(utils.make_data, args=(this_setting,), 
                       modules=("glob", "from ML_SK.parser import atomic,common",
                                "sklearn.feature_selection", "from ML_SK import utils",
                                "from ML_SK.utils import Data,cluster"),
                       callback=merge.do_merge)
            skip.append(i)


        # submit everything
        # check if the cross validated kernel parameters have already been calculated
        params = check_kernel_parameters(this_setting)

        if params:
            jobs2[i] = {}
            # change the settings object such that the full prediction is returned
            this_setting.metric = "full"
            for n in range(settings.testK):
                x_test, x_val, x_train = data[md5_data]().get(data[md5_data]().x(),n)
                y_test, y_val, y_train = data[md5_data]().get(data[md5_data]().y()[:,this_setting.observable_index],n)
                size = x_train.shape[0]+x_val.shape[0]
                idx = np.random.choice(np.arange(size),size=min(this_setting.train_size,size),replace=False)
                x_train = np.concatenate([x_train,x_val])[idx]
                y_train = np.concatenate([y_train,y_val])[idx]
                x_test = x_test[:this_setting.test_size]
                y_test = y_test[:this_setting.test_size]
                if i not in opt_params_dict: opt_params_dict[i] = {}
                opt_params_dict[i][n] = params[n]
                if settings.verbose >= 2:
                    print "Reading optimal kernel parameters %s for job %d,%d" % (opt_params_dict[i][n], i, n)
                jobs2[i][n] = job_server.submit(utils.KRR,
                        args=(opt_params_dict[i][n], this_setting, x_train, x_test, y_train, y_test),
                        #depfuncs=(do_KRR,KRR_laplace,predict),
                    modules = ("numpy as np","scipy.optimize", "sklearn.kernel_ridge","warnings","from ML_SK import utils","from ML_SK.utils import Data,cluster"))
        else:
            if settings.verbose >= 2:
                print "Submitting job %d with md5 = %s" % (i, md5_data)
            data[md5_data]()
            jobs1[i] = process_cv(data[md5_data](), this_setting, job_server)
        break

def check_kernel_parameters(settings):
    exists = True
    params = []
    for n in range(settings.testK):
        md5 = settings.get_md5_kernel_parameters(n)
        md5_kernel_parameters_path = settings.kernel_parameters_folder + "/" + md5 + ".pickle"
        if os.path.isfile(md5_kernel_parameters_path):
            with open(md5_kernel_parameters_path) as f:
                params.append(pickle.load(f))
        else:
            exists = False
            break
    if exists:
        return params
    else:
        return False


def process_all(d, settings, job_server):
    # TODO: should include some preprocessing
    if settings.verbose >= 2:
        print d
    input = [dict(zip(d, v)) for v in product(*d.values())]

    del_idx = []
    # remove bad combinations
    for i,d in enumerate(input):
        # remove training sizes larger than the amount of data
        if d["target_element"] == "H":
            if d["train_size"] > 48618*(settings.testK-1.0)/settings.testK:
                del_idx.append(i)
        elif d["target_element"] == "C":
            if d["train_size"] > 28001*(settings.testK-1.0)/settings.testK:
                del_idx.append(i)
        elif d["target_element"] == "N":
            if d["train_size"] > 8695*(settings.testK-1.0)/settings.testK:
                del_idx.append(i)

        # cut_off2 needs to be between cut_off and 2*cut_off
        if ("cut_off2" in d.keys()):
            if (d["cut_off2"] >= 2*d["cut_off"] and d["cut_off"] < 900) or (d["cut_off2"] <= d["cut_off"]):
                del_idx.append(i)


    for i in del_idx[::-1]:
        input.pop(i)


    jobs1 = {}
    jobs2 = {}
    data = {}
    results = []
    inp_settings = np.empty(len(input),dtype=object)
    skip = []
    opt_params_dict = {}


    # next round of jobs
    # iteratively add jobs to queue as dependant jobs are completed
    flag = True # is a job still running?
    c = 0
    while flag == True:
        # read as much data as the memory can handle
        add_to_queue(input,inp_settings,settings, data, jobs1, jobs2, skip, job_server, opt_params_dict)
        c += 1
        if c % 25 == 0 and settings.verbose >= 2:
            print job_server.print_stats()
        flag = False
        # here i is different jobs
        for i in jobs1.keys():
            # make a copy here such that we can make temporary changes in
            # the settings later
            this_setting = copy.deepcopy(inp_settings[i])
            md5 = this_setting.md5_data
            flag = True
            if len(jobs1[i].keys()) == 0:
                if settings.verbose >= 1:
                    print "deleting jobs1",i
                del jobs1[i]
                ## if no further jobs will use the current data object, delete it
                #if md5 not in (inp_settings[i].md5_data for i in jobs1.keys()):
                #    if settings.verbose >= 1:
                #        mem = sys.getsizeof(pickle.dumps(data[md5]()))*1024.**-3
                #        print "deleting data object with md5:",md5
                #        print "memory regained: %.2f" % mem
                #    del data[md5]
                continue
            # here n is the testing step of crossvalidation for the given job
            for n in jobs1[i].keys():
                # if all jobs for i, n is finished, continue next step
                if reduce(operator.mul, [job.finished for job in jobs1[i][n].values()]):
                    params = [job() for job in jobs1[i][n].values()]
                    # from the validation step, select the median optimized
                    # parameters for the testing step
                    opt_params = np.median(params, axis=0)
                    if i not in opt_params_dict: opt_params_dict[i] = {}
                    opt_params_dict[i][n] = opt_params
                    # save these for future calcs
                    if this_setting.save_kernel_parameters:
                        md5_kernel_parameters = this_setting.get_md5_kernel_parameters(n)
                        md5_kernel_parameters_path = settings.kernel_parameters_folder + "/" + md5_kernel_parameters + ".pickle"
                        if not os.path.isfile(md5_kernel_parameters_path):
                            if settings.verbose >= 2:
                                print "saving optimal parameters for:",i,n
                            with open(md5_kernel_parameters_path, "w") as f:
                                pickle.dump(opt_params, f)
                    if settings.verbose >= 2:
                        print "deleting jobs1",i,n, "with params:", params
                    del jobs1[i][n]
                else:
                    continue

                if settings.verbose >= 1:
                    print "optimal parameter values for job %d,%d:" % (i, n), opt_params
                x_test, x_val, x_train = data[md5]().get(data[md5]().x(),n)
                y_test, y_val, y_train = data[md5]().get(data[md5]().y()[:,this_setting.observable_index],n)
                #l_test = data[md5]().get(data[md5]().y()[:,settings.label_index],n)[0]
                size = x_train.shape[0]+x_val.shape[0]
                idx = np.random.choice(np.arange(size),size=min(this_setting.train_size,size),replace=False)
                x_train = np.concatenate([x_train,x_val])[idx]
                y_train = np.concatenate([y_train,y_val])[idx]
                x_test = x_test[:this_setting.test_size]
                y_test = y_test[:this_setting.test_size]
                #l_test = l_test[:this_setting.test_size]

                if i not in jobs2.keys(): jobs2[i] = {}

                # change the settings object such that the full prediction is returned
                this_setting.metric = "full"
                # TODO remove hardcoding
                jobs2[i][n] = job_server.submit(utils.KRR,
                        args=(opt_params, this_setting, x_train, x_test, y_train, y_test),
                        #depfuncs=(do_KRR,KRR_laplace,predict),
                    modules = ("numpy as np","scipy.optimize", "sklearn.kernel_ridge","warnings","from ML_SK import utils","from ML_SK.utils import Data"))

        for i in (j for j in jobs2.keys() if j not in jobs1.keys()):
            this_setting = inp_settings[i]
            md5 = this_setting.md5_data
            flag = True
            # if all jobs for i is finished, continue next step
            if reduce(operator.mul, [job.finished for job in jobs2[i].values()]):
                #out = [job() for job in jobs2[i].values()]
                mae, rmsd = [], []
                for n in jobs2[i].keys():
                    y_test, y_pred = jobs2[i][n]()
                    l_test = data[md5]().get(data[md5]().y()[:,settings.label_index],n)[0][:this_setting.test_size]
                    mae.append(sklearn.metrics.mean_absolute_error(y_pred, y_test))
                    rmsd.append(sklearn.metrics.mean_squared_error(y_pred, y_test))
                    if this_setting.save_prediction:
                        utils.save_dataframe(this_setting, y_test, y_pred, l_test, opt_params_dict[i][n])
                # if no further jobs will use the current data object, delete it
                if md5 not in (inp_settings[k].md5_data for k in jobs2.keys()):
                    if settings.verbose >= 1:
                        print "deleting data object with md5:",md5
                    del data[md5]

                if settings.verbose >= 1:
                    print "deleting jobs2",i
                    print "mae:", np.median(mae), mae
                    print "rmsd:", np.median(rmsd), rmsd
                del jobs2[i]
                median_metrics = np.median(mae), np.median(rmsd)
                results.append((this_setting, median_metrics))

        if flag == False:
            break

        time.sleep(5)

    return results


# TODO: do preprocess transform
def process_job_cv_val_step(settings, data, n, m):
    x_test, x_val, x_train = data.get(data.x(),n,m)
    y_test, y_val, y_train = data.get(data.y()[:,settings.observable_index],n,m)
    ##l_test, l_val, l_train = data.get(data.y[:,settings.label_index],n,m)
    this_x_train = x_train[:settings.cv_train_size]
    this_y_train = y_train[:settings.cv_train_size]
    # combine the leftovers from training set in validation
    # to reduce variance of the estimate
    # TODO speedtest this
    this_x_val = np.concatenate([x_val,x_train[settings.cv_train_size:]])[:settings.cv_val_size]
    this_y_val = np.concatenate([y_val,y_train[settings.cv_train_size:]])[:settings.cv_val_size]
    #this_x_val = x_val[:settings.cv_val_size]
    #this_y_val = y_val[:settings.cv_val_size]

    return utils.fit_params(settings, this_x_train, this_x_val, this_y_train, this_y_val)


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
                    #depfuncs = (utils.fit_params,utils.fit_KRRparams, utils.opt),
                    #depfuncs = (utils.const,),
                    modules = ("numpy as np","scipy.optimize", "sklearn.kernel_ridge","warnings","from ML_SK import utils"))

    return jobs

# TODO correct for RC Anders
# TODO multidimensional regression

if __name__ == '__main__':

    #jobid = parallel.start_workers(5)
    #job_server = parallel.start_server(8)
    if sys.argv[-2] == "CA" and sys.argv[-1] != "4":
        quit()
    settings = init_settings()
    settings.cut_off = float(sys.argv[6])
    settings.target_element = "C"

    settings.data_folder              = '/home/lab/cstein_cs/trajpdb/small'
    settings.save_data_pickle         = 0
    settings.descriptor               = "SortedCoulombMatrix"
    settings.force_central_first      = True
    settings.save_prediction          = 0
    settings.save_kernel_parameters   = 0
    settings.max_neighbours           = (550*int(sys.argv[-1])**2)/12**2
    settings.md5_data = settings.get_md5_data()
    data = utils.make_data(settings)
    quit()
    #settings.label_index              = 3
    #settings.verbose                  = 2
    #settings.testK                    = 15
    #settings.valK                     = 5
    #settings.observable_index         = 0
    #settings.baseline_index           = 5

    # TODO move this to plotting
    #if do_plots:
    #    settings.observable_index = 0

    #    for atom in ["C","N","H"]:
    #        settings.target_element = atom
    #        do_initial_plots(settings, 5, 6, job_server)

    d = {}
        #self.cut_off = 1.0 to 10.0
        #self.cut_off2 = 1000. # second cut_off that works on atoms on opposite sides of the central atom
        #self.descriptor = 'CoulombMatrix' # The base descriptor (CoulombMatrix, SortedCoulombMatrix, RandomSortedCoulombMatrix)
        #self.regressor = 'KernelRidge' # Algorithm to the the predicitons (KernelRidge, ...)
        #self.kernel = 'laplacian' # Kernel in KernelRidge (laplacian, ...)
        #self.exponent = 1 # exponent of 1/r in the coulomb matrix. Should probably be betweet 1 and 12
        #self.sncf = False # Use of the british railroad metric. For central atom j: 1/Rijk -> 1/(Rij + Rjk + Rik)
        #self.reduce = False # Dimensionality reduction during parsing (bob, bobd, eig, red, False)
        #self.mulliken = None # Use mulliken charges instead of static charges if available. (None, mulliken_charge_index) # NOTE: the index can't be 0
        #self.dampening = None # functions for dampening (...)
        #self.damp_const1 = 1 # parameter in dampening. Typically the width of the dempening potential.
        #self.damp_const2 = 1 # parameter in dampening. Typically the distance over where the potential is dampened. lower_cut_off = cut_off - dr
        #self.dampening2 = None # functions for dampening (...)
        #self.damp2_const1 = 1 # parameter in dampening. Typically the width of the dempening potential.
        #self.damp2_const2 = 1 # parameter in dampening. Typically the distance over where the potential is dampened. lower_cut_off = cut_off - dr
        #self.cm_cut = 0 # set values in the matrix lower than this to zero
        #self.force_central_first = True # Force the central atom to appear first even in the sorted matrices
        #self.random = 0 # number of random matrices to create
        #self.random_sigma = 1 # stdev of the random variation in the l2norm of the rnadom matrices
        #self.self_energy_param = (0.5, 2.4, 0, 1)# functional form of the diagonal elements of the coulomb matrix
        #self.observable_index = 0 # index for the observable to be predicted.
        #self.metric = "mae" # use mae or rmsd for optimizing parameters

    #settings.save_data_pickle         = 0
    #settings.save_prediction          = 0
    #settings.save_kernel_parameters   = 0
    #settings.output_folder   = "/home/lab/dev/ML_SK/output/
    #settings.data_folder              = '/home/lab/dev/ML_SK/data/hf_delta'
    d['cv_train_size'] = [3000]
    d['cv_val_size'] = [50000]
    d['test_size'] = [50000]

    d['target_element'] = ["C","N","H"]
    #d['train_size'] = [200,350,600,1000,2000,3500,6800,11000,22000,33000]
    d['train_size'] = [int(sys.argv[2])]
    d['cut_off'] = [float(sys.argv[1])]#[1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,8.0,8.5,9.0,9.5,10.0]#,3.5,4.5,5.5]#,3.5,4.5,5.5]#,3.5,4.5,5.5,6.5,7.5]#,3.0,4.0,5.0]#,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0]
    #d['cut_off2'] = [7.5,6.5,5.5,4.5]#, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0]
    #d['descriptor'] = ['CoulombMatrix']#, 'SortedCoulombMatrix', 'RandomSortedCoulombMatrix']
    #d['regressor'] = ['KernelRidge'] # optparam
    #d['kernel'] = ['laplacian']#, 'generalized_normal', 'matern1', 'matern3', 'matern5'] #optparam
    #d['exponent'] = range(2,12)
    #d['sncf'] = [True]
    #d['reduce'] = ["red"]#, "bobd", "eig", "red", False]
    #d['mulliken'] = [4]
    #d['dampening'] = [None, "smooth", "cos", "norm", "laplace", "smooth_norm", "smooth_laplace"]
    #d['dampening2'] = [None, "smooth", "cos", "norm", "laplace", "smooth_norm", "smooth_laplace"]
    #d['damp_const1'] = [1, 1e-1, 1e-2, 1e-3, 1e-4]
    #d['damp2_const1'] = [1, 1e-1, 1e-2, 1e-3, 1e-4]
    #d['damp_const1'] = [2., 1.5, 1., 0.5]
    #d['damp2_const1'] = [2., 1.5, 1., 0.5]
    #d['cm_cut'] = [0, 0.1, 0.3,1.0, 3, 10]
    #d['force_central_first'] = True, False
    #d['random'] = [0, 2, 4, 8]
    #d['random_sigma'] = [1e-3, 1e-2, 1e-1, 1, 10]
    #d['self_energy_param'] = [(0.5, 2.4, -0.2, 2.0)]
    #d['metric'] = ["rmsd"]


    results = process_all(d, settings, job_server)
    #for i, (s, j) in enumerate(results):
    #    print i, j,
    #    print s.__dict__['cv_train_size'],
    #    print s.__dict__['cv_val_size'],
    #    print s.__dict__['test_size'],
    #    print ""

    #### crossval
    ##### regressor

    ### Different preprocessing
    #kpreprocessing(data,job_server)

    job_server.print_stats()
    # kill slurm jobid
    #parallel.kill_workers(jobid)
    # kill workers that the server can find
    job_server.destroy()
