
import numpy as np
import os,sys
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import pandas
import ML_SK.plotting.plots as plots
import scipy.interpolate
import statsmodels.stats.api as sms
import copy

# pandas output options
pandas.set_option('display.max_rows', 5000)
pandas.set_option('display.max_columns', 50)
pandas.set_option('display.max_colwidth', 300)
pandas.set_option('display.width', 1000)

# matplotlib latex font
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rc('text', usetex=True)


############################################################################
########################## Some predefined plots ###########################
############################################################################


def scatter_kde_plot(X,Y, classification, xlabel1="", xlabel2="", ylabel="", rowlabels=None, filename="img"):
    import seaborn as sns
    '''
    will create subplot of size (K,2)

    X = 1darray of size N
    Y = lists of size K containing ndarrays of size N
    classification = 1darray of size N with classification indices
    '''

    # seaborn style
    sns.set_style("whitegrid",{'grid.color':'.92','axes.edgecolor':'0.92'})

    # number of plots
    K = len(Y)

    num = (K-1)/4+1

    for n in range(num):
        f, axes = plt.subplots(4,2, sharex="col", sharey="col")

        for i in range(n*4,min((n+1)*4,K)):
            x = [X[classification[i] == j] for j in range(i+1)]
            y = [Y[i][classification[i] == j] for j in range(i+1)]
            plots.plot_points(x,y,axes[i-n*4,0], plot_line=True, inp="list")
            plots.plot_kde(Y[i]-X, axes[i-n*4,1])
            axes[i-n*4,0].set_ylabel(ylabel)
        axes[min([i-n*4,K-1]),0].set_xlabel(xlabel1)
        axes[min([i-n*4,K-1]),1].set_xlabel(xlabel2)

        # from http://stackoverflow.com/a/25814386/2653663
        if rowlabels != None:
            pad = 15
            for ax, row in zip(axes[:,0], rowlabels[n*4:K-n*4]):
                ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                            xycoords=ax.yaxis.label, textcoords="offset points",
                            size="large", ha="right", va="center")

        # this is a bit of a mess. Should set the size of the figure at the top
        f.tight_layout(pad=0.0,h_pad=0.5,w_pad=0.0,
                rect=(0.0+pad/1000.+0.02*max([len(t) for t in rowlabels]),0,1,1))
        plt.savefig(filename+"_"+str(n))
        plt.clf()

def heatmap(pickle_names):
    #import seaborn as sns
    #sns.set_style("white")
    #cmap = sns.diverging_palette(240,10, as_cmap=True)
    df = pandas.read_pickle(pickle_names[0])
    #selection = True
    #for i in df:
    #    if i not in ["pickle", 'cut_off', 'cut_off2', 'data_folder',
    #            'kernel_parameters', 'mae', 'observable_index',
    #            'rmsd', 'target_element', 'train_size', "remove_labels",
    #            "exponent","sncf", "cm_cut"]:
    #        quit("Set %s in heatmap plot" % i)
    #    else:
    #        if i == "cut_off2":
    #            selection = selection & (df.cut_off2 == 1000.)
    #        elif i == "data_folder":
    #            selection = selection & (df.data_folder == '/home/lab/dev/ML_SK/data/delta')
    #        elif i == "observable_index":
    #            selection = selection & (df.observable_index == 0)
    #        elif i == "remove_labels":
    #            selection = selection & (~df.remove_labels.isnull())
    #        elif i == "exponent":
    #            selection = selection & (df.exponent == 1)
    #        elif i == "sncf":
    #            selection = selection & (df.sncf == False)
    #        elif i == "cm_cut":
    #            selection = selection & (df.cm_cut == 0.)


    ## NOTE: choose different data/settings here
    #df = df.loc[selection]
    #
    #for i in df.pickle.values:
    #    print "output/" + i + ".pickle"

    #print pandas.unique(df.exponent.values)
    #df = df.loc[(df.sncf == False) & (df.exponent == 12)]

    for ele in np.unique(df["target_element"].values):
        x1,x2,x3,x4,y1,y2,y3,y4,z1,z2,z3,z4 = [],[],[],[],[],[],[],[],[],[],[],[]
        for tsize in np.unique(df.loc[df.target_element == ele].train_size.values):
            for cut in np.unique(df["cut_off"].values):
                this_df = df.loc[(df.target_element == ele)
                               & (df.train_size == tsize)
                               & (df.cut_off == cut)].drop(["target_element",
                                                            "train_size",
                                                            "cut_off"],axis=1)
                x1.extend([tsize for _ in range(len(np.unique(this_df.mae.values)))])
                y1.extend([cut for _ in range(len(np.unique(this_df.mae.values)))])
                z1.extend(list(np.unique(this_df.mae.values)))
                x2.extend([tsize for _ in range(len(np.unique(this_df.rmsd.values)))])
                y2.extend([cut for _ in range(len(np.unique(this_df.rmsd.values)))])
                z2.extend(list(np.unique(this_df.rmsd.values)))
                if np.unique(this_df.mae.values).size > 0:#in [5,15]:
                    x3.append(tsize)
                    x4.append(tsize)
                    y3.append(cut)
                    y4.append(cut)
                    z3.append(np.median(np.unique(this_df.mae.values)))
                    z4.append(np.median(np.unique(this_df.rmsd.values)))
                    print "heatmap:", ele, tsize, cut, z3[-1], z4[-1]
                elif np.unique(this_df.mae.values).size == 0:
                    #print "heatmap: missing:", ele, tsize, cut
                    continue
                elif (ele, tsize, cut) in [("H",33000,6.5)]:
                    continue
                else:
                    print ele, tsize, cut, np.unique(this_df.mae.values).size
                    print this_df.head(30)
                    for i in this_df.pickle.values:
                        print i
                    quit("heatmap value size error")
                #for i in this_df.pickle.values:
                #    print "output/%s.pickle" % i
        #continue

        x1,x2,x3,x4,y1,y2,y3,y4,z1,z2,z3,z4 = [np.asarray(i) for i in [x1,x2,x3,x4,y1,y2,y3,y4,z1,z2,z3,z4]]


        #for t, f in [("rbf","linear"),("rbf","multiquadric"), ("rbf","inverse"),("rbf","gaussian"),("rbf","cubic"),("rbf","quintic"),("rbf","thin_plate"), \
        #             ("int","linear"),("int","cubic")]:
        for t,f in [("int","linear")]:
            if t == "int":
                c=0
                #for X,Y,Z in [(x1,y1,z1), (x2,y2,z2), (x3,y3,z3), (x4,y4,z4)]:
                for X,Y,Z in [(x3,y3,z3)]:
                    x = np.log10(X)
                    y = Y
                    z = np.log10(Z)
                    # Set up a regular grid of interpolation points
                    xi, yi = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
                    fig = plt.Figure(dpi=600,figsize=(16,12), frameon=False)
                    FigureCanvas(fig) # to be able to save as pdf
                    ax = fig.add_subplot(111)
                    # Interpolate
                    zi = scipy.interpolate.griddata(np.asarray(zip(x,y)),z,(xi,yi),method=f).T

                    im = ax.imshow(10**(zi), origin='lower', #cmap=matplotlib.coolwarm,
                                    extent=[x.min(), x.max(), y.min(), y.max()], aspect='auto')
                    cbar = fig.colorbar(im)
                    cbar.ax.set_xlabel(r'MAE in ppm')
                    ax.set_xlabel(r'Training set size')
                    ax.set_ylabel(r'Cutoff distance in $\mathrm \AA$')

                    # set limits
                    ax.set_xlim([2.2,ax.get_xlim()[1]])
                    ax.set_ylim([1.2,ax.get_ylim()[1]])

                    # set ticks
                    if ele == "H":
                        cbar.set_ticks(np.arange(0.06,0.18,0.02))
                    if ele == "C":
                        cbar.set_ticks(np.arange(0.4,1.6,0.2))
                    if ele == "N":
                        ax.set_xticks(np.arange(2.5,4.0,0.5))

                    # change tick labels
                    xlabels = ax.get_xticks().tolist()
                    ylabels = ax.get_yticks().tolist()
                    xlabels = [r'$10^{%s}$' % str(i) for i in xlabels]
                    if ele == "C":
                        xlabels[0] = ""
                        ylabels[0] = ""
                    if ele == "H":
                        xlabels[-2:] = ""
                    ax.set_xticklabels(xlabels)
                    ax.set_yticklabels(ylabels)

                    # remove top and right spine
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)

                    # Only show ticks on the left and bottom spines
                    ax.yaxis.set_ticks_position('left')
                    ax.xaxis.set_ticks_position('bottom')
                    im.axes.get_yaxis().set_ticks_position('left')
                    im.axes.get_xaxis().set_ticks_position('bottom')

                    # change text size
                    ax.xaxis.label.set_fontsize(28)
                    ax.yaxis.label.set_fontsize(28)
                    cbar.ax.xaxis.label.set_fontsize(23)
                    for i in ax.get_xticklabels() + ax.get_yticklabels():
                        i.set_fontsize(22)
                    for i in cbar.ax.get_yticklabels():
                        i.set_fontsize(21)

                    # fit a simple function to the minimum
                    xopt = scipy.optimize.minimize(opt_heatmap_line, [1.0,2.2,0.02], args=(x,y,z),
                            method='Nelder-Mead')#,tol=1e-4)#options={'ftol':settings.tol,'xtol':settings.tol})
                    print "f:", xopt.fun, xopt.x

                    xi = np.arange(x.min(),x.max(),0.05)
                    yi = xopt.x[2]*np.exp(xi*xopt.x[0])+xopt.x[1]
                    ax.plot(xi,yi,"--",c="w",linewidth=1.0)


                    fig.savefig("%s_%d_%s_%s.pdf" % (ele,c,t,f), bbox_inches='tight')
                    c+=1
                    fig.clf()
            elif t == "rbf":
                for smooth in [0.01]:#,0.1,0.25,0.5,1,2,5,10,50]:
                    c=0
                    for X,Y,Z in [(x3,y3,z3)]:
                        try:
                            x = np.log10(X)
                            y = Y
                            z = np.log10(Z)
                            # Set up a regular grid of interpolation points
                            xi, yi = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
                            fig = plt.Figure(dpi=600,figsize=(8,6), frameon=False)
                            FigureCanvas(fig) # to be able to save as pdf
                            ax = fig.add_subplot(111)
                            # Interpolate
                            zi = scipy.interpolate.Rbf(x, y, z, function=f, smooth=smooth)(xi,yi).T
                            #zi = scipy.interpolate.griddata(np.asarray(zip(x,y)),z,(xi,yi),method="cubic").T

                            im = ax.imshow(10**(zi), origin='lower', #cmap=matplotlib.coolwarm,
                                            extent=[x.min(), x.max(), y.min(), y.max()], aspect='auto')
                            cbar = fig.colorbar(im)
                            cbar.ax.set_xlabel(r'MAE in ppm')
                            ax.set_xlabel(r'Training set size')
                            ax.set_ylabel(r'Cutoff distance in $\mathrm \AA$')

                            # change tick labels
                            ax.set_xticklabels([r'$10^{%s}$' % str(i) for i in (ax.get_xticks().tolist())])

                            # remove top and right spine
                            ax.spines['right'].set_visible(False)
                            ax.spines['top'].set_visible(False)
                            ax.spines['left'].set_visible(False)
                            ax.spines['bottom'].set_visible(False)

                            # Only show ticks on the left and bottom spines
                            ax.yaxis.set_ticks_position('left')
                            ax.xaxis.set_ticks_position('bottom')
                            im.axes.get_yaxis().set_ticks_position('left')
                            im.axes.get_xaxis().set_ticks_position('bottom')

                            # fit a simple function to the minimum
                            xopt = scipy.optimize.minimize(opt_heatmap_line, [1.0,2.2,0.02], args=(x,y,z),
                                    method='Nelder-Mead')#,tol=1e-4)#options={'ftol':settings.tol,'xtol':settings.tol})
                            print "f:", xopt.fun, xopt.x
                            xi = np.arange(x.min(),x.max(),0.05)
                            yi = xopt.x[2]*np.exp(xi*xopt.x[0])+xopt.x[1]
                            ax.plot(xi,yi,"--",c="w",linewidth=1.0)

                            fig.savefig("%s_%d_%s_%s_%.1f.pdf" % (ele,c,t,f,smooth), bbox_inches='tight')
                            c+=1
                            fig.clf()
                            plt.clf()
                        except ValueError:
                            fig.clf()
                            plt.clf()

def opt_heatmap_line(x0,*args):
    #if x0[1]+x0[2] < 1:
    #    return 50
    x,y,z = args
    xi = np.unique(x)
    #xi = np.arange(x.min(),x.max(),0.1)
    #yi = x0[0]*xi+x0[1]+x0[2]*xi**2
    yi = x0[2]*np.exp(x0[0]*xi)+x0[1]
    zi = scipy.interpolate.griddata(np.asarray(zip(x,y)),z,(xi,yi),method="linear").T
    #if yi[0] < 1 or yi[0] > 4:
    #    return sum((10**zi)**2)*(1+0.1*(yi[0]-2.5)**2)
    #return sum((10**zi)**2)
    return sum((10**zi)**2)

def merge(pickle_names):
    def add_pickle(x,i):
        try:
            x.insert(0,"pickle",i.split("/")[-1].split(".")[0])
        except ValueError:
            pass
        return x

    # remove merged and reduced pickles from input
    c = 0
    for i in range(len(pickle_names)):
        j = i-c
        if "merged" in pickle_names[j] or "reduced" in pickle_names[j]:
            pickle_names.pop(j)
            c+=1
    folder = "/".join(pickle_names[0].split("/")[:-1])
    # read pickles in batches of 100, then merge
    pn = [pickle_names[100*i:100*(i+1)] for i in range((len(pickle_names)-1)/100+1)]
    DF = []
    for n, m in enumerate(pn):
        print (n+1)*100, len(pickle_names)
        DF.append(pandas.concat((add_pickle(pandas.read_pickle(i),i) for i in m),copy=False))
    if len(DF) > 0:
        df = pandas.concat(DF,copy=False)
    else:
        df = DF[0]

    for i in pandas.unique(df.pickle.values):
        print "output/" + i + ".pickle"

    # remove constant columns
    drop = []
    for i in df:
        if i in ["target_element","train_size", "cut_off", "mae", "rmsd",
                "y_pred","y_test","kernel_parameters","labels","pickle"]:
            continue
        if i in ["label_index","atomic","baseline_index"] \
        or i in ["descriptor", "regressor",
                "dampening", "damp_const1", "damp_const2",
                "dampening2", "damp2_const1", "damp2_const2", "random", "random_sigma"] \
        or df[i].astype(str).nunique() == 1:
            drop.append(i)
            print "drop", i, df[i].values[0]
        elif df[i].astype(str).nunique() < 50 and i != "remove_labels":
            print "keep", i, pandas.unique(df[i].values)

    df.drop(drop, axis=1, inplace=True)
    df.to_pickle(folder + "/merged.pickle")

    df.drop_duplicates(["mae"], inplace=True)
    df.drop(["y_pred","y_test","labels"], axis=1, inplace=True)
    df.to_pickle(folder + "/reduced.pickle")

def distance(pickle_names):
    import seaborn as sns
    # seaborn style
    sns.set_style("whitegrid",{'grid.color':'.92','axes.edgecolor':'0.92'})

    # read pickles
    df = pandas.DataFrame()
    df = pandas.concat((pandas.read_pickle(i) for i in pickle_names))

    #selection = True
    #for i in df:
    #    if i not in ["pickle", 'cut_off', 'cut_off2', 'data_folder',
    #            'kernel_parameters', 'mae', 'observable_index',
    #            'rmsd', 'target_element', 'train_size', "remove_labels",
    #            "exponent","sncf", "cm_cut"]:
    #        quit("Set %s in heatmap plot" % i)
    #    else:
    #        if i == "cut_off2":
    #            selection = selection & (df.cut_off2 == 1000.)
    #        elif i == "data_folder":
    #            selection = selection & (df.data_folder == '/home/lab/dev/ML_SK/data/delta')
    #        elif i == "observable_index":
    #            selection = selection & (df.observable_index == 0)
    #        elif i == "remove_labels":
    #            selection = selection & (df.remove_labels.isnull())
    #        elif i == "exponent":
    #            selection = selection & (df.exponent == 1)
    #        elif i == "sncf":
    #            selection = selection & (df.sncf == False)
    #        elif i == "cm_cut":
    #            selection = selection & (df.cm_cut == 0.)


    ## NOTE: choose different data/settings here
    #df = df.loc[selection]

    train_size = {"C":22000, "H":22000, "N":6800}

    colors = {"C":"k", "H":"r", "N":"b"}

    for metric in ("mae",):#"rmsd"):
        fig, ax = plt.subplots(dpi=600,figsize=(16,12))
        #FigureCanvas(fig) # to be able to save as pdf
        for a,ele in enumerate(np.unique(df["target_element"].values)):
            # take the largest training size available
            x = []
            z1,z2,z3 = [], [], []
            for cut in np.unique(df.loc[(df.target_element == ele) &
                                (df.train_size == train_size[ele])]["cut_off"].values):
                if cut in []:#,7.5,8.0,8.5,9.0,9.5,10.0]:#,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0]:#,3.0,3.5,4.0,4.5,5.0,
                    continue
                this_df = df.loc[(df.target_element == ele)
                               & (df.cut_off == cut)
                               & (df.train_size == train_size[ele])]

                values = pandas.unique(eval("this_df." + metric + ".values")).astype(float)
                values = np.sort(values)
                if values.size not in [5,15]:
                    print this_df.head(values.size)
                    for i in this_df.pickle.values:
                        print i
                    quit()

                # NOTE: add this line to force only N points
                #values = np.random.choice(values,size=4,replace=False)

                med = np.median(values)
                print ele, cut, med
                #conf = np.exp(sms.DescrStatsW(np.log(values)).tconfint_mean())
                # bootstrap stdev
                boot = np.median(np.random.choice(values, size=(1000,values.size)),axis=1)
                boot = np.sort(boot)
                conf = [boot[160],boot[-160]]
                x.append(cut)
                z1.append(med)
                z2.append(conf[0])
                z3.append(conf[1])

            x,z1,z2,z3 = [np.asarray(i) for i in [x,z1,z2,z3]]
            if len(x) == 1:
                continue
            # interpolate
            #z2i = scipy.interpolate.interp1d(x,np.log(z2),kind="cubic")
            #z3i = scipy.interpolate.interp1d(x,np.log(z3),kind="cubic")
            z2i = scipy.interpolate.PchipInterpolator(x,np.log(z2))
            z3i = scipy.interpolate.PchipInterpolator(x,np.log(z3))
            #z2i = scipy.interpolate.Akima1DInterpolator(x,np.log(z2))
            #z3i = scipy.interpolate.Akima1DInterpolator(x,np.log(z3))
            x_new = np.arange(x.min(),x.max(),0.01)
            ax.plot(x,z1,"-o",color=colors[ele],label=ele, linewidth=0.4, markersize=4)
            ax.fill_between(x_new,np.exp(z2i(x_new)),np.exp(z3i(x_new)),color=colors[ele],alpha=0.2,antialiased=True)
            #ax.fill_between(x,z2,z3,color=colors[ele],alpha=0.2,antialiased=True)

        ax.set_yscale("log")
        ax.set_yticks([0.05,0.15,0.5,1.5])
        ax.set_yticklabels([str(i) for i in (ax.get_yticks().tolist())])
        ax.set_ylim([0.05,2.0])
        ax.set_xlabel(r"Cutoff distance in $\mathrm \AA$")
        ax.set_ylabel(r"%s in ppm" % metric.upper())

        # change text size
        ax.xaxis.label.set_fontsize(28)
        ax.yaxis.label.set_fontsize(28)
        for i in ax.get_xticklabels() + ax.get_yticklabels():
            i.set_fontsize(22)

        # legend next to figure
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=22)
        fig.tight_layout(rect=(0,0,0.9,1))
        fig.savefig("distance_%s.pdf" % metric)
        fig.clf()

def learning_rate(pickle_names):
    import seaborn as sns
    # seaborn style
    sns.set_style("whitegrid",{'grid.color':'.92','axes.edgecolor':'0.92'})

    # read pickles
    df = pandas.read_pickle(pickle_names[0])
    df1 = pandas.read_pickle(pickle_names[1])
    #df2 = pandas.read_pickle(pickle_names[2])

    #colors = {"C":"k", "H":"r", "N":"b"}
    colors = {0:"k", 1:"r", 2:"b"}
    names = {0:"OPBE", 1:"HF"}

    for metric in ("mae",):#"rmsd"):
        fig = plt.subplots(dpi=600,figsize=(16,5))
        #FigureCanvas(fig) # to be able to save as pdf
        axes = []
        gs = matplotlib.gridspec.GridSpec(1,3)
        for a,ele in enumerate(np.unique(df["target_element"].values)):
            if a == 0:
                ax = plt.subplot(gs[a])
                axes.append(plt.subplot(ax))
            else:
                ax = plt.subplot(gs[a])
                axes.append(plt.subplot(ax,sharey=axes[0]))
                axes[a].label_outer()

            for df_idx, dfi in enumerate((df,df1)):#,df2):
                x = []
                z1,z2,z3 = [],[],[]
                for tsize in np.unique(dfi.loc[dfi.target_element == ele]["train_size"].values):
                    medians = []
                    cut_offs = []
                    for cut in np.unique(dfi.loc[(dfi.target_element == ele) & (dfi.train_size == tsize)]["cut_off"].values):
                        print ele, tsize, cut
                        this_df = dfi.loc[(dfi.target_element == ele)
                                       & (dfi.train_size == tsize)
                                       & (dfi.cut_off == cut)].drop(["target_element",
                                                                    "train_size",
                                                                    "cut_off"],axis=1)
                        values = np.unique(eval("this_df." + metric + ".values"))
                        if values.size not in [5,15,14,6]:
                            print ele, tsize, cut, values.size
                            for i in this_df.pickle.values:
                                print "output/"+i+".pickle"
                            quit("value size")
                        medians.append(np.median(values))
                        cut_offs.append(cut)
                    idx = np.argmin(medians)
                    x.append(tsize)
                    print "atom:",ele,"\ttsize:",tsize,"\tcut_off:",cut_offs[idx]
                    this_df = dfi.loc[(dfi.target_element == ele)
                                   & (dfi.train_size == tsize)
                                   & (dfi.cut_off == cut_offs[idx])].drop(["target_element",
                                                                "train_size",
                                                                "cut_off"],axis=1)
                    values = np.unique(eval("this_df." + metric + ".values"))
                    #values = np.random.choice(values,size=4,replace=False)
                    med = np.median(values)
                    #conf = np.exp(sms.DescrStatsW(np.log(values)).tconfint_mean())
                    z1.append(med)
                    #z2.append(conf[0])
                    #z3.append(conf[1])

                x,z1,z2,z3 = [np.asarray(i) for i in [x,z1,z2,z3]]
                #z2i = scipy.interpolate.interp1d(x,z2,kind="linear", assume_sorted=True)
                #z3i = scipy.interpolate.interp1d(x,z3,kind="linear", assume_sorted=True)
                #z2i = scipy.interpolate.PchipInterpolator(x,z2)
                #z3i = scipy.interpolate.PchipInterpolator(x,z3)
                #x_new = np.arange(x.min(),x.max(),5)
                axes[a].plot(x,z1,"-o",color=colors[df_idx],label=names[df_idx], linewidth=0.4, markersize=4)
                #plt.fill_between(x_new,z2i(x_new),z3i(x_new),color=colors[ele],alpha=0.2,antialiased=True)
                axes[a].set_xscale("log")
                axes[a].set_yscale("log")

                # set ticks
                axes[a].set_yticks([0.05,0.15,0.5,1.5])
                axes[a].set_xticks([1e2,1e3,1e4,1e5])
                axes[a].set_yticklabels([str(i) for i in (axes[0].get_yticks().tolist())])
                axes[a].set_ylim([0.04,1.5])
        axes[0].set_ylabel(r"%s in ppm" % metric.upper())
        axes[1].set_xlabel(r"Training set size")

        # change text size
        for ax in axes:
            ax.xaxis.label.set_fontsize(28)
            ax.yaxis.label.set_fontsize(28)
            for i in ax.get_xticklabels() + ax.get_yticklabels():
                i.set_fontsize(22)

        # legend on top
        handle_labels = []
        for ax in axes:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width, box.height*0.8])
            handle_labels.append(ax.get_legend_handles_labels())
        #axes[1].legend(handle_labels[0][0]+handle_labels[1][0]+handle_labels[2][0],
        #               handle_labels[0][1]+handle_labels[1][1]+handle_labels[2][1],
        #               loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2, fancybox=True)
        axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2, fancybox=True, fontsize=22)
        plt.tight_layout(rect=(0,0,1,0.9))
        plt.savefig("learning_rate_%s.pdf" % metric)
        plt.clf()

def error(pickle_names):
    import seaborn as sns
    # seaborn style
    sns.set_style("whitegrid",{'grid.color':'.92','axes.edgecolor':'0.92'})

    # read pickles
    df = pandas.read_pickle(pickle_names[0])
    #for i in df:
    #    if i not in ["pickle", 'cut_off', 'cut_off2', 'data_folder',
    #            'kernel_parameters', 'mae', 'observable_index',
    #            'rmsd', 'target_element', 'train_size', "remove_labels","labels",
    #            "y_pred","y_test"]:
    #        quit("Set %s in error plot" % i)


    ## NOTE: choose different data/settings here
    #df = df.loc[(df.cut_off2 == 1000.)
    #           & (df.data_folder =='/home/lab/dev/ML_SK/data/delta')
    #           & (df.observable_index == 0)
    #           & (df.remove_labels.isnull())]

    colors = {0:"k", 1:"r", 2:"b"}

    for metric in ("mae",):#"rmsd"):
        for a,ele in enumerate(np.unique(df["target_element"].values)):
            if ele != "C" and ele != "H":
                continue
            fig, axes = plt.subplots(1,2, dpi=600, figsize=(14,6))
            axes[1].set_xlim([-6,6])
            for tsize in [200,2000,22000]:#np.unique(df.loc[df.target_element == ele]["train_size"].values):

                medians, cut_offs = [], []
                for cut in np.unique(df.loc[(df.target_element == ele) &
                                            (df.train_size == tsize)]["cut_off"].values):
                    this_df = df.loc[(df.target_element == ele)
                                   & (df.train_size == tsize)
                                   & (df.cut_off == cut)].drop(["target_element",
                                                                "train_size",
                                                                "cut_off"],axis=1)
                    # TODO median or average or something
                    medians.append(np.median(np.unique(eval("this_df." + metric + ".values"))))
                    cut_offs.append(cut)
                idx = np.argmin(medians)
                this_df = df.loc[(df.target_element == ele)
                               & (df.train_size == tsize)
                               & (df.cut_off == cut_offs[idx])].drop(["target_element",
                                                            "train_size",
                                                            "cut_off"],axis=1)
                this_df.drop_duplicates(["y_test","labels"], inplace=True)

                x = this_df.y_pred.values
                y = this_df.y_test.values

                plots.plot_points(x,y,axes[0], plot_line=True, label=str(tsize))
                plots.plot_kde(y-x, axes[1],color=colors[a],clip=axes[1].get_xlim())
                print tsize, np.mean(abs(y-x))
                # cut biggest outliers in KDE (99% confidence)

            handle_labels = []
            box2 = axes[1].get_position()
            axes[1].set_position([box2.x0*0.9, box2.y0, box2.width*1.1, box2.height*0.8])
            box1 = axes[0].get_position()
            axes[0].set_position([box1.x0, box1.y0, box1.width*0.9, box1.height*0.8])
            handle_labels.append(axes[0].get_legend_handles_labels())
            handle_labels.append(axes[1].get_legend_handles_labels())
            axes[0].set_ylabel(r"Prediction in ppm")
            axes[0].set_xlabel(r"Reference in ppm")
            axes[1].set_xlabel(r"Prediction error in ppm")
            axes[1].legend(handle_labels[0][0]+handle_labels[1][0],
                       handle_labels[0][1]+handle_labels[1][1],
                       loc="upper left", bbox_to_anchor=(0.5, 0.0), ncol=3, fancybox=True)



            fig.tight_layout(rect=(0,0,1,1))
            fig.savefig("%s_error_%s.pdf" % (ele, metric))


if __name__ == "__main__":
    # TODO linear model over cut offs
    if len(sys.argv) > 2:
        if sys.argv[1] == "heatmap":
            heatmap(sys.argv[2:])
        elif sys.argv[1] == "distance":
            distance(sys.argv[2:])
        elif sys.argv[1] == "learning":
            learning_rate(sys.argv[2:])
        elif sys.argv[1] == "error":
            error(sys.argv[2:])
        elif sys.argv[1] == "label_error":
            error(sys.argv[2:])
        elif sys.argv[1] == "merge":
            merge(sys.argv[2:])
    if sys.argv[1] == "linear":
        from sklearn.linear_model import Lasso, LassoLars

        df = pandas.read_pickle("merged.pickle")
        df = df.loc[(df.cut_off2 == 1000.)
                   & (df.data_folder =='/home/lab/dev/ML_SK/data/delta')
                   & (df.observable_index == 0)
                   & (df.remove_labels.isnull())
                   & (df.target_element == "C")]
        df = df[df.train_size == 6800]
        df.sort_values("y_test",inplace=True)
        x = []
        cut = []
        for c in pandas.unique(df.cut_off.values):
            if c not in [3.5,3.0,2.5]:
                continue
            x.append(df[df.cut_off == c].y_pred.values)
            cut.append(c)
        x = np.asarray(x).T
        y = df[df.cut_off == c].y_test.values

        m = Lasso(alpha=1e-5,fit_intercept=False,positive=True, max_iter=5000, selection="random")
        from sklearn.cross_validation import train_test_split
        d = []
        #for i in range(1000):
        #    xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=1)
        #    m.fit(xtrain,ytrain)
        #    print m.coef_
        #    quit()
        #    pred = m.predict(xtest)
        #    y0 = (ytest-pred)
        #    d.append(y0)
        #d = np.asarray(d)
        #print np.mean(abs(d)), np.mean(d**2)**0.5
        print np.mean(abs(y.reshape(-1,1)-x))
        quit()
        #m.coef_[0] = 0
        #coef = m.coef_.copy()
        #print cut
        #print min(np.mean(abs(ytest.reshape(-1,1)-xtest),axis=0)), min(np.mean((ytest.reshape(-1,1)-xtest)**2,axis=0)**0.5)
        #for i in range(2):
        #    print i,
        #    m.coef_ = coef.copy()
        #    m.coef_[i] = 0
        #    m.coef_ /= sum(m.coef_)
        #    print np.mean(abs(ytest-pred)), np.mean((ytest-pred)**2)**0.5

    if sys.argv[1] == "stats":
        if len(sys.argv) == 2:
            df = pandas.read_pickle("reduced.pickle")
        else:
            df = pandas.read_pickle(sys.argv[2])
        df = df.loc[(df.train_size == 6800) & (df.target_element=="N")]
        for i in df:
            if i in ["kernel_parameters","mae","rmsd","pickle"]:
                continue
            print i, pandas.unique(df[i].values)

        for cut in pandas.unique(df.cut_off.values):
            for cut2 in pandas.unique(df.cut_off2.values):
                this_df = df.loc[(df.cut_off == cut) & (df.cut_off2 == cut2)]
                values = this_df.mae.values
                if values.size in [5,15]:
                    print cut, cut2, np.median(values)


