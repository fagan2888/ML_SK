import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import matplotlib.cm as cm

def plot_points(X, Y, ax = plt, plot_line=True, label = "", inp = None):
    '''
    Plots sets of points
    '''
    if inp == "list":
        for i in range(len(X)):
            ax.plot(X[i],Y[i],".", ms=4, label=label)
    else:
        ax.plot(X,Y,".", ms=4, label=label)

    if plot_line:
        if inp == "list":
            line = ([np.min(np.concatenate(X))]*2,[np.max(np.concatenate(X))]*2)
        else:
            line = ([np.min(X)]*2,[np.max(X)]*2)

        #print line
        #ax.plot(*line,linestyle="--",c="k",lw=0.5)
        ax.plot(*line,linestyle="-",c="k",lw=10.5)


def plot_kde(x,ax,clip=None,color=None):
    import seaborn as sns
    '''Plots an 1D Kernel Density Estimate'''
    ax.grid(False)
    kde = sns.kdeplot(x,ax=ax, gridsize=1000,clip=clip,color=color) #clip, cut, 
    sns.kdeplot(x,ax=ax,shade=True, alpha=0.1, color=color, gridsize=1000,clip=clip) #clip, cut, 
    ax.set_ylabel("Density")
    sns.despine(ax=ax)
    plt.gca().axes.get_yaxis().set_ticks([])


def common_plot(x,y,str_x,diag=None,prediction=False):
  plt.title(str_x)

  plt.plot(x,y,'bo')#,diagonal,diagonal,'r')

  if diag:
    diagonal=np.linspace(x.min(),x.max(), 10)
    plt.plot(diagonal,diagonal,'r')

  plt.xlim([x.min(),x.max()])
  if prediction==False:
    plt.ylim([y.min(),y.max()])
  else:
    plt.ylim([x.min(),x.max()])
  plt.savefig(str_x)
  plt.clf()


def plot_labels(x,y,z,str_x, show=False, save=True):
    if show+save == 0:
        return
    plt.title(str_x)

    n = len(set(z))

    color=iter(cm.rainbow(np.linspace(0,1,n)))
    for l in set(z):
        c = next(color)
        plt.plot(x[z == l], y[z == l],c=c, marker=".", linestyle="")
        a, b = ss.linregress(x[z == l],y[z == l])[:2]
        X = np.linspace(x[z == l].min(), x[z == l].max(), 2)
        Y = a*X+b
        plt.plot(X,Y,c=c,linestyle="-")

    diagonal=np.linspace(x.min(),x.max(), 2)
    # normalize = True?
    plt.plot(diagonal,diagonal,'k--')

    xlim = [(x.min() - x.mean())*1.1+x.mean(),(x.max() - x.mean())*1.1+x.mean()]
    ylim = [(y.min() - y.mean())*1.1+y.mean(),(y.max() - y.mean())*1.1+y.mean()]
    plt.ylim(ylim)
    plt.xlim(xlim)
    if show:
        plt.show()
    if save:
        plt.savefig("plots/" + str_x, dpi=300)
        plt.clf()


def plot_scatter(x,y,str_x="", show=False):
    import seaborn as sns
    plt.title(str_x)

    plt.scatter(x, y)
    a, b = ss.linregress(x, y)[:2]
    X = np.linspace(x.min(), x.max(), 2)
    Y = a*X+b
    plt.plot(X,Y,linestyle="-")

    diagonal=np.linspace(x.min(),x.max(), 2)
    # normalize = True?
    plt.plot(diagonal,diagonal,'k--')

    xlim = [(x.min() - x.mean())*1.1+x.mean(),(x.max() - x.mean())*1.1+x.mean()]
    ylim = [(y.min() - y.mean())*1.1+y.mean(),(y.max() - y.mean())*1.1+y.mean()]
    plt.ylim(ylim)
    plt.xlim(xlim)
    if show:
        plt.show()
    else:
        plt.savefig("plots/" + str_x, dpi=300)
        plt.clf()
