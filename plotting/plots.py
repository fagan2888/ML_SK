
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import matplotlib.cm as cm

def plot_matrix(data,label=None):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  cax = ax.matshow(data)
  fig.colorbar(cax)

  #ax.set_xticklabels(label)
  #ax.set_yticklabels(label)
  plt.show()


def plot_histo(x,str_x,y,str_y,train_size):
  '''Plots a two-panel histogram'''
  bins = np.linspace(x.min(), x.max(), 140)
  plt.figure(1)
  plt.subplot(211); plt.title(str_x); plt.hist(x,bins); plt.xlim(([x.min(),x.max()]))
  plt.subplot(212); plt.title(str_y); plt.hist(y,bins); plt.xlim(([x.min(),x.max()]))
  plt.show()

def plot_kde(x,title, show = False, save = True):
    '''Plots an 1D Kernel Density Estimate'''
    if show+save == 0:
        return
    grid = np.linspace((x.min()-x.mean())*1.1, (x.max()-x.mean())*1.1, 1000) + x.mean()
    kde = ss.gaussian_kde(x)
    y = kde(grid)

    plt.plot(grid, y, c="b")
    plt.fill_between(grid, y, color="b", alpha=0.1)
    plt.title(title)
    if show:
        plt.show()
    if save:
        plt.savefig("plots/" + title, dpi=300)
        plt.clf()


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


def plot_scatter(x,y,str_x, show=False, save=True):
    if show+save == 0:
        return
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
    if save:
        plt.savefig("plots/" + str_x, dpi=300)
        plt.clf()
