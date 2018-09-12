import numpy as np
import itertools
import matplotlib
from matplotlib import pyplot as plt
import os

def plot_histogram(pos_features,neg_features,pos_labels='Knowns',neg_labels='Unknowns',title="Histogram",file_name='{}foo.pdf'):
    """
    This function plots the Histogram for Magnitudes of feature vectors.
    """
    pos_mag=np.sqrt(np.sum(np.square(pos_features),axis=1))
    neg_mag=np.sqrt(np.sum(np.square(neg_features),axis=1))
    
    pos_hist=np.histogram(pos_mag, bins=500)
    neg_hist=np.histogram(neg_mag, bins=500)

    fig, ax = plt.subplots(figsize=(4.5, 1.75))
    ax.plot(pos_hist[1][1:],pos_hist[0].astype(np.float16)/max(pos_hist[0]),label=pos_labels,color='g')
    ax.plot(neg_hist[1][1:],neg_hist[0].astype(np.float16)/max(neg_hist[0]),color='r',label=neg_labels)
    
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.xscale('log')
    plt.tight_layout()
    if title is not None:
        plt.title(title)
    plt.savefig(file_name.format('Hist','pdf'),bbox_inches='tight')
    plt.show()
        

def plot_softmax_histogram(pos_prob,neg_prob,pos_labels='Knowns',neg_labels='Unknowns',title=None,file_name='{}foo.pdf'):    
    """
    This function plots the Histogram for softmax scores.
    """
    if pos_prob.shape[1]==11:
        pos=np.max(pos_prob[:,:-1],axis=1)
        neg=np.max(neg_prob[:,:-1],axis=1)
    else:
        pos=np.max(pos_prob,axis=1)
        neg=np.max(neg_prob,axis=1)
    pos_hist=np.histogram(pos, bins=50)#'auto')
    neg_hist=np.histogram(neg, bins=50)#'auto')
    
    fig, ax = plt.subplots(figsize=(4.5, 1.75))
    ax.plot(neg_hist[1][1:],neg_hist[0].astype(np.float16),'+',label=neg_labels,color='r',alpha=0.7)
    ax.plot(pos_hist[1][1:],pos_hist[0].astype(np.float16),'x',label=pos_labels,color='g',alpha=0.7)
    
    plt.yscale('log')
    ax.set_xticks(np.arange(0, 1.1, .1))
    ax.get_yaxis().set_ticks([1,10,100,1000,10000])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    if title is not None:
        plt.title(title)
    plt.savefig(file_name.format('SoftMax_Hist','pdf'), bbox_inches='tight')
    plt.show()



def _plot_softmax_histogram(pos_prob,neg_prob,pos_labels='Knowns',neg_labels='Unknowns',title=None,file_name='{}foo.pdf'):    
    if pos_prob.shape[1]==11:
        pos=np.max(pos_prob[:,:-1],axis=1)
        neg=np.max(neg_prob[:,:-1],axis=1)
    else:
        pos=np.max(pos_prob,axis=1)
        neg=np.max(neg_prob,axis=1)
    pos_hist=np.histogram(pos, bins=50)
    neg_hist=np.histogram(neg, bins=50)
    
    pos_=np.cumsum(np.array(pos_hist[0]).astype(np.float16))
    pos_=pos_/np.max(pos_)
    neg_=np.cumsum(np.array(neg_hist[0]).astype(np.float16))
    neg_=np.max(neg_)-neg_
    neg_=neg_/np.max(neg_)
    
    fig, ax = plt.subplots(figsize=(4.5, 1.75))
    ax.plot(neg_hist[1][1:],neg_.astype(np.float16),label=neg_labels,color='r',alpha=0.7)
    ax.plot(pos_hist[1][1:],pos_.astype(np.float16),label=pos_labels,color='g',alpha=0.7)

    plt.yscale('log')
    ax.set_xticks(np.arange(0, 1.1, .1))
    ax.set_ylim([0.0001,1.3])    
    plt.grid()
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    if title is not None:
        plt.title(title)
    plt.savefig(file_name.format('SoftMax_Hist','pdf'), bbox_inches='tight')
    plt.show()



def plotter_2D(
                pos_features,
                labels,
                neg_features=None,
                pos_labels='Knowns',
                neg_labels='Unknowns',
                title=None,
                file_name='foo.pdf',
                final=False,
                pred_weights=None,
                heat_map=False
            ):
    import data_prep
    plt.figure(figsize=[6,6])
    
    if heat_map:
        min_x,max_x=np.min(pos_features[:,0]),np.max(pos_features[:,0])
        min_y,max_y=np.min(pos_features[:,1]),np.max(pos_features[:,1])
        x=np.linspace(min_x*1.5, max_x*1.5,200)
        y=np.linspace(min_y*1.5, max_y*1.5,200)
        pnts=list(itertools.chain(itertools.product(x,y)))
        pnts = np.array(pnts)

        e_=np.exp(np.dot(pnts,pred_weights))
        e_=e_/np.sum(e_,axis=1)[:,None]
        res=np.max(e_,axis=1)

        plt.pcolor(x,y,np.array(res).reshape(200,200).transpose(),rasterized=True)

    plt.scatter(pos_features[:,0], pos_features[:,1], c=data_prep.colors[labels.astype(np.int)],edgecolors='none',s=0.5)
    if neg_features is not None:
        plt.scatter(neg_features[:,0], neg_features[:,1],c='k',edgecolors='none',s=15,marker="*")
    if final:
        plt.gca().spines['right'].set_position('zero')
        plt.gca().spines['bottom'].set_position('zero')
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.tick_params(axis='both',bottom=False, left=False,labelbottom=False,labeltop=False,labelleft=False , labelright=False)
        plt.axis('equal')

    plt.savefig(file_name.format('2D_plot','png'),bbox_inches='tight')
    plt.show()
    if neg_features is not None:
        plot_histogram(pos_features,neg_features, pos_labels=pos_labels,neg_labels=neg_labels,title=title,file_name=file_name)
