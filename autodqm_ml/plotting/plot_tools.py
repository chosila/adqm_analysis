import matplotlib.pyplot as plt 
import numpy as np 
from pathlib import Path
import awkward


def plot1D(original_hist, reconstructed_hist, run, hist_path, algo, threshold):    
    """
    plots given original and recontructed histogram. Will plot the MSE plot if the SSE is over the threshold. 

    :param original_hist: original histogram to be plotted  
    :type original_hist: numpy array of shape (n, )
    :param reconstructed_hist: reconstructed histogram from the ML algorithm
    :type reconstructed_hist: numpy array of shape (n, )
    :param hist_path: name of histogram
    :type hist_path: str
    :param algo: name of algorithm used. This is used to label the folder. Can use self.name to be consistent between this and plotMSESummary
    :param type: str
    :param threshold: threshold to determind histogram anomaly
    :type threshold: int
    """
    fig, ax = plt.subplots()
    mse = np.mean(np.square(original_hist - reconstructed_hist))
    
    # for bin edges
    binEdges = np.linspace(0, 1, original_hist.shape[0])
    width = binEdges[1] - binEdges[0]
    # plot original/recon 
    ax.bar(binEdges, original_hist, alpha=0.5, label='original', width=width)
    ax.bar(binEdges, reconstructed_hist, alpha=0.5, label='reconstructed', width=width)
    plotname = hist_path.split('/')[-1]
    ax.set_title(f'{plotname} {run} {algo}')
    leg = ax.legend(loc='upper right')
    text = '\n'.join((
        f'mse: {mse:.4e}',
        ))
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(.8, 0.1*max(original_hist.max(), reconstructed_hist.max()), text, wrap=True, bbox=props )
    # create directory to save plot
    Path(f'plots/{algo}/{run}').mkdir(parents=True, exist_ok=True)
    fig.savefig(f'plots/{algo}/{run}/{plotname}.png')
    plt.close('all')
    
    if mse > threshold: 
        fig2, ax2 = plt.subplots()
        ax2.bar(binEdges, np.square(original_hist - reconstructed_hist), alpha=0.5, width=width)
        ax2.set_title(f'MSE {plotname} {run}')
        fig2.savefig(f'plots/{algo}/{run}/{plotname}-MSE.png')
        plt.close('all')
    

def plotMSESummary(original_hists, reconstructed_hists, threshold, hist_paths, runs, algo): 
    """ 
    Plots all the MSE on one plot that also shows how many passese threhold
    

    :param original_hist: original histogram to be plotted  
    :type original_hist: numpy array of shape (n, )
    :param reconstructed_hist: reconstructed histogram from the ML algorithm
    :type reconstructed_hist: numpy array of shape (n, )
    :param threshold: threshold to determind histogram anomaly
    :type threshold: int
    :param hist_path: list of name of histograms
    :type hist_path: list
    :param runs: list of runs used for testing. Must be same as list passed into the pca or autoencoder.plot function 
    :param type: list
    :param algo: name of algorithm used. This is used to place the plot in the correct folder. Can use self.name to be consistent between this and plot1D
    :param type: str

    """ 
    ## convert to awkward array as not all hists have the same length
    original_hists = awkward.Array(original_hists)
    reconstructed_hists = awkward.Array(reconstructed_hists)
    
    fig, ax = plt.subplots()
    mse = np.mean(np.square(original_hists - reconstructed_hists), axis=1)
    
    ## count number of good and bad histogrms
    num_good_hists = np.count_nonzero(mse < threshold)
    num_bad_hists = np.count_nonzero(mse > threshold)
    
    ## get names of top 5 highest mse histograms
    ## plot_names[argsort[-1]] should be the highest mse histogram
    sortIdx = np.argsort(mse)

    hist,_, _ = ax.hist(mse)
    ax.set_xlabel('MSE values') 
    ax.set_title('Summary of all MSE values')
    
    numHistText = [f'num good hists: {num_good_hists}', f'num bad hists: {num_bad_hists}']
    ## mse summary is for all plots in all runs, so need to make names accordingly
    hist_names = [hist_path.split('/')[-1] for hist_path in hist_paths]
    hist_names_runs = [f'{hist_name} ({run})' for run in runs for hist_name in hist_names]


    ## account for case less than 5 plots were tested
    maxidx = min(5, len(hist_names_runs))+1
    ## plot_name is the whole directory, so we only need the last for histname
    rankHistText = [f'{hist_names_runs[sortIdx[i]].split("/")[-1]}: {mse[sortIdx[i]]:.4e}' for i in range(-1, -maxidx, -1)]
    text = '\n'.join(
        numHistText + rankHistText
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(1.1*max(mse), 0.5*max(hist), text, wrap=True, bbox=props)
    
    fig.savefig(f'plots/{algo}/MSE_Summary.png', bbox_inches='tight')
