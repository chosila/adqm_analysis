import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
#np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
#np.set_printoptions(threshold=sys.maxsize)

parser = argparse.ArgumentParser()
parser.add_argument('algo', type=str, help="'pull' or 'chi2' are the options if you used analyze.py to create your output files")
parser.add_argument('subsystem', type=str, help='subsystem name matching the one that you gave analyze.py')
parser.add_argument('N', type=int, help='number of hists that fail threshold. Usually 1,3, or 5')
args = parser.parse_args()

#pull_fig, pull_ax = plt.subplots()
#chi2_fig, chi2_ax = plt.subplots()
#combined_fig, combined_ax = plt.subplots()
#percNpull_fig, percNpull_ax = plt.subplots()
#percNchi2_fig, percNchi2_ax = plt.subplots()
#percNcomb_fig, percNcomb_ax = plt.subplots()
N = args.N
algo = args.algo
subsystem = args.subsystem
## plot the pull and chi2 using rob's plotting scripts
fig, axs = plt.subplots(ncols=2,nrows=1,figsize=(12,6))
fig0, ax0 = plt.subplots(figsize=(6,6))
fig1, ax1 = plt.subplots(figsize=(6,6))


for numref, marker, color in zip(['1_REF', '4_REF', '8_REF'], ['-rD', '-bo', '-g^'], ['purple', 'yellow', 'orange']):
    plotkwargs = {'label':numref,  'marker':'.'}
    # assumed all csv files are same name but in different directories
    df = pd.read_csv(f'csv/{subsystem}_{numref}.csv')

    ## split into good (0,-1) and bad runs (1)
    df_g = df[df['label'] == 0]
    df_b = df[df['label'] == 1]


    ## remove bad rows (-99), col = ['label', 'run_number']
    df_g = df_g.filter(regex=f'{algo}_score') #'score')
    df_g = df_g[df_g != -99].dropna(how='all')
    df_b = df_b.filter(regex=f'{algo}_score')  # 'score')
    df_b = df_b[df_b != -99].dropna(how='all')

    ## sort descending
    sorted_df_g = -np.sort(-df_g, axis=0)

    ## calculate thresholds
    cuts = np.array([(col[1:] + col[:-1])/2 for col in sorted_df_g.T]).T
    print(df_g.shape)
    zerothcut = sorted_df_g[0,:] + (sorted_df_g[0, :] - sorted_df_g[1,:])/2
    cuts = np.insert(cuts, 0, zerothcut, axis=0)

    # print(sorted_df_g)
    #print(zerothcut)

    ## get counts and mean
    counts_g = np.array([np.count_nonzero(df_g >= cut, axis=1) for cut in cuts])
    counts_b = np.array([np.count_nonzero(df_b >= cut, axis=1) for cut in cuts])
    avg_cnt_g = counts_g.mean(axis=1)
    avg_cnt_b = counts_b.mean(axis=1)


    # ## plot -- commented out as we are using rob's plotting style now
    plotdir = 'plots/'
    # pull_ax.scatter(avg_cnt_g, avg_cnt_b, **plotkwargs)
    # pull_ax.plot(range(0,10), range(0,10), color='r', linewidth=1, linestyle='--')
    # pull_ax.set_xlim(0,5)
    # pull_ax.set_ylim(0,15)
    # pull_ax.set_title('average runs flagged: {}'.format(numref))
    # pull_ax.legend()
    # pull_ax.set_xlabel('average good runs flagged')
    # pull_ax.set_ylabel('average bad runs flagged')
    # pull_fig.savefig(plotdir+'avg_{}.png'.format(numref), bbox_inches='tight')
    # print('save ' + plotdir+'avg_{}.png'.format(numref))

    # --------------- percent runs given N fails -----------------
    perc_g = np.count_nonzero(counts_g > N, axis=1)/counts_g.shape[1]
    perc_b = np.count_nonzero(counts_b > N, axis=1)/counts_b.shape[1]

    # percNpull_ax.scatter(perc_g, perc_b , **plotkwargs)
    # percNpull_ax.legend()
    # percNpull_ax.set_title('fraction runs with N >= {} failing runs {}'.format(N, numref))
    # percNpull_ax.set_xlabel('fraction good runs')
    # percNpull_ax.set_ylabel('fraction bad runs')
    # percNpull_ax.set_xlim(0,.5)
    # percNpull_ax.set_ylim(0,1)
    # percNpull_ax.plot(range(0,10), range(0,10), color='r', linewidth=1, linestyle='--')
    # percNpull_fig.savefig(plotdir+'percN_{}.png'.format(numref), bbox_inches='tight')
    # print('save ' + plotdir+'percN_{}.png'.format(numref))
    # print('--------------------------')



    ##--------- plotting the output in the same way as rob --------------
    ax1.set_xlabel('Fraction of good runs with at least N=3 histogram flags')
    ax1.set_ylabel('Fraction of bad runs with at least N=3 histogram flags')
    algorithm_name = "combined"

    # commented out but keep for the aggregated scores plots
    #for jj in range(len(N_bad_hists)):
    #  print(N_bad_hists[jj])
    #  print(tFRF_ROC_good_X[jj])
    #  print(tFRF_ROC_bad_Y[jj])
    ax1.plot(perc_g, perc_b, marker, mfc=color, mec='k', markersize=8, linewidth=1, label=numref)
    ax1.axis(xmin=0,xmax=0.4,ymin=0,ymax=0.8)
    ax1.axline((0, 0), slope=1, linestyle='--',linewidth=0.8,color='gray')
    ax1.annotate(algorithm_name + " RF ROC", xy=(0.05, 0.95), xycoords='axes fraction', xytext=(10, -10), textcoords='offset points', ha='left', va='top', fontsize=12, weight='bold')
    ax1.legend(loc='lower right')


    ax0.set_xlabel('Mean number of histogram flags per good run')
    ax0.set_ylabel('Mean number of histogram flags per bad run')
    ax0.plot(avg_cnt_g, avg_cnt_b, marker, mfc=color, mec='k', markersize=8, linewidth=1, label=numref)
    ax0.axline((0, 0), slope=1, linestyle='--',linewidth=0.8,color='gray')
    ax0.annotate(algorithm_name + " HF ROC", xy=(0.05, 0.95), xycoords='axes fraction', xytext=(10, -10), textcoords='offset points', ha='left', va='top', fontsize=12, weight='bold')
    ax0.axis(xmin=0,xmax=8,ymin=0,ymax=25)
    ax0.legend(loc='lower right')


    ## --------------------------------------------------------------------

    # # at mean histogram good = 1.5, what is the threshold? So we need to look at the mean valeus
    #def find_nearest(array, value):
    #    array = np.asarray(array)
    #    idx = (np.abs(array - value)).argmin()
    #    return idx

    ## score distribution for MH ==~ 1.5
    #idx = find_nearest(avg_cnt_g, 1.5)
    #
    #cutsatMH = cuts[idx, :]
    #fig, ax = plt.subplots()
    #hist, bins, _ = ax.hist(cutsatMH, bins = 20)
    #ax.cla()
    #logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    #ax.hist(toplot, bins=logbins)
    #ax.set_xscale('log')
    #ax.set_title(f'distribution of {algo} threshold values at MH ~ 1.5')
    #plt.savefig(f'plots/analysis/distributionmhh15_{algo}.png')


fig0.savefig(plotdir + "HF_ROC_comparison_" + algorithm_name + ".pdf",bbox_inches='tight')
print("SAVED: " + plotdir + "RF_HF_ROC_comparison_" + algorithm_name + ".pdf")
fig1.savefig(plotdir + "RF_ROC_comparison_" + algorithm_name + ".pdf",bbox_inches='tight')
print("SAVED: " + plotdir + "RF_ROC_comparison_" + algorithm_name + ".pdf")
