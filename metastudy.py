import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
## !!!TODO::change this to take arguments of csv files instead of hard code xlsx file
pull_fig, pull_ax = plt.subplots()
chi2_fig, chi2_ax = plt.subplots()
combined_fig, combined_ax = plt.subplots()
percNpull_fig, percNpull_ax = plt.subplots()
percNchi2_fig, percNchi2_ax = plt.subplots()
percNcomb_fig, percNcomb_ax = plt.subplots()
N = 3

for numref in ['1_REF', '4_REF', '8_REF']:
    plotkwargs = {'label':numref,  'marker':'.'}
    df = pd.read_excel('L1T_HLTPhysics.xlsx', sheet_name = '{}_L1THLTPhysics'.format(numref) )
    ## split into good (0,-1) and bad runs (1)
    df_g = df[df['label'] != 1]
    df_b = df[df['label'] == 1]

    ## remove bad rows (-99), col = ['label', 'run_number']
    df_g = df_g.filter(regex='score')
    df_g = df_g[df_g != -99].dropna(how='all')
    df_b = df_b.filter(regex='score')
    df_b = df_b[df_b != -99].dropna(how='all')

    ## split into pull and chi2 and convert to numpy then sort descending
    pull_g = -np.sort(-df_g.filter(regex='pull'), axis=0)
    chi2_g = -np.sort(-df_g.filter(regex='chi2'), axis=0)
    pull_b = -np.sort(-df_b.filter(regex='pull'), axis=0)
    chi2_b = -np.sort(-df_b.filter(regex='chi2'), axis=0)

    # -------------- number of counts failing thresholds ---------
    ## calculate thresholds
    pull_cuts = np.array([(col[1:] + col[:-1])/2 for col in pull_g.T]).T # first .T is so we can loop over columns instead of rows, the second .T converrts it back
    zerothCut = pull_g[0,:] + (pull_g[0,:] - pull_g[1,:])/2
    pull_cuts = np.insert(pull_cuts, 0, zerothCut, axis=0)

    chi2_cuts = np.array([(col[1:] + col[:-1])/2 for col in chi2_g.T]).T
    zerothCut = chi2_g[0,:] + (chi2_g[0,:] - chi2_g[1,:])/2
    chi2_cuts = np.insert(chi2_cuts, 0, zerothCut, axis=0)

    pull_cuts = pull_cuts
    chi2_cuts = chi2_cuts

    ## get counts and mean
    pull_counts_g = np.array([np.count_nonzero(df_g.filter(regex='pull') >= cut, axis=1) for cut in pull_cuts])
    pull_counts_b = np.array([np.count_nonzero(df_b.filter(regex='pull')>= cut, axis=1) for cut in pull_cuts])
    chi2_counts_g = np.array([np.count_nonzero(df_g.filter(regex='chi2') >= cut, axis=1) for cut in chi2_cuts])
    chi2_counts_b = np.array([np.count_nonzero(df_b.filter(regex='chi2')>= cut, axis=1) for cut in chi2_cuts])

    all_counts_g = np.append(pull_counts_g, chi2_counts_g, axis=1)
    all_counts_b = np.append(pull_counts_b, chi2_counts_b, axis=1)

    avg_pullcnt_g = pull_counts_g.mean(axis=1)
    avg_pullcnt_b = pull_counts_b.mean(axis=1)
    avg_chi2cnt_g = chi2_counts_g.mean(axis=1)
    avg_chi2cnt_b = chi2_counts_b.mean(axis=1)
    avg_allcnt_g  = all_counts_g.mean(axis=1)
    avg_allcnt_b  = all_counts_b.mean(axis=1)


    ## plot
    pull_ax.scatter(avg_pullcnt_g, avg_pullcnt_b, **plotkwargs)
    pull_ax.set_xlim(0,4)
    pull_ax.set_ylim(0,8)
    pull_ax.set_title('average runs flagged: pull')
    pull_ax.legend()
    pull_ax.set_xlabel('average good runs flagged')
    pull_ax.set_ylabel('average bad runs flagged')

    chi2_ax.scatter(avg_chi2cnt_g, avg_chi2cnt_b , **plotkwargs)
    chi2_ax.set_xlim(0,4)
    chi2_ax.set_ylim(0,8)
    chi2_ax.legend()
    chi2_ax.set_title('average runs flagged: chi2')
    chi2_ax.set_xlabel('average good runs flagged')
    chi2_ax.set_ylabel('average bad runs flagged')

    combined_ax.scatter(avg_allcnt_g, avg_allcnt_b, **plotkwargs)
    combined_ax.legend()
    combined_ax.set_xlim(0,4)
    combined_ax.set_ylim(0,8)
    combined_ax.set_title('average runs flagged: combined')
    combined_ax.set_xlabel('average good runs flagged')
    combined_ax.set_ylabel('average bad runs flagged')


    # --------------- percent runs given N fails -----------------
    perc_pull_g=np.count_nonzero(pull_counts_g >= N, axis=1)/pull_counts_g.shape[1]
    perc_pull_b=np.count_nonzero(pull_counts_b >= N, axis=1)/pull_counts_b.shape[1]
    perc_chi2_g=np.count_nonzero(chi2_counts_g >= N, axis=1)/chi2_counts_g.shape[1]
    perc_chi2_b=np.count_nonzero(chi2_counts_b >= N, axis=1)/chi2_counts_b.shape[1]
    perc_all_g =np.count_nonzero(all_counts_g >= N, axis=1)/all_counts_g.shape[1]
    perc_all_b =np.count_nonzero(all_counts_b >= N, axis=1)/all_counts_b.shape[1]

    percNpull_ax.scatter(perc_pull_g, perc_pull_b , **plotkwargs)
    percNpull_ax.legend()
    percNpull_ax.set_title('fraction runs with N >= {} failing runs pull cut'.format(N))
    percNpull_ax.set_xlabel('fraction good runs')
    percNpull_ax.set_ylabel('fraction bad runs')
    percNpull_ax.set_xlim(0,.4)
    percNpull_ax.set_ylim(0,.8)

    percNchi2_ax.scatter(perc_chi2_g, perc_chi2_b , **plotkwargs)
    percNchi2_ax.legend()
    percNchi2_ax.set_title('fraction runs with N >= {} failing runs chi2 cut'.format(N))
    percNchi2_ax.set_xlabel('fraction good runs')
    percNchi2_ax.set_ylabel('fraction bad runs')
    percNchi2_ax.set_xlim(0,.4)
    percNchi2_ax.set_ylim(0,.8)

    percNcomb_ax.scatter(perc_all_g, perc_all_b, **plotkwargs)
    percNcomb_ax.set_title('fraction runs with N >= {} failing runs chi2 or pull cut'.format(N))
    percNcomb_ax.set_xlabel('fraction good runs')
    percNcomb_ax.set_ylabel('fraction bad runs')
    percNcomb_ax.set_xlim(0,.4)
    percNcomb_ax.set_ylim(0,.8)
    box = percNcomb_ax.get_position()
    percNcomb_ax.legend()
    # percNcomb_ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # percNcomb_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
