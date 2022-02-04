import numpy
import random
from sklearn import metrics
from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)

def calc_auc(y, pred, sample_weight = None, interp = 10000):
    """
    Make interpolated roc curve and calculate AUC.
    Keyword arguments:
    y -- array of labels
    pred -- array of mva scores
    sample_weight -- array of per-event weights
    interp -- number of points in resulting fpr and tpr arrays
    """

    if sample_weight is None:
        sample_weight = numpy.ones_like(y)

    fpr, tpr, thresh = metrics.roc_curve(
        y,
        pred,
        pos_label = 1,
        sample_weight = sample_weight
    )

    fpr = sorted(fpr)
    tpr = sorted(tpr)

    fpr_interp = numpy.linspace(0, 1, interp)
    tpr_interp = numpy.interp(fpr_interp, fpr, tpr) # recalculate tprs at each fpr

    auc = metrics.auc(fpr, tpr)

    results = {
        "fpr" : fpr_interp,
        "tpr" : tpr_interp,
        "auc" : auc
    }
    return results

def bootstrap_indices(x):
    """
    Return array of indices of len(x) to make bootstrap resamples 
    """

    return numpy.random.randint(0, len(x), len(x))
    

def calc_roc_and_unc(y, pred, sample_weight = None, n_bootstrap = 100, interp = 10000):
    """
    Calculates tpr and fpr arrays (with uncertainty for tpr) and auc and uncertainty
    Keyword arguments:
    y -- array of labels
    pred -- array of mva scores
    sample_weight -- array of per-event weights
    n_bootstrap -- number of bootstrap resamples to use for calculating uncs
    interp -- number of points in resulting fpr and tpr arrays
    """

    y = numpy.array(y)
    pred = numpy.array(pred)

    if sample_weight is None:
        sample_weight = numpy.ones_like(y)
    else:
        sample_weight = numpy.array(sample_weight)

    logger.debug("[roc_tools.py : calc_roc_and_unc] Calculating AUC and uncertainty with %d bootstrap samples." % (n_bootstrap))
    results = calc_auc(y, pred, sample_weight)
    fpr, tpr, auc = results["fpr"], results["tpr"], results["auc"]
    
    fprs = [fpr] 
    tprs = [tpr]
    aucs = [auc]

    for i in tqdm(range(n_bootstrap)):
        idx = bootstrap_indices(y)
        
        label_bootstrap   = y[idx]
        pred_bootstrap    = pred[idx]
        weights_bootstrap = sample_weight[idx]

        results_bootstrap = calc_auc(label_bootstrap, pred_bootstrap, weights_bootstrap, interp)
        fpr_b, tpr_b, auc_b = results_bootstrap["fpr"], results_bootstrap["tpr"], results_bootstrap["auc"]
        fprs.append(fpr_b)
        tprs.append(tpr_b)
        aucs.append(auc_b)

    unc = numpy.std(aucs)
    tpr_mean = numpy.mean(tprs, axis=0)
    tpr_unc = numpy.std(tprs, axis=0)
    fpr_mean = numpy.mean(fprs, axis=0)

    results = {
        "auc" : auc,
        "auc_unc" : unc,
        "fpr" : fpr_mean,
        "tpr" : tpr_mean,
        "tpr_unc" : tpr_unc
    }

    return results


def find_nearest(array,value):
    val = numpy.ones_like(array)*value
    idx = (numpy.abs(array-val)).argmin()
    return array[idx], idx
