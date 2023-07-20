import numpy
from scipy.stats import ks_2samp
import scipy.stats as stats 

from autodqm_ml.algorithms.anomaly_detection_algorithm import AnomalyDetectionAlgorithm

import logging
logger = logging.getLogger(__name__)

class StatisticalTester(AnomalyDetectionAlgorithm):
    """
    Class to perform statistical-based tests for judging anomalies.
    For 1d histograms, it will perform a ks-test.
    For 2d histograms, it will perform a pull-value test.
    """

    def __init__(self, **kwargs):
        super(StatisticalTester, self).__init__(**kwargs)
        self.reference = kwargs.get("reference", None)

    def predict(self):
        if self.reference is None:
            self.reference = self.df.run_number[100]
            logger.warning("[StatisticalTester : predict] No reference run was provided, will use the first run in the df (%d) as the reference." % (self.reference))
        
        self.reference_hists = self.df[self.df.run_number == self.reference][0] 

        for histogram, info in self.histograms.items():
            score = numpy.zeros(len(self.df))
            for i in range(len(score)):
                ##if info["n_dim"] == 1:
                ##    d_value, p_value = self.ks_test(self.df[histogram][i], self.reference_hists[histogram])
                ##    score[i] = d_value
                ##elif info["n_dim"] == 2:
                ##    pull_value = self.pull_value_test(self.df[histogram][i], self.reference_hists[histogram])
                ##    score[i] = pull_value
                print('ref :', self.reference)
                print('data :', self.df['run_number'][i])
                d_value, p_value = self.beta_binomial(self.df[histogram][i], self.reference_hists[histogram])
                score[i] = p_value
            self.add_prediction(histogram, score)


    def ks_test(self, target, reference):
        """
        Perform ks test on two 1d histograms.
        """

        score, p_value = ks_2samp(
                numpy.array(target), 
                numpy.array(reference) 
        )

        return score, p_value

    def beta_binomial(self, target, reference):
        """beta_binmial works on both 1D and 2D"""
        # need the float64 for betabinom
        print('target:', target)
        print('reference:', reference)
        
        data_hist_raw = numpy.round(numpy.copy(numpy.float64(target   ))) # multiply by large number make bin values look like ints 
        ref_hist_raw = [numpy.round(numpy.copy(numpy.float64(reference)))]
        #print('data_hist_raw: ', data_hist_raw)
        #print('ref_hist_raw: ', ref_hist_raw)

        # if data or refernce is 0, does not run BetaBinom
        if numpy.sum(target) <= 0:
            return None

        # num entries
        data_hist_entries = numpy.sum(data_hist_raw)
        ref_hist_entries = numpy.sum(ref_hist_raw)

        # only filled bins used for chi2
        nBinsUsed = numpy.count_nonzero(numpy.add(ref_hist_raw, data_hist_raw))
        
        # calculate pull and chi2, and get probability-weighted reference histogram
        [pull_hist, ref_hist_prob_wgt] = self.pull(data_hist_raw, ref_hist_raw, tol = 0.01)
        pull_hist = pull_hist*numpy.sign(data_hist_raw - ref_hist_prob_wgt)
        print('pull_hist:', pull_hist)
        chi2 = numpy.square(pull_hist).sum() / nBinsUsed
        print('chi2:', chi2)
        max_pull = self.maxPullNorm(numpy.amax(pull_hist), nBinsUsed)
        min_pull = self.maxPullNorm(numpy.amin(pull_hist), nBinsUsed)
        if abs(min_pull) > max_pull:
            max_pull = min_pull

        # access per-histogram settings for max_pull and chi2
        # if 'opts' in histpair.config.keys():
        #     for opt in histpair.config['opts']:
        #         if 'pull_cap' in opt: pull_cap = float(opt.split('=')[1])
        #         if 'chi2_cut' in opt: chi2_cut = float(opt.split('=')[1])
        # if 'pull_cut' in opt: pull_cut = float(opt.split('=')[1])

        # nRef=1
        # if nRef == 1:
        #     Ref_entries_str = str(int(ref_hist_entries[0]))
        # else:
        # Ref_Entries_str = " - ".join([str(int(min(ref_hist_entries))), str(int(max(ref_hist_entries)))])
        print('max_pull: ', max_pull)
        
        return chi2, max_pull

    def pull(self, D_raw, R_list_raw, tol = 0.01):
        probs = []
        nRef=1
        for R_raw in R_list_raw:
            ## Compute per-bin probabilities with beta-binomial function
            ## Protect against zero values with a floor at 10^-300 (37 sigma)
            probs.append(numpy.maximum(self.ProbRel(D_raw, R_raw, 'BetaB', tol), pow(10, -300)) )
            
            ## Per-bin probability is the per-bin average over all ref hists
            prob = numpy.array(probs).sum(axis=0) / nRef
            pull = self.Sigmas(prob)

            ## Reference histogram weighted by per-bin probabilities
            R_prob_wgt_avg = numpy.zeros_like(D_raw)
            
            for iR in range(len(R_list_raw)):
                R_raw = R_list_raw[iR]
                ## Get reference hist normalized to 1
                R_prob_wgt = R_raw / numpy.sum(R_raw)
                ## Compute per-bin probabilities relative to sum of probabilites
                prob_rel = numpy.divide(probs[iR], numpy.array(probs).sum(axis=0))
                ## Scale normalized reference by per-bin relative probabilities
                R_prob_wgt = numpy.multiply(R_prob_wgt, prob_rel)
                ## Add into average probability-weighted distribution
                R_prob_wgt_avg = numpy.add(R_prob_wgt_avg, R_prob_wgt)
                
                ## Normalize to data
                R_prob_wgt_avg = R_prob_wgt_avg * numpy.sum(D_raw)
                
                return [pull, R_prob_wgt_avg]

    def maxPullNorm(self, maxPull, nBinsUsed, cutoff=pow(10,-15)):
        sign = numpy.sign(maxPull)
        ## sf (survival function) better than 1-cdf for large pulls (no precision error)
        probGood = stats.chi2.sf(numpy.power(min(abs(maxPull), 37), 2), 1)

        ## Use binomial approximation for low probs (accurate within 1%)
        if nBinsUsed * probGood < 0.01:
            probGoodNorm = nBinsUsed * probGood
        else:
            probGoodNorm = 1 - numpy.power(1 - probGood, nBinsUsed)

        pullNorm = self.Sigmas(probGoodNorm) * sign

        return pullNorm


    ## Mean expectation for number of expected data events
    def Mean(self, Data, Ref, func):
        nRef = Ref.sum()
        nData = Data.sum()
        if func == 'Gaus1' or func == 'Gaus2':
            return 1.0*nData*Ref/nRef
            
        ## https://en.wikipedia.org/wiki/Beta-binomial_distribution#Moments_and_properties
        ## Note that n = nData, alpha = Ref+1, and beta = nRef-Ref+1, alpha+beta = nRef+2
        if func == 'BetaB' or func == 'Gamma':
            return 1.0*nData*(Ref+1)/(nRef+2)

        print('\nInside Mean, no valid func = %s. Quitting.\n' % func)
        sys.exit()
        
    ## Standard deviation of gaussian and beta-binomial functions
    def StdDev(self, Data, Ref, func):
        nData = Data.sum()
        nRef = Ref.sum()
        mask = Ref > 0.5*nRef
        if func == 'Gaus1':
            ## whole array is calculated using the (Ref <= 0.5*nRef) formula, then the ones where the
            ## conditions are actually failed is replaced using mask with the (Ref > 0.5*nRef) formula
            output = 1.0*nData*numpy.sqrt(numpy.clip(Ref, a_min=1, a_max=None))/nRef
            output[mask] = (1.0*nData*numpy.sqrt(numpy.clip(nRef-Ref, a_min=1, a_max=None)))[mask]/nRef
        elif func == 'Gaus2':
            ## instead of calculating max(Ref, 1), set the whole array to have a lower limit of 1
            clipped = numpy.clip(Ref, a_min=1, a_max=None)
            output = 1.0*nData*numpy.sqrt( clipped/numpy.square(nRef) + self.Mean(nData, Ref, nRef, func)/numpy.square(nData) )
            clipped = numpy.clip(nRef-Ref, a_min=1, a_max=None)
            output[mask] = (1.0*nData*numpy.sqrt( clipped/numpy.square(nRef) + (nData - self.Mean(nData, Ref, nRef, func))/numpy.square(nData) ))
        elif (func == 'BetaB') or (func == 'Gamma'):
            output = 1.0*numpy.sqrt( nData*(Ref+1)*(nRef-Ref+1)*(nRef+2+nData) / (numpy.power(nRef+2, 2)*(nRef+3)) )
            
        else:
            print('\nInside StdDev, no valid func = %s. Quitting.\n' % func)
            sys.exit()

        return output


    ## Number of standard devations from the mean in any function
    def numStdDev(self, Data, Ref, func):
        nData = Data.sum()
        nRef = Ref.sum()
        return (Data - self.Mean(Data, Ref, func)) / self.StdDev(Data, Ref, func)


    ## Predicted probability of observing Data / nData given a reference of Ref / nRef
    def Prob(self, Data, nData, Ref, nRef, func, tol=0.01):
        scaleTol = numpy.power(1 + numpy.power(Ref * tol**2, 2), -0.5)
        nRef_tol = numpy.round(scaleTol * nRef)
        Ref_tol =  numpy.round(Ref * scaleTol )
        nData_arr = numpy.zeros_like(Data) + numpy.float64(nData)

        if func == 'Gaus1' or func == 'Gaus2':
            return stats.norm.pdf(self.numStdDev(Data, Ref_tol, func) )
        if func == 'BetaB':
            ## https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.betabinom.html
            ## Note that n = nData, alpha = Ref+1, and beta = nRef-Ref+1, alpha+beta = nRef+2
            return stats.betabinom.pmf(Data, nData_arr, Ref_tol + 1, nRef_tol - Ref_tol + 1)
        ## Expression for beta-binomial using definition in terms of gamma functions
        ## https://en.wikipedia.org/wiki/Beta-binomial_distribution#As_a_compound_distribution
        if func == 'Gamma':
            ## Note that n = nData, alpha = Ref+1, and beta = nRef-Ref+1, alpha+beta = nRef+2
            n_  = nData_arr
            k_  = Data
            a_  = Ref_tol + 1
            b_  = nRef_tol - Ref_tol + 1
            ab_ = nRef_tol + 2
            logProb  = gammaln(n_+1) + gammaln(k_+a_) + gammaln(n_-k_+b_) + gammaln(ab_)
            logProb -= ( gammaln(k_+1) + gammaln(n_-k_+1) + gammaln(n_+ab_) + gammaln(a_) + gammaln(b_) )
            return numpy.exp(logProb)

        print('\nInside Prob, no valid func = %s. Quitting.\n' % func)
        sys.exit()


    ## Predicted probability relative to the maximum probability (i.e. at the mean)
    def ProbRel(self, Data, Ref, func, tol=0.01):
        nData = Data.sum()
        nRef = Ref.sum()
        ## Find the most likely expected data value
        exp_up = numpy.clip(numpy.ceil(self.Mean(Data, Ref, 'Gaus1')), a_min=None, a_max=nData) # make sure nothing goes above nData
        exp_down = numpy.clip(numpy.floor(self.Mean(Data, Ref, 'Gaus1')), a_min=0, a_max=None) # make sure nothing goes below zero
        ## Find the maximum likelihood
        maxProb_up = self.Prob(exp_up, nData, Ref, nRef, func, tol)
        maxProb_down = self.Prob(exp_down, nData, Ref, nRef, func, tol)
        maxProb = numpy.maximum(maxProb_up, maxProb_down)
        thisProb = self.Prob(Data, nData, Ref, nRef, func, tol)
        
        ## Sanity check to not have relative likelihood > 1
        ratio = numpy.divide(thisProb, maxProb, out=numpy.zeros_like(thisProb), where=maxProb!=0)
        cond = thisProb > maxProb
        ratio[cond] = 1
        
        return ratio


    ## Convert relative probability to number of standard deviations in normal distribution
    def Sigmas(self, probRel):
        ## chi2.isf function fails for probRel < 10^-323, so cap at 10^-300 (37 sigma)
        probRel = numpy.maximum(probRel, pow(10, -300))
        return numpy.sqrt(stats.chi2.isf(probRel, 1))
        ## For very low prob, can use logarithmic approximation:
        ## chi2.isf(prob, 1) = 2 * (numpy.log(2) - numpy.log(prob) - 3)

    def pull_value_test(self, target, reference):
        """
        Perform pull value test on two 2d histograms.
            
        :param histogram: Histogram object
        :type histogram: autodqm_ml.data_formats.histogram.Histogram
        :param threshold: value for which to declare a histogram anomalous
        :type threshold: float
        """
    
        # TODO
        return 0
