
## Description
This is a modified version of autodqm_ml that takes advantage of the fetching and training scripts to run beta-binomial meta-analysis of 2022 HLT physics data. If you already have a conda environment of AutoDQM_ML installed, and you don't want to install another conda env that is very similar, you can just clone this repo and skip to `pip install -e.` inside the adqm_analysis directory. It will overide the old version of autodqm-ml package you installed earlier. Just do pip instal command again in the original autodqm-ml directory to undo this change. If you do not want the hassle of remembering which adqm-ml was installed, you can create a new conda environment following steps below:

## Installation
**1. Clone repository**
```
git clone https://github.com/chosila/adqm_analysis.git
cd adqm_analysis
```
**2. Install dependencies**

Dependencies are listed in ```environment.yml``` and installed using `conda`. If you do not already have `conda` set up on your system, you can install (for linux) with:
```
curl -O -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
```
You can then set `conda` to be available upon login with
```
~/miniconda3/bin/conda init # adds conda setup to your ~/.bashrc, so relogin after executing this line
```

Once `conda` is installed and set up, install dependencies with (warning: this step may take a while)
```
conda env create -f environment.yml <path to install conda env>
```

Some packages cannot be installed via `conda` or take too long and need to be installed with `pip` (after activating your `conda` env above):
```
pip install yahist
```

Note: if you are running on `lxplus`, you may run into permissions errors, which may be fixed with:
```
chmod 755 -R /afs/cern.ch/user/s/<your_user_name>/.conda
```
and then rerunning the command to create the `conda` env. The resulting `conda env` can also be several GB in size, so it may also be advisable to specify the installation location in your work area if running on `lxplus`, i.e. running the `conda env create` command with `-p /afs/cern.ch/work/...`.

**3. Install autodqm-ml**

Install with:
```
pip install -e .
```

Once your setup is installed, you can activate your python environment with
```
conda activate adqm_analysis
```

**Note**: `CMSSW` environments can interfere with `conda` environments. Recommended to unset your CMSSW environment (if any) by running
```
eval `scram unsetenv -sh`
```
before attempting installation and each time before activating the `conda` environment.



## Instructions to use this package for meta-analysis


A lot of these instructions are very similar to the instructions on the AutoDQM_ML tutorial.

1. Make JSON of histograms to use for this analysis. General shape of the JSON are name of the subsystem as the key, followed by list of path to histograms within the root file. Example of what it should look like: <https://github.com/chosila/adqm_analysis/blob/adqm_analysis/metadata/histogram_lists/l1tshift_1.json>

How to choose good_runs/bad_runs
    1. go to run registry <https://cmsrunregistry.web.cern.ch/online/global?>. Offline or Online depends on the set of histograms you're working with.
    2. GOOD RUNS: "class = Collisions22", "ls_duration > 77", "<subsystem> = GOOD"
    3. BAD RUNS:  "class = Collisions22", "ls_duration > 13", "<subsystem> = BAD"

NOTE: Since we are fetching a very large number of root files and all the histograms need to be readout, it is very memory intensive and tend to fail if too many histograms are fetched at once. I have found  that around ~15 histograms at a time is within the safe limit of not encountering any memory issues. To deal with this, I split my histogram_list json files into multiple JSON files, and each one is fetched and run individually.

2. fetch the data.

Within the metadata directory, there is also the dataset_lists directory which contains HLT_physics_2022.json. We will be using this data set list for the analysis. The fetching script will read the root files of runs specified in the dataset_lists json file, and read out the histograms specified by your histogram_lists json files. The output will be saved into a .parquet file. The command to do this is:
```
python scripts/fetch_data.py --output_dir "<output dir name>" --contents <path to histogram_lists json> --datasets "metadata/dataset_lists/HLT_physics_2022.json"
```
Output parquet file will bein the the output_dir that you specified.

3. run statistical test

```
python scripts/train.py --input_file "<output dir name>/<output file>.parquet"
                        --output_dir "<new output dir name>"
                        --algorithm "statistical_tester"
                        --tag "beta_binom"
                        --histograms "<path to histogram_list json file"
```

The new output dir should be a different one from before, as the new parquet file created in this step could override your fetched dataset if you use the same output dir. This step creates another parquet file in the new output directory. This new parquet file contains the beta-binomial chi2 and pull-value score of the comparison you just did.


Beta-binomial comparison can be run using multiple reference runs. In the AutoDQM paper, we compared the performance of 1,4, and 8 references. Currently to change how many references are used in the comparison, you need to modify this value <https://github.com/chosila/adqm_analysis/blob/49529a9080ad923a70e3075029ec2328b0a7dee1/autodqm_ml/algorithms/statistical_tester.py#L25>. I apologize for the jank. This was backengineered into the autodqm-ml code and I coudln't find an elegant solution to how to control nRef from the command line.

The output file from this step always has the same name, so if you want to run the same histogram_lists JSON but with different number of reference runs, you will need to rename the parquet file you just created in this step before rerunning the train command, or give it a differnt output directory.


4. convert parquet to csv.

After producing all the histogram lists, you will need to convert this parquet file into a csv, as the final plotting script uses csv. I have a script `HLT_l1tShift_addstat/combinetocsv.py` that converts HLTPhysics1.parquet, HLTPhysics2.parquet....HLTPhysics4.parquet. into a csv called L1T_HLTPhysics.csv. If you also named your parquet files as `<subsystem><number>.parquet`, you can modify this file by changing the range to match the number of files you need <https://github.com/chosila/adqm_analysis/blob/49529a9080ad923a70e3075029ec2328b0a7dee1/HLT_l1tShift_addstat/combinetocsv.py#L8>, as well as the name of the input file <https://github.com/chosila/adqm_analysis/blob/49529a9080ad923a70e3075029ec2328b0a7dee1/HLT_l1tShift_addstat/combinetocsv.py#L10> and the name of the output file <https://github.com/chosila/adqm_analysis/blob/49529a9080ad923a70e3075029ec2328b0a7dee1/HLT_l1tShift_addstat/combinetocsv.py#L21>. Move the output of this merge into a csv directory for the next step.

NOTE: parquet files corresponding to different number of references should not be merged together.

5. Run the ROC making script

We will use the output of the merged csv to run the scripts to plot "ROC" curves that are used in the AutoDQM paper.

```
python make_roc.py <algo> <subsystem> <N>
```

This will create pdfs in the plots/ directory. This script was written specifically for my study which had csv files corresponding to a study using 1,4, and 8 reference runs. If you onlyt have 1 csv file, you can modify this line <https://github.com/chosila/adqm_analysis/blob/49529a9080ad923a70e3075029ec2328b0a7dee1/make_roc.py#L30> to `zip(['1_REF'], ['-rD'], ['purple'])` and this line <https://github.com/chosila/adqm_analysis/blob/49529a9080ad923a70e3075029ec2328b0a7dee1/make_roc.py#L33> to the name of your csv file.