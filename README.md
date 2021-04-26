# AutoDQM_ML

## Installation
**1. Clone repository**
```
git clone https://github.com/AutoDQM/AutoDQM_ML.git 
cd AutoDQM_ML
```
**2. Install dependencies**

Dependencies are listed in ```environment.yml```. Install with
```
conda env create -f environment.yml
```

**3. Install autodqm-ml**

**Users** can install with:
```
python setup.py install
```
**Developers** are suggested to install with:
```
pip install -e .
```
to avoid rerunning the whole installation every time there is a change.

Once your setup is installed, you can activate your python environment with
```
conda activate autodqm-ml
```

**Note**: `CMSSW` environments can interfere with `conda` environments. Recommended to unset your CMSSW environment (if any) by running
```
eval `scram unsetenv -sh`
```
before attempting installation and each time before activating the `conda` environment.
