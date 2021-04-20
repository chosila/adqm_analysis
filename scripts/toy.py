import uproot

import dqm_data.utils


hpath = "DQMData/Run {}/CSC/Run summary/CSCOfflineMonitor"
files = dqm_data.utils.get_files("/eos/cms/store/group/comm_dqm/DQMGUI_data/Run2018/SingleMuon/", limit = 10)
print(files)

for file in files:
    run = int(file.split("/")[-1].split("__")[0][-6:])
    with uproot.open(file) as f:
        hist_1d = f[hpath.format(run) + "/recHits/hRHTimingp12"].to_numpy()
        hist_2d = f[hpath.format(run) + "/Occupancy/hOStripSerial"].to_numpy()
        print("1d hist", hist_1d)
        print("2d hist", hist_2d)
        #entries = f[hpath.format(run)].classnames()
        #for dtype, entry in entries.items():

