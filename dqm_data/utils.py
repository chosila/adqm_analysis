import os

def get_files(path, limit = None):
    """
    Return list of all matching root files accessible through xrootd
    """

    files = []

    dirs = os.popen("xrdfs root://eoscms.cern.ch/ ls %s" % path).read().split("\n")
    for dir in dirs:
        if dir == "":
            continue
        if ".root" in dir:
            files.append(dir)
        else:
            files += get_files(dir)

        if limit is not None:
            if len(files) >= limit:
                return files[:limit]

    for idx, file in enumerate(files):
        files[idx] = "root://eoscms.cern.ch/" + file

    return files


def get_histograms(files):
    
    return
