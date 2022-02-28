
import time
import os
import multiprocessing

from ..linkage.cocitation import Cocitations
from ..clustering.leiden import TimeCluster
from ..clustering.reports import ClusterReports

num_processes = multiprocessing.cpu_count()


def run(
    inputFilepath: str,
    cociteOutpath: str,
    timeclusterOutpath: str,
    reportsOutpath: str,
    resolution: float,
    intersliceCoupling: float,
    minClusterSize: int = 1000,
    timerange: tuple = (1945, 2005),
    referenceColumnName: str = 'reference',
    numberproc: int = num_processes,
    limitRefLength=False, debug=False
):
    """Runs all steps of the temporal clustering pipepline.

    Creates cocitation networks, finds temporal clusters, writes report files
    for large clusters.

    Default time range is 1945 to 2005. Minimal size for considered clusters is
    1000 nodes. Lists of references are assumed to be contained in column
    "reference".

    By default this routine takes all available cpu cores. Limit this
    to a lower value to allow parallel performance of other tasks.

    :param inputFilepath:  Path to corpora input data
    :type text: str
    :param cociteOutpath: Output path for cocitation networks
    :type text: str
    :param timeclusterOutpath: Output path for time clusters
    :type text: str
    :param reportsOutpath: Output path for reports
    :type text: str
    :param resolution: Main parameter for the clustering quality function (Constant Pots Model)
    :type resolution: float
    :param intersliceCoupling: Coupling parameter between two year slices, also influences cluster detection
    :type intersliceCoupling: float
    :param minClusterSize: The minimal cluster size, above which clusters are considered (default=1000)
    :type minClusterSize: int
    :param timerange: Time range to evalute clusters for (usefull for limiting computation time, default = (1945, 2005))
    :type timerange: tuple
    :param referenceColumnName: Column name containing the references of a publication
    :type referenceColumnName: str
    :param numberProc: Number of CPUs the package is allowed to use (default=all)
    :type numberProc: int
    :param limitRefLength: Either False or integer giving the maximum number of references a considered publication is allowed to contain
    :type limitRefLength: bool or int
    """
    for path in [cociteOutpath, timeclusterOutpath, reportsOutpath]:
        os.makedirs(path)
    starttime = time.time()
    cocites = Cocitations(
        inpath=inputFilepath,
        outpath=cociteOutpath,
        columnName=referenceColumnName,
        numberProc=numberproc,
        limitRefLength=limitRefLength,
        timerange=timerange,
        debug=debug
    )
    cocites.processFolder()
    timeclusters = TimeCluster(
        inpath=cociteOutpath,
        outpath=timeclusterOutpath,
        resolution=resolution,
        intersliceCoupling=intersliceCoupling,
        timerange=timerange,
        debug=debug
    )
    timeclfile, _ = timeclusters.optimize()
    clusterreports = ClusterReports(
        infile=timeclfile,
        metadatapath=inputFilepath,
        outpath=reportsOutpath,
        numberProc=numberproc,
        minClusterSize=minClusterSize,
        timerange=(timerange[0], timerange[1] + 3)
    )
    clusterreports.gatherClusterMetadata()
    clusterreports.writeReports()
    print(f'Done after {time.time() - starttime} seconds.')
