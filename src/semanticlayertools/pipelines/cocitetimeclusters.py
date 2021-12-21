"""Runs all steps to create reports for cocite temporal network clustering."""
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
