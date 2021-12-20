"""Runs all steps to create reports for cocite temporal network clustering."""
import tempfile
from datetime import datetime
import os
import multiprocessing

from ..linkage.cocitation import Cocitations
from ..clustering.leiden import TimeCluster
from ..clustering.reports import ClusterReports

num_processes = multiprocessing.cpu_count()


def run(
    basepath,
    cociteOutpath,
    timeclusterOutpath,
    reportsOutpath,
    resolution,
    intersliceCoupling,
    minClusterSize: int = 1000,
    timerange=(1945, 2005),
    referenceColumnName: str = 'reference',
    numberproc: int = num_processes,
    limitRefLength=False, debug=False
):
    cocites = Cocitations(
        basepath, cociteOutpath, referenceColumnName,  limitRefLength, debug
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
        metadatapath=basepath,
        outpath=reportsOutpath,
        numberProc=numberproc,
        minClusterSize=minClusterSize,
        timerange=(timerange[0], timerange[1] + 3)
    )
    clusterreports.gatherClusterMetadata()
    clusterreports.writeReports()
    print('Done')
