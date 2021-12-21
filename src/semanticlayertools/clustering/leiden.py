import os
import time
import re
from typing import TypeVar

from tqdm import tqdm

import igraph as ig
import leidenalg as la

debugVar = TypeVar('debugVar', bool, str)


class TimeCluster():
    """Cluster time-sliced data with the Leiden algorithm."""

    def __init__(
        self, inpath: str, outpath: str,
        resolution: float = 0.003, intersliceCoupling: float = 0.4,
        timerange: tuple = (1945, 2005),
        debug: debugVar = False
    ):
        starttime = time.time()
        self.inpath = inpath
        self.outpath = outpath
        self.res_param = resolution
        self.interslice_param = intersliceCoupling
        self.timerange = timerange
        self.debug = debug

        self.outfile = os.path.join(
            outpath,
            f'timeclusters_{timerange[0]}-{timerange[1]}_res_{resolution}_intersl_{intersliceCoupling}.csv'
        )
        if os.path.isfile(self.outfile):
            raise OSError(f'Output file at {self.outfile} exists. Aborting.')

        edgefiles = [x for x in os.listdir(inpath) if x.endswith('_GC.net')]

        self.graphDict = {}

        for idx in tqdm(range(len(edgefiles)), leave=False):
            try:
                year = re.findall(r'\d{4}', edgefiles[idx])[0]
            except:
                raise
            if timerange[0] <= int(year) <= timerange[1]:
                graph = ig.Graph.Read_Pajek(os.path.join(inpath, edgefiles[idx]))
                self.graphDict[year] = graph

        self.optimiser = la.Optimiser()

        print(
            "Graphs between "
            f"{min(list(self.graphDict.keys()))} and "
            f"{max(list(self.graphDict.keys()))} "
            f"loaded in {time.time() - starttime} seconds."
        )

    def optimize(self, clusterSizeCompare: int=1000):
        """Optimize clusters accross time slices."""
        starttime = time.time()

        layers, interslice_layer, _ = la.time_slices_to_layers(
            list(self.graphDict.values()),
            interslice_weight=self.interslice_param,
            vertex_id_attr='name'
        )
        print('\tSet layers.')

        partitions = [
            la.CPMVertexPartition(
                H,
                node_sizes='node_size',
                weights='weight',
                resolution_parameter=self.res_param
            ) for H in layers
        ]
        print('\tSet partitions.')

        interslice_partition = la.CPMVertexPartition(
            interslice_layer,
            resolution_parameter=0,
            node_sizes='node_size',
            weights='weight'
        )
        print('\tSet interslice partions.')

        self.optimiser.optimise_partition_multiplex(
            partitions + [interslice_partition]
        )

        subgraphs = interslice_partition.subgraphs()

        commun = []
        for idx, part in enumerate(subgraphs):
            nodevals = [
                (
                    x['name'],
                    list(self.graphDict.keys()).pop(x['slice']),
                    idx
                ) for x in part.vs
            ]
            commun.extend(nodevals)

        with open(self.outfile, 'w') as outfile:
            outfile.write('node,year,cluster\n')
            for elem in commun:
                outfile.write(
                    f"{elem[0]},{elem[1]},{elem[2]}\n"
                )
        largeclu = [(x,len(x.vs)) for x in subgraphs if len(x.vs)>clusterSizeCompare]
        print(
            f'Finished in {time.time() - starttime} seconds.'
            f"Found {len(subgraphs)} clusters, with {len(largeclu)} larger then {clusterSizeCompare} nodes."
        )

        return self.outfile, commun
