import os
import time
import re
from tqdm import tqdm

import igraph as ig
import leidenalg as la


class TimeCluster():
    """Cluster time-sliced data with the Leiden algorithm.

    Calculates temporal clusters of e.g. time-sliced cocitation or citation
    data, using the Leiden algorithm . Two nodes are assumed to be identical in
    different year slices, if the node name is the same.
    This could be e.g. the bibcode or DOI.

    Input files are assumed to include the year in the filename, have an ending
    `_GC.net` to denote their giant component character and should be in Pajek
    format.

    The resolution parameter can be seen as a limiting density, above
    which neighbouring nodes are considered a cluster. The interslice coupling
    describes the influcence of yearly order on the clustering process. See doc
    for the Leiden algorithm for more detailed info.

    :param inpath: Path for input network data
    :type inpath: str
    :param outpath: Path for writing output data
    :type outpath: str
    :param resolution: Main parameter for the clustering quality function (Constant Pots Model)
    :type resolution: float
    :param intersliceCoupling: Coupling parameter between two year slices, also influences cluster detection
    :type intersliceCoupling: float
    :param timerange: The time range for considering input data (default=1945,2005))
    :type timerange: tuple
    :raises OSError: If the output file already exists at class instantiation

    .. seealso::
       Traag, V.A., Waltman. L., Van Eck, N.-J. (2018).
       From Louvain to Leiden: guaranteeing well-connected communities.
       Scientific reports, 9(1), 5233. 10.1038/s41598-019-41695-z
    """

    def __init__(
        self, inpath: str, outpath: str,
        resolution: float = 0.003,
        intersliceCoupling: float = 0.4,
        timerange: tuple = (1945, 2005),
        useGC: bool = True,
    ):
        starttime = time.time()
        self.inpath = inpath
        self.outpath = outpath
        self.res_param = resolution
        self.interslice_param = intersliceCoupling
        self.timerange = timerange

        self.outfile = os.path.join(
            outpath,
            f'timeclusters_{timerange[0]}-{timerange[1]}_res_{resolution}_intersl_{intersliceCoupling}.csv'
        )
        if os.path.isfile(self.outfile):
            raise OSError(f'Output file at {self.outfile} exists. Aborting.')

        if useGC is True:
            edgefiles = [x for x in os.listdir(inpath) if x.endswith('_GC.net')]
        elif useGC is False:
            edgefiles = [x for x in os.listdir(inpath) if x.endswith('.ncol')]

        self.graphDict = {}

        for idx in tqdm(range(len(edgefiles)), leave=False):
            try:
                year = re.findall(r'\d{4}', edgefiles[idx])[0]
            except Exception:
                raise
            if timerange[0] <= int(year) <= timerange[1]:
                if useGC is True:
                    graph = ig.Graph.Read_Pajek(os.path.join(inpath, edgefiles[idx]))
                elif useGC is False:
                    graph = ig.Graph.Read_Ncol(os.path.join(inpath, edgefiles[idx]))
                self.graphDict[year] = graph

        self.optimiser = la.Optimiser()

        print(
            "Graphs between "
            f"{min(list(self.graphDict.keys()))} and "
            f"{max(list(self.graphDict.keys()))} "
            f"loaded in {time.time() - starttime} seconds."
        )

    def optimize(self, clusterSizeCompare: int = 1000):
        """Optimize clusters accross time slices.

        This runs the actual clustering and can be very time and memory
        consuming for large networks. Depending on the obtained cluster results,
        this method has to be run iteratively with varying resolution parameter.
        Output is written to file, with filename containing chosen parameters.

        The output CSV contains information on which node in which year belongs
        to which cluster. As a first measure of returned clustering, the method
        prints the number of clusters found above a threshold defined by
        `clusterSizeCompare`. This does not influence the output clustering.

        :param clusterSizeCompare: Threshold for `interesting` clusters
        :type clusterSizeCompare: int
        :returns: Tuple of output file path and list of found clusters in tuple format (node, year, cluster)
        :rtype: tuple

        .. seealso::
           Documentation of time-layer creation routine:
           `Leiden documentation <https://leidenalg.readthedocs.io/en/latest/multiplex.html#temporal-community-detection>`_
        """
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
        largeclu = [
            (x, len(x.vs)) for x in subgraphs if len(x.vs) > clusterSizeCompare
        ]
        print(
            f'Finished in {time.time() - starttime} seconds.'
            f"Found {len(subgraphs)} clusters, with {len(largeclu)} larger then {clusterSizeCompare} nodes."
        )

        return self.outfile, commun
