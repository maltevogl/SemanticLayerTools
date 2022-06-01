import os
import re
import multiprocessing
from tqdm import tqdm
import pandas as pd
import numpy as np
import igraph as ig


class CaluculateCentralitues():
    """Calculate centralities for networks."""

    def __init__(
        self, clusterFile: str, graphPath: str, clusterMetadataPath: str,
        outpath: str, minClusterSize: int = 1000, idcolumn: str = 'nodeID'
    ):
        clusterdf = pd.read_csv(clusterFile)
        basedata = clusterdf.groupby(
            ['year', 'cluster']
        ).size().to_frame('counts').reset_index()
        self.largeClusterList = list(
            basedata.groupby('cluster').sum().query(
                f'counts > {self.minClusterSize}'
            ).index
        )
        self.clusternodes = clusterdf.query(
            'cluster in @self.largeClusterList'
        )
        self.graphpath = graphPath
        self.metadatapath = clusterMetadataPath
        self.outpath = os.path.join(outpath, clusterFile.split(os.pathsep)[-1])
        self.idColumn = idcolumn
        if os.path.isdir(self.outpath):
            raise OSError(f'Output folder {self.outpath} exists. Aborting.')
        else:
            os.mkdir(self.outpath)
            for clu in self.largeClusterList:
                os.mkdir(os.path.join(self.outpath, f'Cluster_{clu}'))

    def _mergeData(self, filename):
        """Merge metadata for cluster nodes.

        Writes all metadata for nodes in cluster to folders.

        :param filename: Metadata input filename
        :type filename: str
        """
        filepath = os.path.join(self.metadatapath, filename)
        data = pd.read_json(filepath, lines=True)
        selectMerge = data.merge(
            self.clusternodes,
            left_on=self.idcolumn,
            right_on='node',
            how='inner'
        )
        if selectMerge.shape[0] > 0:
            for clu, g0 in selectMerge.groupby('cluster'):
                g0.to_json(
                    os.path.join(
                        self.outpath,
                        f'Cluster_{clu}',
                        'merged_' + filename
                    ), orient='records', lines=True
                )
        return ''

    def gatherClusterMetadata(self):
        """Initial gathering of metadata for clusters.

        For all files in the metadata path, call `_mergeData` if the found
        year in the filename falls in the bounds.

        This step needs to be run once, the all cluster metadata is generated
        and can be reused.
        """
        filenames = os.listdir(self.metadatapath)
        yearFiles = []
        for x in filenames:
            try:
                year = int(re.findall(r'\d{4}', x)[0])
            except Exception:
                raise
            if self.timerange[0] <= year <= self.timerange[1]:
                yearFiles.append(x)
        with multiprocessing.Pool(self.numberProc) as pool:
            _ = pool.map(self._mergeData, tqdm(yearFiles, leave=False))
        return

    def run(
        self, centrality: str = "all",
        timerange: tuple = (1945, 2005), useGC: bool = True
    ):
        """Run calculation."""
        bins = 10 ** np.linspace(np.log10(0.00001), np.log10(1.0), 100)
        with open(f'{self.outpath}centralities_logbin.csv', 'a') as result:
            for year in tqdm(self.yearrange):
                if useGC is False:
                    graph = ig.Graph.Read_Ncol(
                        f'{self.graphPath}{year}_meta.ncol',
                        names=True,
                        weights=True
                    )
                elif useGC is True:
                    graph = ig.Graph.Read_Pajek(
                        f'{self.graphPath}{year}_meta_GC.net',
                        names=True,
                        weights=True
                    )
                authority = graph.authority_score(scale=True)
                betweenness = graph.betweenness(directed=False)
                closeness = graph.closeness(mode='all', normalized=True)
                degrees = graph.degree(graph.vs, mode='all')
                maxDeg = max(degrees)
                maxBet = max(betweenness)
                # Write results to files
                for centName, centrality in [
                    ('Authority', authority),
                    ('Betweenness', [x/maxBet for x in betweenness]),
                    ('Closeness', closeness),
                    # ('Eigenvector', eigenvector),
                    ('Degree', [x/maxDeg for x in degrees])
                ]:
                    histoCentrality = np.histogram(centrality, bins=bins)
                    for val, bin_ in zip(histoCentrality[0], histoCentrality[1]):
                        result.write(
                            f"{centName}, {year}, {bin_}, {val/len(graph.vs)}\n")
        return "Done"
