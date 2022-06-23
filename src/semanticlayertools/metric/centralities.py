import os
import re
import multiprocessing
from tqdm import tqdm
import pandas as pd
import numpy as np
import igraph as ig


class CalculateCentralities():
    """Calculate centralities for networks."""

    def __init__(
        self,
        clusterFile: str,
        graphPath: str,
        outPath: str,
    ):
        self.outPath = os.path.join(
            outPath, clusterFile.split("/")[-1].split(".")[0]
        )
        self.graphPath = graphPath
        if os.path.isdir(self.outPath):
            raise OSError(f'Output folder {self.outPath} exists. Aborting.')
        else:
            os.mkdir(self.outPath)

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
                        self.outPath,
                        f'Cluster_{clu}',
                        'merged_' + filename
                    ), orient='records', lines=True
                )
        return ''

    def setupClusterData(
        self, clusterFile: str, clusterMetadataPath: str,
        minClusterSize: int = 1000,
        idcolumn: str = 'nodeID'
    ):
        """Initial gathering of metadata for previously found time clusters.

        Set minClusterSize to limit clusters considered for analysis.

        For all files in the metadata path, call `_mergeData` if the found
        year in the filename falls in the bounds.

        This step needs to be run once, all cluster metadata is generated
        and can be reused"""
        self.idcolumn = idcolumn
        self.metadatapath = clusterMetadataPath
        clusterdf = pd.read_csv(clusterFile)
        basedata = clusterdf.groupby(
            ['year', 'cluster']
        ).size().to_frame('counts').reset_index()
        self.largeClusterList = list(
            basedata.groupby('cluster').sum().query(
                f'counts > {minClusterSize}'
            ).index
        )
        self.clusternodes = clusterdf.query(
            'cluster in @self.largeClusterList'
        )
        for clu in self.largeClusterList:
            os.mkdir(os.path.join(self.outPath, f'Cluster_{clu}'))

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
        return self

    def fullNetwork(
        self, centrality: str = "all",
        timerange: tuple = (1945, 2005), useGC: bool = True
    ):
        """Run calculation based on Pajek or NCol network data.

        By default timerange is limited to 1945 to 2005. Change accordingly.
        For centralities choose "all" or one of the following:
            "authority"
            "betweenness"
            "closeness"
            "degree"
        """
        self.centralities = {}
        self.timerange = timerange
        bins = 10 ** np.linspace(np.log10(0.00001), np.log10(1.0), 100)
        binsNormal = np.linspace(0, 1, 100)
        with open(f'{self.outPath}centralities_logbin.csv', 'a') as result:
            for year in tqdm(self.timerange):
                try:
                    if useGC is False:
                        graph = ig.Graph.Read_Ncol(
                            f'{self.graphPath}{year}_meta.ncol',
                            names=True,
                            weights=True,
                            directed=False
                        )
                    elif useGC is True:
                        graph = ig.Graph.Read_Pajek(
                            f'{self.graphPath}{year}_meta_GC.net'
                        )
                except FileNotFoundError:
                    continue
                if centrality == "authority" or "all":
                    centrality = graph.authority_score(scale=True)
                    self.centralities[centrality] = centrality
                    histoCentrality = np.histogram(centrality, bins=bins)
                    for val, bin_ in zip(histoCentrality[0], histoCentrality[1]):
                        result.write(
                            f"Authority, {year}, {bin_}, {val/len(graph.vs)}\n")
                elif centrality == "betweenness" or "all":
                    centrality = graph.betweenness(directed=False)
                    maxBet = max(centrality)
                    centrality_norm = [x/maxBet for x in centrality]
                    self.centralities[centrality] = centrality_norm
                    histoCentrality = np.histogram(centrality_norm, bins=bins)
                    for val, bin_ in zip(histoCentrality[0], histoCentrality[1]):
                        result.write(
                            f"Betweenness, {year}, {bin_}, {val/len(graph.vs)}\n")
                elif centrality == "degree" or "all":
                    centrality = graph.degree(graph.vs, mode='all')
                    maxDeg = max(centrality)
                    centrality_norm = [x/maxDeg for x in centrality]
                    self.centralities[centrality] = centrality_norm
                    histoCentrality = np.histogram(centrality_norm, bins=bins)
                    for val, bin_ in zip(histoCentrality[0], histoCentrality[1]):
                        result.write(
                            f"Degree, {year}, {bin_}, {val/len(graph.vs)}\n")
                elif centrality == "closeness" or "all":
                    centrality = graph.closeness(mode='all', normalized=True)
                    self.centralities[centrality] = centrality
                    histoCentrality = np.histogram(centrality, bins=binsNormal)
                    for val, bin_ in zip(histoCentrality[0], histoCentrality[1]):
                        result.write(
                            f"Closeness, {year}, {bin_}, {val/len(graph.vs)}\n")
        return self
