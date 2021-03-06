import os
import time
import re
import multiprocessing
from itertools import combinations
from collections import Counter
from typing import TypeVar

import igraph as ig
import pandas as pd
import numpy as np
from tqdm import tqdm

num_processes = multiprocessing.cpu_count()

limitRefLength = TypeVar('limitRefLength', bool, int)
debugVar = TypeVar('debugVar', bool, str)


class Cocitations():
    """Create cocitation networks.

    Calculates all combinations of all references of publications in given
    corpus file(s). Can be limited for maximal number of references to consider
    (e.g. papers with less then 200 references), to speed up creation of
    networks.

    For each corpus file, graphs are generated by the weighted cocitation tuples,
    using the Igraph package. Information on obtained clusters are written to
    '_graphMetadata.txt' files. The subgraph of the Giant component is saved in
    Pajek format with the ending '_GC.net'. The full edge data is written in
    edge-Format to a '.ncol' file.

    :param inpath: Path for input data
    :type inpath: str
    :param outpath: Path for writing output data
    :type outpath: str
    :param columnName: Column name containing the references of a publication
    :type columnName: str
    :param numberProc: Number of CPUs the package is allowed to use (default=all)
    :type numberProc: int
    :param limitRefLength: Either False or integer giving the maximum number of references a considered publication is allowed to contain
    :type limitRefLength: bool or int
    :param timerange: Time range to consider (default=(1945,2005))
    :type timerange: tuple
    :param debug: False/True or l2 to show level 2 debugging messages
    """

    def __init__(
        self, inpath, outpath, columnName,
        numberProc: int = num_processes,
        limitRefLength: limitRefLength = False,
        timerange: tuple = (1945, 2005),
        debug: debugVar = False
    ):
        self.inpath = inpath
        self.outpath = outpath
        self.columnName = columnName
        self.numberProc = numberProc
        self.limitRefLength = limitRefLength
        self.timerange = timerange
        self.debug = debug

    def getCombinations(self, chunk):
        """Calculate combinations of references in publications chunk.

        :param chunk: A chunk of the corpus dataframe
        :type chunk: `pd.Dataframe`
        :returns: A list of all reference combinations for each corpus entry
        :rtype: list
        """
        res = []
        if type(self.limitRefLength) == int:
            reflen = chunk[self.columnName].apply(
                lambda x: True if type(x) == list and len(x) <= self.limitRefLength else False
            )
            data = chunk[reflen].copy()
        else:
            data = chunk.copy()
        for idx, row in data.iterrows():
            comb = combinations(row[self.columnName], 2)
            for elem in list(comb):
                res.append((elem))
        return res

    def calculateCoCitation(self, filepath):
        """Run calculation for single input file.

        Creates three files: Metadata-File with all components information,
        Giant component network data in pajek format and full graph data in
        edgelist format.

        :param filepath: Path for input corous
        :type filepath: str
        :returns: A tuple of GC information: Number of nodes and percentage of total, Number of edges and percentage of total
        :rtype: tuple
        """
        infilename = filepath.split(os.path.sep)[-1].split('.')[0]
        starttime = time.time()
        try:
            data = pd.read_json(filepath, lines=True).dropna(subset=[self.columnName])
            chunk_size = int(data.shape[0] / self.numberProc)
            if chunk_size == 0:  # Deal with small data samples.
                chunk_size = 1
            chunks = np.array_split(data, chunk_size)
            pool = multiprocessing.Pool(processes=self.numberProc)
            cocitations = pool.map(self.getCombinations, chunks)
            cocitCounts = Counter([x for y in cocitations for x in y])
            sortCoCitCounts = [
                (x[0][0], x[0][1], x[1]) for x in cocitCounts.most_common()
            ]
            tempG = ig.Graph.TupleList(sortCoCitCounts, weights=True, vertex_name_attr='id')
            components = tempG.components()
            sortedComponents = sorted(
                [(x, len(x), len(x) * 100 / len(tempG.vs)) for x in components], key=lambda x: x[1], reverse=True
            )
            with open(os.path.join(self.outpath, infilename + '_graphMetadata.txt'), 'w') as outfile:
                outfile.write(f'Graph derived from {filepath}\nSummary:\n')
                outfile.write(tempG.summary() + '\n\nComponents (ordered by size):\n\n')
                for idx, elem in enumerate(sortedComponents):
                    gcompTemp = tempG.vs.select(elem[0]).subgraph()
                    outfile.write(
                        f"{idx}:\n\t{elem[1]} nodes ({elem[2]:.3f}% of full graph)\n\t{len(gcompTemp.es)} edges ({len(gcompTemp.es)*100/len(tempG.es):.3f}% of full graph)\n\n"
                    )
                    if idx == 0:
                        gcouttuple = (
                            elem[1],
                            elem[2],
                            len(gcompTemp.es),
                            len(gcompTemp.es) * 100 / len(tempG.es)
                        )
            giantComponent = sortedComponents[0]
            giantComponentGraph = tempG.vs.select(giantComponent[0]).subgraph()
            giantComponentGraph.write_pajek(
                os.path.join(self.outpath, infilename + '_GC.net')
            )
            with open(os.path.join(self.outpath, infilename + '.ncol'), 'w') as outfile:
                for edge in sortCoCitCounts:
                    outfile.write(f"{edge[0]} {edge[1]} {edge[2]}\n")
        except Exception:
            raise
        if self.debug == "l2":
            print(f'\tDone in {time.time() - starttime} seconds.')
        return gcouttuple

    def processFolder(self):
        """Calculate cocitation for all files in folder."""
        starttime = time.time()
        with open(
            os.path.join(
                self.outpath, 'Giant_Component_properties.csv'
            ), 'w'
        ) as gcmetafile:
            gcmetafile.write('year,nodes,nodespercent,edges,edgepercent\n')
            for file in tqdm(os.listdir(self.inpath), leave=False):
                try:
                    year = re.findall(r'\d{4}', file)[0]
                except Exception:
                    raise
                if self.timerange[0] <= int(year) <= self.timerange[1]:
                    try:
                        outtuple = self.calculateCoCitation(
                            os.path.join(self.inpath, file)
                        )
                        gcmetafile.write(
                            f'{year},{outtuple[0]},{outtuple[1]},{outtuple[2]},{outtuple[3]}\n'
                        )
                    except Exception:
                        raise
        if self.debug is True:
            print(f'\tDone in {time.time() - starttime} seconds.')
