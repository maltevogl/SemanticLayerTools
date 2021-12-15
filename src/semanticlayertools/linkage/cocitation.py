"""Link documents by cocitation."""
import os
import time
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
    """Create cocitation networks."""

    def __init__(
        self, inpath, outpath, columnName,
        numberProc: int=num_processes, limitRefLength: limitRefLength=False, debug: debugVar=False,
    ):
        self.inpath = inpath
        self.outpath = outpath
        self.columnName = columnName
        self.numberProc = numberProc
        self.limitRefLength = limitRefLength
        self.debug = debug

    def getCombinations(self, chunk):
        """Calculate combinations."""
        res = []
        if type(self.limitRefLength) == int:
            reflen = chunk[self.columnName].apply(
                lambda x: True if type(x)==list and len(x)<=self.limitRefLength else False
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
        """Do calculation for input file."""
        infilename = filepath.split(os.path.sep)[-1].split('.')[0]
        starttime = time.time()
        try:
            data = pd.read_json(filepath, lines=True).dropna(subset=[self.columnName])
            chunk_size = int(data.shape[0] / self.numberProc)
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
                [(x, len(x), len(x)*100/len(tempG.vs)) for x in components], key=lambda x: x[1], reverse=True
            )
            with open(os.path.join(self.outpath,infilename + '_graphMetadata.txt'), 'w') as outfile:
                outfile.write(f'Graph derived from {filepath}\nSummary:\n')
                outfile.write(tempG.summary() + '\n\nComponents (ordered by size):\n\n')
                for idx, elem in enumerate(sortedComponents):
                    gcompTemp = tempG.vs.select(elem[0]).subgraph()
                    outfile.write(
                        f"{idx}:\n\t{elem[1]} nodes ({elem[2]:.3f}% of full graph)\n\t{len(gcompTemp.es)} edges ({len(gcompTemp.es)*100/len(tempG.es):.3f}% of full graph)\n\n"
                    )
            giantComponent = sortedComponents[0]
            giantComponentGraph = tempG.vs.select(giantComponent[0]).subgraph()
            giantComponentGraph.write_pajek(
                os.path.join(self.outpath,infilename + '_GC.net')
            )
            with open(os.path.join(self.outpath,infilename + '.ncol'), 'w') as outfile:
                for edge in sortCoCitCounts:
                    outfile.write(f"{edge[0]} {edge[1]} {edge[2]}\n")
        except:
            raise
        if self.debug == "l2":
            print(f'\tDone in {time.time() - starttime} seconds.')
        return

    def processFolder(self):
        """Calculate cocitation for all files in folder."""
        starttime = time.time()
        for file in tqdm(os.listdir(self.inpath)):
            try:
                self.calculateCoCitation(os.path.join(self.inpath, file))
            except:
                raise
        if self.debug is True:
            print(f'\tDone in {time.time() - starttime} seconds.')
