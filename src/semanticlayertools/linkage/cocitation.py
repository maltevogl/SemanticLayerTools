"""Link documents by cocitation."""
import os
import time
import tempfile
import multiprocessing
from itertools import combinations
from collections import Counter

import pandas as pd
import numpy as np

num_processes = multiprocessing.cpu_count()


class Cocitations():
    """Cocitation calculations."""

    def __init__(
        self, inpath, outpath, columnName, numberProc=num_processes, debug=False
    ):
        self.inpath = inpath
        self.outpath = outpath
        self.columnName = columnName
        self.numberProc = numberProc
        self.debug = debug

    def getCombinations(self, chunk):
        """Calculate combinations."""
        res = []
        for idx, row in chunk.iterrows():
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
            sortCoCitCounts = cocitCounts.most_common()
            with open(self.outpath + infilename + '.csv', 'w') as outfile:
                for edge in sortCoCitCounts:
                    outfile.write(f"{edge[0][0]},{edge[0][1]},{edge[1]}\n")
        except:
            raise
        if self.debug == "l2":
            print(f'\tDone in {starttime - time.time()} seconds.')
        return

    def processFolder(self):
        """Calculate cocitation for all files in folder."""
        starttime = time.time()
        for file in os.listdir(self.inpath):
            try:
                self.calculateCoCitation(os.path.join(self.inpath, file))
            except:
                raise
        if self.debug is True:
            print(f'\tDone in {starttime - time.time()} seconds.')
