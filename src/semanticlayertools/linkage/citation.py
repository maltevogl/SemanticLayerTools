import os
import time
import re
import math
import multiprocessing
from itertools import combinations, islice
from collections import Counter
from typing import TypeVar

import igraph as ig
import pandas as pd
import numpy as np
from tqdm import tqdm

num_processes = multiprocessing.cpu_count()

limitRefLength = TypeVar('limitRefLength', bool, int)


class Couplings():
    """Calculate different coupling networks based on citation data."""
   
    def __init__(
        self,
        inpath: str,
        outpath: str,
        pubIDColumn: str = "nodeID",
        referencesColumn: str = 'reference',
        timerange: tuple([int, int]) = (1945, 2005),
        timeWindow: int = 3,
        numberProc: int = num_processes,
        limitRefLength: bool = False,
        debug: bool = False
    ):
        self.inpath = inpath
        self.outpath = outpath
        self.pubIdCol = pubIDColumn
        self.refCol = referencesColumn
        self.timerange = timerange
        self.numberProc = numberProc
        self.limitRefLength = limitRefLength
        self.window = timeWindow
        self.debug = debug

    def _window(self, seq, n):
        """Return a sliding window (of width n) over data from the iterable.

        s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...
        """
        it = iter(seq)
        result = tuple(islice(it, n))
        if len(result) == n:
            yield result
        for elem in it:
            result = result[1:] + (elem,)
            yield result
   
    def _generateSliceData(self, sl):
        """Generate dataframe for given timeslice."""
        yearFiles = [
            file for file in os.listdir(self.inpath) if any([yearname in file for yearname in [str(year) for year in sl]])
        ]
        if self.debug is True:
            print(f'\tCreating data for yearfiles: {yearFiles}.')

        dflist = []
        for elem in yearFiles:
            df = pd.read_json(self.inpath + elem, lines=True)
            dflist.append(df)

        dfSlice = pd.concat(dflist, ignore_index=True)
        dfDataRef = dfSlice[~dfSlice[self.refCol].isna()]
        targetSet = [(row[self.pubIdCol], set(row[self.refCol])) for idx, row in dfDataRef.iterrows()]
        return dfSlice, dfDataRef, targetSet

    def _writeGraphMetadata(self, citetype, yearname, graph, writeGC=True):
        """Write metadata and giant component of given graph."""
        components = graph.components()
        sortedComponents = sorted(
            [(x, len(x), len(x) * 100 / len(graph.vs)) for x in components], key=lambda x: x[1], reverse=True
        )
        giantComponent = sortedComponents[0]
        giantComponentGraph = graph.vs.select(giantComponent[0]).subgraph()
        if writeGC is True:
            giantComponentGraph.write_pajek(
                os.path.join(self.outpath, f"{citetype}_{yearname}_GC.net")
            )
        with open(os.path.join(self.outpath, f'{citetype}_{yearname}_metadata.txt'), 'w') as outfile:
            outfile.write(f'Graph derived from {self.inpath}\nSummary:\n')
            outfile.write(graph.summary() + '\n\nComponents (ordered by size):\n\n')
            for idx, elem in enumerate(sortedComponents):
                gcompTemp = graph.vs.select(elem[0]).subgraph()
                outfile.write(
                    f"{idx}:\n\t{elem[1]} nodes ({elem[2]:.3f}% of full graph)\n\t{len(gcompTemp.es)} edges ({len(gcompTemp.es)*100/len(graph.es):.3f}% of full graph)\n\n"
                )
        return
                
    def getBibliometricCoupling(self):
        """Calculate bibliometric coupling."""
        slices = self._window(
            range(self.timerange[0], self.timerange[1] + 1, 1), self.window
        )
        overallStarttime = time.time()
        for sl in list(slices):
            yearname = sl[-1]
            if os.path.isfile(f'{self.outpath}bibcoup_{yearname}.ncol'):
                raise IOError(f"Output file at {self.outpath}bibcoup_{yearname}.ncol exists. Please move or delete.")
            if self.debug is True:
                print(f'Working on year slice {yearname}.')

            dfSlice, dfDataRef, targetSet = self._generateSliceData(sl)

            comLen = math.factorial(len(targetSet))/(math.factorial(2)*(math.factorial(len(targetSet) - 2)))
            if self.debug is True:
                print(f"\tWill have to calculate {comLen} combinations for {dfDataRef.shape[0]} entries with references ({dfSlice.shape[0]} entries in total).\n\tEstimated runtime {((comLen)/1838000)/3600:.2f} hours.")

            starttime = time.time()
            with open(f'{self.outpath}bibcoup_{yearname}.ncol', 'w+') as outfile:
                for tup in tqdm(combinations(targetSet, 2), leave=False):
                    overlap = tup[0][1].intersection(tup[1][1])
                    if overlap:
                        if len(overlap) > 1:
                            outfile.write(
                                f"{tup[0][0]} {tup[1][0]} {len(overlap)}\n"
                            )
            tempG = ig.Graph.Read_Ncol(
                f'{self.outpath}bibcoup_{yearname}.ncol',
                names=True,
                weights=True,
                directed=False
            )
            tempG.vs['id'] = tempG.vs['name'] 
            self._writeGraphMetadata("bibcoup", yearname, tempG)
            print(f"Done in {(time.time()-starttime)/3600:.2f} hours.")
        print(f"Finished all slices in {(time.time()-overallStarttime)/3600:.2f} hours.")

    def _getCombinations(self, chunk):
        """Calculate combinations of references in publications chunk.

        :param chunk: A chunk of the corpus dataframe
        :type chunk: `pd.Dataframe`
        :returns: A list of all reference combinations for each corpus entry
        :rtype: list
        """
        res = []
        chunk = chunk.dropna(subset=[self.refCol])
        if type(self.limitRefLength) == int:
            reflen = chunk[self.refCol].apply(
                lambda x: type(x) == list and len(x) <= self.limitRefLength
            )
            data = chunk[reflen].copy()
        else:
            data = chunk.copy()
        for idx, row in data.iterrows():
            comb = combinations(row[self.refCol], 2)
            res.extend(list(comb))
        return res

    def getCocitationCoupling(self):
        """Calculate cocitation coupling.

        Creates three files: Metadata-File with all components information,
        Giant component network data in pajek format and full graph data in
        edgelist format.

        The input dataframe is split in chunks depending on the available cpu processes. All possible combinations for all 
        elements of the reference column are calculated. The resulting values are counted to define the weight of two 
        papers being cocited in the source dataframe.

        :returns: A tuple of GC information: Number of nodes and percentage of total, Number of edges and percentage of total
        :rtype: tuple
        """
        slices = self._window(
            range(self.timerange[0], self.timerange[1] + 1, 1), self.window
        )
        overallStarttime = time.time()
        for sl in list(slices):
            yearname = sl[-1]
            if os.path.isfile(f'{self.outpath}cocite_{yearname}_GC.net'):
                raise IOError(f"Output file at {self.outpath}cocite_{yearname}_GC.net exists. Please move or delete.")
            if self.debug is True:
                print(f'Working on year slice {yearname}.')

            dfSlice, dfDataRef, targetSet = self._generateSliceData(sl)
            data = dfSlice.dropna(subset=[self.refCol])
            chunk_size = int(data.shape[0] / self.numberProc)
            if chunk_size == 0:
                chunk_size = 1
            chunks = np.array_split(data, chunk_size)
            pool = multiprocessing.Pool(processes=self.numberProc)
            cocitations = pool.map(self._getCombinations, chunks)
            cocitCounts = Counter([x for y in cocitations for x in y])
            sortCoCitCounts = [
                (x[0][0], x[0][1], x[1]) for x in cocitCounts.most_common()
            ]
            tempG = ig.Graph.TupleList(
                sortCoCitCounts,
                weights=True,
                vertex_name_attr='id'
            )
            self._writeGraphMetadata("cocite", yearname, tempG)
            with open(os.path.join(self.outpath, f"cocite_{yearname}.ncol"), 'w') as outfile:
                for edge in sortCoCitCounts:
                    outfile.write(f"{edge[0]} {edge[1]} {edge[2]}\n")
        if self.debug is True:
            print(f'\tDone in {time.time() - overallStarttime} seconds.')

    def getCitationCoupling(self):
        """Calculate direct citation coupling."""
        slices = self._window(
            range(self.timerange[0], self.timerange[1] + 1, 1), self.window
        )
        overallStarttime = time.time()
        for sl in list(slices):
            yearname = sl[-1]
            if os.path.isfile(f'{self.outpath}citecoup_{yearname}.ncol'):
                raise IOError(f"Output file at {self.outpath}citecoup_{yearname}.ncol exists. Please move or delete.")
            if self.debug is True:
                print(f'Working on year slice {yearname}.')

            dfSlice, dfDataRef, targetSet = self._generateSliceData(sl)

            sourceSet = set(dfSlice[self.pubIdCol].unique())
            directedCitationEdges = []
            for target in tqdm(targetSet):
                overlap = sourceSet.intersection(target[1])
                if overlap:
                    overlapList = list(overlap)
                    for elem in overlapList:
                        directedCitationEdges.append(
                            (target[0], elem, 1)
                        )
            tempG = ig.Graph.TupleList(
                directedCitationEdges,
                directed=True,
                weights=True,
                vertex_name_attr='id'
            )
            self._writeGraphMetadata("citecoup", yearname, tempG, writeGC=False)
            with open(os.path.join(self.outpath, f"citecoup_{yearname}.ncol"), 'w') as outfile:
                for edge in directedCitationEdges:
                    outfile.write(f"{edge[0]} {edge[1]} {edge[2]}\n")
        if self.debug is True:
            print(f'\tDone in {time.time() - overallStarttime} seconds.')



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
    :param debug: bool
    """

    def __init__(
        self, inpath: str, outpath: str, columnName: str,
        numberProc: int = num_processes,
        limitRefLength: limitRefLength = False,
        timerange: tuple([int,int]) = (1945, 2005),
        debug: bool = False
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
        chunk = chunk.dropna(subset=[self.columnName])
        if type(self.limitRefLength) == int:
            reflen = chunk[self.columnName].apply(
                lambda x: type(x) == list and len(x) <= self.limitRefLength
            )
            data = chunk[reflen].copy()
        else:
            data = chunk.copy()
        for idx, row in data.iterrows():
            comb = combinations(row[self.columnName], 2)
            res.extend(list(comb))
            #for elem in list(comb):
            #    res.append((elem))
        return res

    def calculateCoCitation(self, filepath):
        """Run calculation for single input file.

        Creates three files: Metadata-File with all components information,
        Giant component network data in pajek format and full graph data in
        edgelist format.

        The input dataframe is split in chunks depending on the available cpu processes. All possible combinations for all 
        elements of the reference column are calculated. The resulting values are counted to define the weight of two 
        papers being cocited in the source dataframe.

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
            cocitCounts = Counter([x for y in cocitations for x in y])  # This defines the weight of the cocitation edge.
            sortCoCitCounts = [
                (x[0][0], x[0][1], x[1]) for x in cocitCounts.most_common()
            ]
            tempG = ig.Graph.TupleList(sortCoCitCounts, weights=True, vertex_name_attr='id')  # Igraph is used to generate the basic graph from the weighted tuples.
            components = tempG.components() # Disconnected components are identified, i.e. subgraphs without edges to other subgraphs
            sortedComponents = sorted(
                [(x, len(x), len(x) * 100 / len(tempG.vs)) for x in components], key=lambda x: x[1], reverse=True
            ) # Sorting the result in reverse order yields the first element as the giant component of the network.
            with open(os.path.join(self.outpath, infilename + '_graphMetadata.txt'), 'w') as outfile: # To judge the quality of the giant component in relation to the full graph, reports are created.
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
            giantComponentGraph = tempG.vs.select(giantComponent[0]).subgraph() # Two different network formats are written: The giant component as a Pajek file, and the full graph in NCOL format. 
            giantComponentGraph.write_pajek(
                os.path.join(self.outpath, infilename + '_GC.net')
            )
            with open(os.path.join(self.outpath, infilename + '.ncol'), 'w') as outfile:
                for edge in sortCoCitCounts:
                    outfile.write(f"{edge[0]} {edge[1]} {edge[2]}\n")
        except IndexError:
            print(filepath)
            return (0, 0, 0, 0)
        if self.debug is True:
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
            for file in tqdm([x for x in os.listdir(self.inpath) if x.endswith('.json')], leave=False):
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
                        print(file)
                        raise
        if self.debug is True:
            print(f'\tDone in {time.time() - starttime} seconds.')
