import os
import time
import math
from operator import itemgetter
from collections import Counter
from itertools import islice, combinations, groupby
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np
import pandas as pd


class CalculateKDL():
    """Calculates KDL scores for time slices.

    .. seealso::

        Stefania Degaetano-Ortlieb and Elke Teich. 2017. 
        Modeling intra-textual variation with entropy and surprisal: topical vs. stylistic patterns.
        In Proceedings of the Joint SIGHUM Workshop on Computational Linguistics for Cultural Heritage, Social Sciences, Humanities and Literature, pages 68–77,
        Vancouver, Canada. Association for Computational Linguistics.

    """

    def __init__(
        self,
        targetData,
        compareData,
        yearColumnTarget: str = 'year',
        yearColumnCompare: str = 'year',
        tokenColumnTarget: str = 'tokens',
        tokenColumnCompare: str = 'tokens',
        debug: bool = False
    ):

        self.baseDF = compareData
        self.targetDF = targetData
        self.yearColTarget = yearColumnTarget
        self.yearColCompare = yearColumnCompare
        self.tokenColumnTarget = tokenColumnTarget
        self.tokenColumnCompare = tokenColumnCompare
        self.ngramData = []
        self.minNgramNr = 1
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

    def _createSlices(self, windowSize):
        """Create slices of dataframe."""
        slices = []
        years = sorted(self.targetDF[self.yearColTarget].unique())
        for x in self._window(years, windowSize):
            slices.append(x)
        return slices

    def _calculateDistributions(self, source, dataframe, timesliceNr, timeslice, specialChar):
        unigram = []
        fourgram = []
        if source == "target":
            yearCol = self.yearColTarget
            tokenCol = self.tokenColumnTarget
        elif source == "compare":
            yearCol = self.yearColCompare
            tokenCol = self.tokenColumnCompare
        df = dataframe[dataframe[yearCol].isin(timeslice)]
        for _, row in df.iterrows():
            for elem in row[tokenCol]:
                elemLen = len(elem.split(specialChar))
                if elemLen == 1:
                    unigram.append(elem)
                elif elemLen == 4:
                    fourgram.append(elem.split(specialChar))
        unigramCounts = dict(Counter(unigram).most_common())
        fourgram.sort(key=lambda x: x[3])
        sorted4grams = [
            [specialChar.join(x) for x in list(group)] for key, group in groupby(fourgram, itemgetter(3))
        ]
        return (timesliceNr, source, timeslice[-1], unigramCounts, sorted4grams)
    

    def getNgramPatterns(self, windowSize=3, specialChar="#"):
        """Create dictionaries of occuring ngrams.
        
        :param specialChar: Special character used to delimit tokens in ngrams (default=#)
        :type specialChar: str
        """
        starttime = time.time()
        self.ngramData = []
        print(f"Got data for {self.baseDF[self.yearColCompare].min()} to {self.baseDF[self.yearColCompare].max()}, starting calculations.")
        for idx, timeslice in tqdm(enumerate(self._createSlices(windowSize)), leave=False):
            sliceName = timeslice[-1]
            if self.debug is True:
                print(f"\tStart slice {sliceName}.")
            self.ngramData.append(
                self._calculateDistributions('target', self.targetDF, idx, timeslice, specialChar)
            )
            self.ngramData.append(
                self._calculateDistributions('compare', self.baseDF, idx, timeslice, specialChar)
            )
        if self.debug is True:
            print(f"Done in  {time.time() - starttime} seconds.")
        return

            
    def getKDLRelations(self, windowSize: int = 3, minNgramNr: int = 5, specialChar: str = "#"):
        """Calculate KDL relations.

        :param specialChar: Special character used to delimit tokens in ngrams (default=#)
        :type specialChar: str
        """
        self.kdlRelations = []
        distributions = pd.DataFrame(self.ngramData, columns=['sliceNr', 'dataType', 'sliceName', 'unigrams', 'fourgrams'])
        for idx in distributions['sliceNr'].unique():
            targetData = distributions.query('dataType == "target" and sliceNr == @idx')
            sorted4gram = targetData['fourgrams'].iloc[0]
            sorted4gramDict = {elem[0].split(specialChar)[3]:elem for elem in sorted4gram}
            unigramCounts = targetData['unigrams'].iloc[0]
            year1 = targetData['sliceName'].iloc[0]
            compareDataPost = distributions.query(f'dataType == "compare" and (sliceNr >= {idx + windowSize} or sliceNr <={idx - windowSize})')
            for _, row in compareDataPost.iterrows():
                kdlVals = []
                idx2 = row['sliceNr']
                year2 = row['sliceName']
                sorted4gram2 = row['fourgrams']
                sorted4gramDict2 = {elem[0].split(specialChar)[3]:elem for elem in sorted4gram2}
                unigramCounts2 = row['unigrams']
                for key, elem1 in sorted4gramDict.items():   
                    if unigramCounts[key] < minNgramNr:
                        continue
                    if not key in sorted4gramDict2.keys():
                        continue
                        
                    elem2 = sorted4gramDict2[key]
                    basisLen1 = len(set(elem1))
                    basisLen2 = len(set(elem2))
                        
                    counts1 = dict(Counter(elem1).most_common())
                    counts2 = dict(Counter(elem2).most_common())
                        
                    probList = []
                    for key, val in counts1.items():
                        if key in counts2.keys():
                            probList.append(
                                val/basisLen1 * math.log( (val * basisLen2)/(basisLen1 * counts2[key]), 2)
                            )
                    kdl = sum(probList)
                    kdlVals.append(kdl)
                
                self.kdlRelations.append(
                    (idx, idx2, year1, year2, sum(kdlVals))
                )
        return self.kdlRelations
