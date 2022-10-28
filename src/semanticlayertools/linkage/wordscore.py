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


class CalculateSurprise():
    """Calculates surprise scores for documents.

    The source dataframe is expected to contain pre-calculated ngrams (tokens) for each document 
    in the form of lists of 1-grams, joined by a special character (default is "#" (hash)). For surprise
    calculation of e.g. 1- and 2-grams the precalculated n-grams need to contain at least 5-grams, to evaluate
    the surprise context of 1- and 2-grams, see reference for details. The main routine of this class is run().

    :param sourceDataframe: Dataframe containing the basic corpus
    :type sourceDataframe: class:`pandas.DataFrame`
    :param pubIDColumn: Column name to use for publication identification (assumend to be unique)
    :type pubIDColumn: str
    :param yearColumn: Column name for temporal ordering publications, used during writing the scoring files
    :type yearColumn: str
    :param tokenColumn: Column name for tokens
    :type tokenColumn: str

    .. seealso::

        Stefania Degaetano-Ortlieb and Elke Teich. 2017. 
        Modeling intra-textual variation with entropy and surprisal: topical vs. stylistic patterns.
        In Proceedings of the Joint SIGHUM Workshop on Computational Linguistics for Cultural Heritage, Social Sciences, Humanities and Literature, pages 68–77,
        Vancouver, Canada. Association for Computational Linguistics.

    """

    def __init__(
        self,
        sourceDataframe,
        pubIDColumn: str = "pubID",
        yearColumn: str = 'year',
        tokenColumn: str = 'tokens',
        debug: bool = False
    ):

        self.baseDF = sourceDataframe
        self.pubIDCol = pubIDColumn
        self.yearCol = yearColumn
        self.tokenColumn = tokenColumn
        self.currentyear = ''
        self.surprise = {}
        self.OneGramCounts = {}
        self.TwoGramCounts = {}
        self.sorted4grams = {}
        self.sorted5grams = {}
        self.minNgramNr = 1
        self.ngramDocTfidf = {}
        self.outputDict = {}
        self.counts = {}
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

    def _createSlices(self, windowsize):
        """Create slices of dataframe."""
        slices = []
        years = sorted(self.baseDF[self.yearCol].unique())
        for x in self._window(years, windowsize):
            slices.append(x)
        return slices

    def getTfiDF(self, year):
        """Calculate augmented term-frequency inverse document frequency."""
        ngramNDocs = {}
        self.ngramDocTfidf[year] = []
        nDocs = len(self.outputDict[year].keys())
        newval = []
        for key, val in self.outputDict[year].items():
            for elem in val:
                newval.append((key, elem[0], elem[1]))
        tempscore = pd.DataFrame(newval)

        for ngram, g0 in tempscore.groupby(1):
            ngramNDocs.update({ngram: len(g0[0].unique())})

        for doi in tempscore[0].unique():
            ngramDict = tempscore[tempscore[0] == doi][1].value_counts().to_dict()
            maxVal = max(ngramDict.values())
            for key, val in ngramDict.items():
                self.ngramDocTfidf[year].append(
                    (doi, key, val, (0.5 + 0.5 * (val / maxVal)) * np.log(nDocs / ngramNDocs[key]))
                )
        return self.ngramDocTfidf

    def getNgramPatterns(self, year, dataframe, ngramLimit=2, specialChar="#"):
        """Create dictionaries of occuring ngrams.
        
        :param year: Current year for calculations
        :type year: int
        :param dataframe: Current slice of main dataframe
        :type dataframe: class:`pandas.DataFrame`
        :param ngramLimit: Maximal ngram to consider for surprise calculation (default=2)
        :type ngramLimit: int
        :param specialChar: Special character used to delimit tokens in ngrams (default=#)
        :type specialChar: str
        """
        self.counts[year] = {}
        self.outputDict[year] = {}
        allNGrams = {}
        for _, row in tqdm(dataframe.iterrows(), leave=False):
            self.outputDict[year].update(
                {
                    row[self.pubIDCol]: [
                        x for x in row[self.tokenColumn] if len(x.split(specialChar)) <= ngramLimit
                    ]
                }
            )
            for elem in row[self.tokenColumn]:
                keyVal = len(elem.split(specialChar))
                try:
                    val = allNGrams[keyVal]
                except KeyError:
                    val = []
                if keyVal > 2:
                    val.append(elem.split(specialChar))
                    allNGrams.update({keyVal: val})
                else:
                    val.append(elem)
                    allNGrams.update({keyVal: val})

        self.OneGramCounts = dict(Counter(allNGrams[1]).most_common())
        self.TwoGramCounts = dict(Counter(allNGrams[2]).most_common())

        all4grams = allNGrams[4]
        all5grams = allNGrams[5]
        all4grams.sort(key=lambda x: x[3])
        all5grams.sort(key=lambda x: (x[3], x[4]))

        self.sorted4grams = [list(group) for key, group in groupby(all4grams, itemgetter(3))]
        self.sorted5grams = [list(group) for key, group in groupby(all5grams, itemgetter(3, 4))]

    def getSurprise(self, target: list, ngramNr: int = 1, specialChar: str = "#"):
        """Calculate surprise score.

        The surprise for a 1- or 2-gram is calculated based on the group of 4- or 5-grams which 
        contain the target in the last or two last positions (e.g. experiment and the#last#experiment),
        and defined as the sum over the base two logarithm of the probabilities for 4- or 5-grams.
        The probability, e.g for a given 4-gram, is the number  of realizations of that 4-gram devided 
        by the number of possible 4-grams.

        :param target: Target list of tuples to use for surprise calculation.
        :type target: list
        :param ngramNr: ngram length to use for calculation (1 or 2)
        :type ngramNr: int
        :param minNgramNr: Minimal number of occurance of a 1- or 2-gram in the corpus to consider calculations (default=5)
        :type minNgramNr: int
        :param specialChar: Special character used to delimit tokens in ngrams (default=#)
        :type specialChar: str
        """
        if ngramNr == 1:
            tokName = target[0][3]
            if self.OneGramCounts[tokName] < self.minNgramNr:
                return {tokName: 0}
        elif ngramNr == 2:
            tokList = [target[0][3], target[0][4]]
            tokName = specialChar.join(tokList)
            if self.TwoGramCounts[tokName] < self.minNgramNr:
                return {tokName: 0}
        joinedTarget = [specialChar.join(x) for x in target]
        basisLen = len(set(joinedTarget))
        counts = dict(Counter(joinedTarget).most_common())
        probList = []
        for key, val in counts.items():
            probList.append(
                - math.log(val / basisLen, 2)
            )
        surpriseVal = 1 / basisLen * sum(probList)
        return {tokName: surpriseVal}

    def _calcBatch1(self, batch):
        res = []
        for elem in batch:
            res.append(self.getSurprise(elem, ngramNr=1))
        return res

    def _calcBatch2(self, batch):
        res = []
        for elem in batch:
            res.append(self.getSurprise(elem, ngramNr=2))
        return res

    def run(
        self, windowsize: int = 3, write: bool = False, outpath: str = './',
        recreate: bool = False, maxNgram: int = 2, minNgramNr: int = 5, limitCPUs: bool = True
    ):
        """Calculate surprise for all documents.
        
        Base corpus is sliced with a rolling window (see windowsize).
        For each slice the ngram distributions are created and surpise
        and tfidf scores calculated. The results are returned or saved.

        :param minNgramNr: Minimal number of occurance of a 1- or 2-gram in the corpus to consider calculations (default=5)
        :type minNgramNr: int
        """
        starttime = time.time()
        print(f"Got data for {self.baseDF[self.yearCol].min()} to {self.baseDF[self.yearCol].max()}, starting calculations.")
        for timeslice in self._createSlices(windowsize):
            dataframe = self.baseDF[self.baseDF[self.yearCol].isin(timeslice)]
            year = timeslice[-1]
            self.currentyear = year
            self.surprise.update({year: {}})
            if write is True:
                filePath = f'{outpath}{str(year)}_surprise.tsv'
                if os.path.isfile(filePath):
                    if recreate is False:
                        raise IOError(
                            f'File at {filePath} exists. Set recreate = True to overwrite.'
                        )
            if self.debug is True:
                print(f"Creating ngram counts for {year}.")
            self.getNgramPatterns(
                year=year,
                dataframe=dataframe,
                ngramLimit=maxNgram
            )
            if self.debug is True:
                print(
                    f'\tFound {len(self.OneGramCounts.keys())} unique 1-grams and {len(self.TwoGramCounts.keys())} unique 2-grams.')
            # Setup multiprocessing
            if limitCPUs is True:
                ncores = int(cpu_count() * 1 / 4)
            else:
                ncores = cpu_count() - 2
            if self.debug is True:
                print(f'\tStarting calculation of surprise for {year}.')
            pool = Pool(ncores)
            # Calculate 1-gram surprises
            chunk_size = int(len(self.sorted4grams) / ncores)
            if self.debug is True:
                print(f"\tStarting 1-gram surprise.\n\tCalculated chunk size is {chunk_size}.")
            batches = [
                list(self.sorted4grams)[i:i + chunk_size] for i in range(0, len(self.sorted4grams), chunk_size)
            ]
            ncoresResults = pool.map(self._calcBatch1, batches)
            results = [x for y in ncoresResults for x in y]
            for elem in results:
                self.surprise[year].update(elem)
            # Calculate 2-gram surprises
            chunk_size = int(len(self.sorted5grams) / ncores)
            if self.debug is True:
                print(f"\tStarting 2-gram surprise.\n\tCalculated chunk size is {chunk_size}.")
            batches = [
                list(self.sorted5grams)[i:i + chunk_size] for i in range(0, len(self.sorted5grams), chunk_size)
            ]
            ncoresResults = pool.map(self._calcBatch2, batches)
            results2 = [x for y in ncoresResults for x in y]
            for elem in results2:
                self.surprise[year].update(elem)
            # Link to publications
            for key, val in self.outputDict[year].items():
                tmpList = []
                for elem in val:
                    if elem in self.surprise[year].keys():
                        try:
                            tmpList.append([elem, self.surprise[year][elem]])
                        except Exception:
                            print(elem)
                            raise
                self.outputDict[year].update({key: tmpList})
            # Calculate TFIDF
            if self.debug is True:
                print("Start tfidf calculations.")
            self.getTfiDF(year)
            # Prepare output data.
            outputList = []
            for pub in dataframe[self.pubIDCol].unique():
                for elem in self.outputDict[year][pub]:
                    outputList.append((pub, elem[0], elem[1]))
            dfTf = pd.DataFrame(self.ngramDocTfidf[year], columns=['doc', 'ngram', 'count', 'tfidf'])
            dfSc = pd.DataFrame(outputList, columns=['doc', 'ngram', 'score'])
            dfM = dfTf.merge(dfSc, on=['doc', 'ngram'], how='outer').drop_duplicates()
            if write is True:
                if recreate is True:
                    try:
                        os.remove(filePath)
                    except FileNotFoundError:
                        pass
                dfM.to_csv(filePath, sep='\t', index=False)
                if self.debug is True:
                    print(f'\tDone creating surprise scores for {year}, written to {filePath}.')
        print(f'Done in {(time.time() - starttime)/60:.2f} minutes.')
        if write is True:
            return
        return self.ngramDocTfidf, self.surprise, self.outputDict


class CalculateScores():
    """Calculates ngram scores for documents.

    All texts of the corpus are tokenized and POS tags are generated.
    A global dictionary of counts of different ngrams is build in `counts`.
    The ngram relations of every text are listed in `outputDict`.

    Scoring is based on counts of occurances of different words left and right of each single
    token in each ngram, weighted by ngram size, for details see reference. #FIXME

    :param sourceDataframe: Dataframe containing the basic corpus
    :type sourceDataframe: class:`pandas.DataFrame`
    :param pubIDColumn: Column name to use for publication identification (assumend to be unique)
    :type pubIDColumn: str
    :param yearColumn: Column name for temporal ordering publications, used during writing the scoring files
    :type yearColumn: str
    :param ngramsize: Maximum of considered ngrams (default: 5-gram)
    :type ngramsize: int

    .. seealso::
        Abe H., Tsumoto S. (2011).
        Evaluating a Temporal Pattern Detection Method for Finding Research Keys in Bibliographical Data.
        In: Peters J.F. et al. (eds) Transactions on Rough Sets XIV. Lecture Notes in Computer Science, vol 6600.
        Springer, Berlin, Heidelberg. 10.1007/978-3-642-21563-6_1


    """

    def __init__(
        self,
        sourceDataframe,
        pubIDColumn: str = "pubID",
        yearColumn: str = 'year',
        tokenColumn: str = 'tokens',
        debug: bool = False
    ):

        self.baseDF = sourceDataframe
        self.pubIDCol = pubIDColumn
        self.yearCol = yearColumn
        self.tokenColumn = tokenColumn
        self.currentyear = ''
        self.allNGrams = {}
        self.scores = {}
        self.ngramDocTfidf = {}
        self.outputDict = {}
        self.counts = {}
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

    def _createSlices(self, windowsize):
        slices = []
        years = sorted(self.baseDF[self.yearCol].unique())
        for x in self._window(years, windowsize):
            slices.append(x)
        return slices

    def getTfiDF(self, year):
        ngramNDocs = {}
        self.ngramDocTfidf[year] = []
        nDocs = len(self.outputDict[year].keys())
        newval = []
        for key, val in self.outputDict[year].items():
            for elem in val:
                newval.append((key, elem[0], elem[1]))
        tempscore = pd.DataFrame(newval)

        for ngram, g0 in tempscore.groupby(1):
            ngramNDocs.update({ngram: len(g0[0].unique())})

        for doi in tqdm(tempscore[0].unique(), leave=False):
            ngramDict = tempscore[tempscore[0] == doi][1].value_counts().to_dict()
            maxVal = max(ngramDict.values())
            for key, val in ngramDict.items():
                self.ngramDocTfidf[year].append(
                    (doi, key, (0.5 + 0.5 * (val / maxVal)) * np.log(nDocs / ngramNDocs[key]))
                )
        return self.ngramDocTfidf

    def getTermPatterns(self, year, dataframe, specialChar="#"):
        """Create dictionaries of occuring ngrams."""
        self.counts[year] = {}
        self.outputDict[year] = {}
        self.allNGrams = {}
        for _, row in tqdm(dataframe.iterrows(), leave=False):
            self.outputDict[year].update(
                {row[self.pubIDCol]: [x for x in row[self.tokenColumn]]}
            )
            for elem in row[self.tokenColumn]:
                lenElem = len(elem.split(specialChar))
                try:
                    val = self.allNGrams[lenElem]
                except KeyError:
                    val = []
                val.append(elem)
                self.allNGrams.update({lenElem: val})
        for key, value in self.allNGrams.items():
            self.counts[year][key] = dict(Counter([x for x in value]))

    def getScore(self, target, specialChar="#"):
        """Calculate ngram score."""
        valueList = []
        for _, subgram in enumerate(target.split(specialChar)):
            contains = [x for x in self.counts[self.currentyear][2].keys() if subgram in x]
            rvalue = len(set(x for x in contains if x[0] == subgram))
            lvalue = len(set(x for x in contains if x[1] == subgram))
            valueList.append((lvalue + 1.0) * (rvalue + 1.0))
        factors = np.prod(valueList, dtype=np.float64)
        return {
            target: 1.0 / self.counts[self.currentyear][len(target)][target] * (factors) ** (1.0 / (2.0 * len(target)))
        }

    def _calcBatch(self, batch):
        res = []
        for elem in tqdm(batch, leave=False):
            res.append(self.getScore(elem))
        return res

    def run(
        self, windowsize: int = 3, write: bool = False, outpath: str = './',
        recreate: bool = False, tokenMinCount: int = 5, limitCPUs: bool = True
    ):
        """Get score for all documents."""
        starttime = time.time()
        print(f"Got data for {self.baseDF[self.yearCol].min()} to {self.baseDF[self.yearCol].max()}, starting calculations.")
        for timeslice in self._createSlices(windowsize):
            dataframe = self.baseDF[self.baseDF[self.yearCol].isin(timeslice)]
            year = timeslice[-1]
            self.currentyear = year
            self.scores.update({year: {}})
            if write is True:
                filePath = f'{outpath}{str(year)}_score.tsv'
                if os.path.isfile(filePath):
                    if recreate is False:
                        raise IOError(
                            f'File at {filePath} exists. Set recreate = True to overwrite.'
                        )
            if self.debug is True:
                print(f"Creating ngram counts for {year}.")
            self.getTermPatterns(
                year=year,
                dataframe=dataframe,
            )
            uniqueNGrams = []
            for key in self.counts[year].keys():
                tempDict = {x: y for x, y in self.counts[year][key].items() if y >= tokenMinCount}
                self.counts[year].update(
                    {key: tempDict}
                )
                uniqueNGrams.extend(list(tempDict.keys()))
            if self.debug is True:
                print(
                    f'\tFound {len(uniqueNGrams)} unique n-grams with at least {tokenMinCount} occurances.')
            if limitCPUs is True:
                ncores = int(cpu_count() * 1 / 4)
            else:
                ncores = cpu_count() - 2
            if self.debug is True:
                print(f'\tStarting calculation of scores for {year}.')
            pool = Pool(ncores)
            chunk_size = int(len(uniqueNGrams) / ncores)
            if self.debug is True:
                print(f"\tCalculated chunk size is {chunk_size}.")
            if chunk_size > 0:
                batches = [
                    list(uniqueNGrams)[i:i + chunk_size] for i in range(0, len(uniqueNGrams), chunk_size)
                ]
                ncoresResults = pool.map(self._calcBatch, batches)
                results = [x for y in ncoresResults for x in y]
            else:
                results = pool.map(self._calcBatch, uniqueNGrams)
            for elem in results:
                self.scores[year].update(elem)
            for key, val in self.outputDict[year].items():
                tmpList = []
                for elem in val:
                    if elem in uniqueNGrams:
                        try:
                            tmpList.append([elem, self.scores[year][elem]])
                        except TypeError:
                            print(elem)
                            raise
                self.outputDict[year].update({key: tmpList})
            if self.debug is True:
                print("Start tfidf calculations.")
            self.getTfiDF(year)
            # Prepare output data.
            outputList = []
            for pub in dataframe[self.pubIDCol].unique():
                for elem in self.outputDict[year][pub]:
                    outputList.append((pub, elem[0], elem[1]))
            dfTf = pd.DataFrame(self.ngramDocTfidf[year], columns=['doc', 'ngram', 'count', 'tfidf'])
            dfSc = pd.DataFrame(outputList, columns=['doc', 'ngram', 'score'])
            dfM = dfTf.merge(dfSc, on=['doc', 'ngram'], how='outer').drop_duplicates()
            if write is True:
                if recreate is True:
                    try:
                        os.remove(filePath)
                    except FileNotFoundError:
                        pass
                dfM.to_csv(filePath, sep="\t", index=False)
                if self.debug is True:
                    print(f'\tDone creating scores for {year}, written to {filePath}.')
        print(f'Done in {(time.time() - starttime)/60:.2f} minutes.')
        if write is True:
            return
        return self.ngramDocTfidf, self.scores, self.outputDict


class LinksOverTime():
    """Create multilayer pajek files for corpus.

    To keep track of nodes over time, we need a global register of node names.
    This class takes care of this, by adding new keys of authors, papers or
    ngrams to the register. Central routine is "writeLinks".

    :param dataframe: Source dataframe containing metadata of texts (authors, publicationID and year)
    :type dataframe: class:`pandas.DataFrame`
    :param authorColumn: Column name for author information, author names are assumed to be separated by semikolon
    :type authorColumn: str
    :param pubIDColumn: Column name to identify publications
    :type pubIDColumn: str
    :param yearColumn: Column name with year information (year encoded as integer)
    :type yearColumn: str
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        authorColumn: str = 'authors',
        pubIDColumn: str = "pubID",
        yearColumn: str = 'year',
        debug: bool = False
    ):
        self.dataframe = dataframe
        self.authorCol = authorColumn
        self.pubIDCol = pubIDColumn
        self.yearColumn = yearColumn
        self.nodeMap = {}
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

    def _createSlices(self, windowsize):
        slices = []
        years = sorted(self.dataframe[self.yearColumn].unique())
        for x in self._window(years, windowsize):
            slices.append(x)
        return slices

    def createNodeRegister(self, scorePath: str, scoreLimit: float, scoreType: str = 'score'):
        """Create multilayer node register for all time slices."""
        starttime = time.time()
        if scoreType == "score":
            scores = [x for x in os.listdir(scorePath) if x.endswith('_score.tsv')]
        elif scoreType == "surprise":
            scores = [x for x in os.listdir(scorePath) if x.endswith('_surprise.tsv')]
        ngrams = [pd.read_csv(
            scorePath + score,
            sep='\t'
        ) for score in scores]
        ngramdataframe = pd.concat(ngrams)
        ngramdataframe = ngramdataframe[ngramdataframe['score'] > scoreLimit]

        authorList = [
            x for y in [
                z.split(';') for z in self.dataframe[self.authorCol].values
            ] for x in y
        ]
        authors = [x for x in set(authorList) if x]
        pubs = self.dataframe[self.pubIDCol].fillna('None').unique()
        ngrams = ngramdataframe['ngram'].unique()
        if self.debug is True:
            print(f"Got {len(authors)} authors, {len(pubs)} papers and {len(ngrams)} unique ngrams.\n\tBuilding node map...")
        for authorval in authors:
            if not self.nodeMap.values():
                self.nodeMap.update({authorval: 1})
            else:
                if authorval not in self.nodeMap.keys():
                    self.nodeMap.update(
                        {authorval: max(self.nodeMap.values()) + 1}
                    )
        for pubval in pubs:
            if pubval not in self.nodeMap.keys():
                self.nodeMap.update({pubval: max(self.nodeMap.values()) + 1})
        ngramdict = {
            y: x for x, y in enumerate(list(ngrams), start=max(self.nodeMap.values()) + 1)
        }
        self.nodeMap.update(ngramdict)
        print(f"Done building node register in {(time.time() - starttime)/60:.2f} minutes.")
        return

    def writeLinks(
        self, sl, scorePath: str, scoreLimit: float, normalize: bool,
        coauthorValue: float = 0.0, authorValue: float = 0.0,
        outpath: str = './', recreate: bool = False
    ):
        """Write multilayer links to file in Pajek format.

        For ngrams with score above the limit, the corresponding tfidf value is extracted. 
        If no preset value is given, links between coauthors and authors and publications 
        are set to the median of the score values of the time slice.
        The created graphs are saved as pajek files, containing the 
        information on node names and layers (1: authors, 2: publications, 3: ngrams). 

        :param sl: Year slice of calculation
        :type sl: list
        :param scorePath: Path to score files.
        :type scorePath: str
        :param scoreLimit: Lower limit of scores to consider for network creation
        :type scoreLimit: float
        :param normalize: Normalize the scores (True/False)
        :type normalize: bool
        :param coauthorValue: Set manual value for coauthor weight (default: Median of score weight) 
        :type coauthorvalue: float
        :param authorValue: Set manual value for author to publication weight (default: Median of score weight)
        :type authorValue: float
        :param outPath: Path to write multilayer pajek files (default = './')
        :type outPath: str
        :param recreate: Rewrite existing files (default = False)
        :type recreate: bool
        """
        slicedataframe = self.dataframe[self.dataframe[self.yearColumn].isin(sl)]
        filePath = outpath + 'multilayerPajek_{0}.net'.format(sl[-1])

        if os.path.isfile(filePath):
            if recreate is False:
                raise IOError(
                    f'File at {filePath} exists. Set recreate = True to rewrite file.'
                )
            if recreate is True:
                os.remove(filePath)

        ngramdataframe = pd.read_csv(
            scorePath,
            sep='\t'
        )
        # Use results from tfidf and scoring
        # Step 1: Limit ngrams using the surprise values
        ngramdataframe = ngramdataframe[ngramdataframe['score'] > scoreLimit]

        if normalize is True:
            maxval = ngramdataframe["tfidf"].max()
            normVal = ngramdataframe["tfidf"] / maxval
            ngramdataframe["tfidf"] = normVal
        # Sets the default value for person to person and person to publication edges
        if coauthorValue == 0.0:
            coauthorValue = ngramdataframe["tfidf"].median()
        if authorValue == 0.0:
            authorValue = ngramdataframe["tfidf"].median()

        authorList = [
            x for y in [
                z.split(';') for z in slicedataframe[self.authorCol].values
            ] for x in y
        ]
        authors = [x for x in set(authorList) if x]
        pubs = slicedataframe[self.pubIDCol].fillna('None').unique()
        ngrams = ngramdataframe['ngram'].unique()

        slicenodes = authors
        slicenodes.extend(pubs)
        slicenodes.extend(ngrams)

        slicenodemap = {x: y for x, y in self.nodeMap.items() if x in slicenodes}

        with open(filePath, 'a') as file:
            file.write("# A network in a general multilayer format\n")
            file.write("*Vertices {0}\n".format(len(slicenodemap)))
            for x, y in slicenodemap.items():
                tmpStr = '{0} "{1}"\n'.format(y, x)
                if tmpStr:
                    file.write(tmpStr)
            file.write("*Multilayer\n")
            file.write("# layer node layer node [weight]\n")
            if self.debug is True:
                print('\tWriting inter-layer links to file.')
            for _, row in slicedataframe.iterrows():
                authors = row[self.authorCol].split(';')
                paper = row[self.pubIDCol]
                if paper not in slicenodemap.keys():
                    print(f'Cannot find {paper}')
                ngramsList = ngramdataframe.query("@ngramdataframe['doc'] == @paper")
                paperNr = slicenodemap[paper]
                if len(authors) >= 2:
                    for pair in combinations(authors, 2):
                        file.write(
                            f'1 {slicenodemap[pair[0]]} 1 {slicenodemap[pair[1]]} {coauthorValue:.3f}\n'
                        )
                for author in authors:
                    try:
                        authNr = self.nodeMap[author]
                        file.write(
                            f'1 {authNr} 2 {paperNr} {authorValue:.3f}\n'
                        )
                    except KeyError:
                        raise
                for _, ngramrow in ngramsList.iterrows():
                    try:
                        ngramNr = self.nodeMap[ngramrow['ngram']]
                        weight = ngramrow["tfidf"]
                        # TODO: Setting precision for weight could be a problem for other datasets.
                        file.write(
                            f'2 {paperNr} 3 {ngramNr} {weight:.3f}\n'
                        )
                    except KeyError:
                        print(ngramrow['ngram'])
                        raise

    def run(
        self, windowsize: int = 3, normalize: bool = True, scoreType: str = "score",
        coauthorValue: float = 0.0, authorValue: float = 0.0, recreate: bool = False,
        scorePath: str = './', outPath: str = './', scoreLimit: float = 0.1
    ):
        """Create data for all slices.
       
        The slice window size needs to correspondent to the one used for calculating the scores to be
        consistent.
       
        Choose normalize=True (default) to normalize ngram weights. In this case the maximal score
        for each time slice is 1.0. Choose the score limit accordingly.
        """
        slices = self._createSlices(windowsize)
        if scoreType == "score":
            scores = sorted([x for x in os.listdir(scorePath) if x.endswith('_score.tsv')])
        elif scoreType == "surprise":
            scores = sorted([x for x in os.listdir(scorePath) if x.endswith('_surprise.tsv')])
        # tfidfs = sorted([x for x in os.listdir(scorePath) if x.endswith('_tfidf.tsv')])
        self.createNodeRegister(scorePath, scoreLimit, scoreType)
        for sl, score in tqdm(zip(slices, scores), leave=False, position=0):
            self.writeLinks(
                sl, os.path.join(scorePath, score), scoreLimit,
                normalize, coauthorValue, authorValue,
                outpath=outPath, recreate=recreate
            )
