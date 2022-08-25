import os
import time
from collections import Counter
from itertools import islice, combinations
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np
import pandas as pd
import spacy


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
        pubIDColumn:str = "pubID",
        yearColumn:str = 'year',
        tokenColumn:str='tokens',
        debug:bool = False
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
        ngramNDocs={}
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
                    (doi, key, (0.5 + 0.5 * (val / maxVal)) * np.log(nDocs/ngramNDocs[key]))
                )
        return self.ngramDocTfidf

    def getTermPatterns(self, year, dataframe):
        """Create dictionaries of occuring ngrams."""
        self.counts[year] = {}
        self.outputDict[year] = {}
        self.allNGrams = {}
        for _, row in tqdm(dataframe.iterrows(), leave=False):
            self.outputDict[year].update(
                {row[self.pubIDCol]:[tuple(x) for x in row[self.tokenColumn]]}
            )
            for elem in row[self.tokenColumn]:
                try:
                    val = self.allNGrams[len(elem)]
                except KeyError:
                    val = []
                val.append(elem)
                self.allNGrams.update({len(elem):val})
        for key, value in self.allNGrams.items():
            self.counts[year][key] = dict(Counter([tuple(x) for x in value]))

    def getScore(self, target):
        """Calculate ngram score."""
        valueList = []
        for _, subgram in enumerate(target):
            contains = [x for x in self.counts[self.currentyear][2].keys() if subgram in x]
            rvalue = len(set(x for x in contains if x[0] == subgram))
            lvalue = len(set(x for x in contains if x[1] == subgram))
            valueList.append((lvalue + 1.0) * (rvalue + 1.0))
        factors = np.prod(valueList, dtype=np.float64)
        return {
            target: 1.0/self.counts[self.currentyear][len(target)][target] * (factors) ** (1.0 / (2.0 * len(target)))
        }

    def _calcBatch(self, batch):
        res = []
        for elem in tqdm(batch, leave=False):
            res.append(self.getScore(elem))
        return res

    def run(
        self, windowsize:int = 3, write:bool = False, outpath:str = './', 
        recreate:bool = False, tokenMinCount=5, limitCPUs:bool = True
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
                filePathTF = f'{outpath}{str(year)}_tfidf.tsv'
                if os.path.isfile(filePath) or os.path.isfile(filePathTF):
                     if recreate is False:
                        raise IOError(
                            f'File at {filePath} or {filePathTF} exists. Set recreate = True to overwrite.'
                        )
            if self.debug is True:
                print(f"Creating ngram counts for {year}.")
            self.getTermPatterns(
                year=year,
                dataframe=dataframe,
            )
            uniqueNGrams = []
            for key in self.counts[year].keys():
                tempDict = {x:y for x,y in self.counts[year][key].items() if y > tokenMinCount}
                self.counts[year].update(
                    {key:tempDict}
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
            batches = [
                list(uniqueNGrams)[i:i + chunk_size] for i in range(0, len(uniqueNGrams), chunk_size)
            ]
            ncoresResults = pool.map(self._calcBatch, batches)
            results = [x for y in ncoresResults for x in y]
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
            if write is True:
                if recreate is True:
                    try:
                        os.remove(filePath)
                        os.remove(filePathTF)
                    except FileNotFoundError:
                        pass
                with open(filePath, 'a') as yearfile:
                    for pub in dataframe[self.pubIDCol].unique():
                        for elem in self.outputDict[year][pub]:
                            yearfile.write(f'{pub}\t{elem[0]}\t{elem[1]}\n')
                with open(filePathTF, 'a') as yearfile2:
                    for elem in self.ngramDocTfidf[year]:
                        yearfile2.write(f'{elem[0]}\t{elem[1]}\t{elem[2]}\n')
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
    ngrams to the register.

    :param dataframe: Source dataframe containing metadata of texts (authors, publicationID and year)
    :type dataframe: class:`pandas.DataFrame`
    :param authorColumn: Column name for author information, author names are assumed to be separated by semikolon
    :param pubIDColumn: Column name to identify publications
    :param yearColumn: Column name with year information (year as integer)
    """

    def __init__(
        self,
        dataframe:pd.DataFrame,
        authorColumn:str = 'authors',
        pubIDColumn:str = "pubID",
        yearColumn:str = 'year',
        debug=False
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

    def createNodeRegister(self, scorePath, scoreLimit):
        """Create multilayer node register for all time slices."""
        starttime = time.time()
        scores = [x for x in os.listdir(scorePath) if x.endswith('_score.tsv')]
        ngrams = [pd.read_csv(
                scorePath + score,
                sep='\t',
                header=None
            ) for score in scores]
        ngramdataframe = pd.concat(ngrams)
        ngramdataframe = ngramdataframe[ngramdataframe[2] > scoreLimit]

        authorList = [
            x for y in [
                z.split(';') for z in self.dataframe[self.authorCol].values
            ] for x in y
        ]
        authors = [x for x in set(authorList) if x]
        pubs = self.dataframe[self.pubIDCol].fillna('None').unique()
        ngrams = ngramdataframe[1].unique()
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
            y:x for x,y in enumerate(list(ngrams), start=max(self.nodeMap.values()) + 1)
        }
        self.nodeMap.update(ngramdict)
        print(f"Done building node register in {(time.time() - starttime)/60:.2f} minutes.")
        return

    def writeLinks(
        self, sl, scorePath:str, scoreLimit:float, normalize:bool,
        tfidfPath:str, coauthorValue: float = 0.0, authorValue: float = 0.0,
        outpath:str = './', recreate: bool = False
    ):
        """Write multilayer links to file in Pajek format."""
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
            sep='\t',
            header=None
        )
        if normalize is True:
            maxval = ngramdataframe[2].max()
            normVal = ngramdataframe[2]/maxval
            ngramdataframe[2] = normVal
        ngramdataframe = ngramdataframe[ngramdataframe[2] > scoreLimit]

        tfidfframe = pd.read_csv(
            tfidfPath, sep="\t", header=None
        )
        tfidfframe = tfidfframe.query("@tfidfframe[1].isin(@ngramdataframe[1].unique())")

        # Sets the default value for person to person and person to publication edges
        if coauthorValue == 0.0:
            coauthorValue = tfidfframe[2].median()
        if authorValue == 0.0:
            authorValue = tfidfframe[2].median()

        authorList = [
            x for y in [
                z.split(';') for z in slicedataframe[self.authorCol].values
            ] for x in y
        ]
        authors = [x for x in set(authorList) if x]
        pubs = slicedataframe[self.pubIDCol].fillna('None').unique()
        ngrams = tfidfframe[1].unique()

        slicenodes = authors
        slicenodes.extend(pubs)
        slicenodes.extend(ngrams)

        slicenodemap = {x:y for x,y in self.nodeMap.items() if x in slicenodes}

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
                ngramsList = tfidfframe.query("@tfidfframe[0] == @paper")
                paperNr = slicenodemap[paper]
                if len(authors) >= 2:
                    for pair in combinations(authors, 2):
                        file.write(
                            f'1 {slicenodemap[pair[0]]} 1 {slicenodemap[pair[1]]} {coauthorValue}\n'
                        )
                for author in authors:
                    try:
                        authNr = self.nodeMap[author]
                        file.write(
                            f'1 {authNr} 2 {paperNr} {authorValue}\n'
                        )
                    except KeyError:
                        raise
                for _, ngramrow in ngramsList.iterrows():
                    try:
                        ngramNr = self.nodeMap[ngramrow[1]]
                        # weight = ngramrow[2]
                        file.write(
                            f'2 {paperNr} 3 {ngramNr} {ngramrow[2]:.2f}\n'
                        )
                    except KeyError:
                        print(ngramrow[1])
                        raise

    def run(self, windowsize:int = 3, normalize:bool = True, 
        coauthorValue: float = 0.0, authorValue: float = 0.0, recreate:bool = False, 
        scorePath:str = './', outPath:str = './', scoreLimit:float = 0.1
    ):
        """Create data for all slices. 
        
        The slice window size needs to correspondent to the one used for calculating the scores to be
        consistent.
        
        Choose normalize=True (default) to normalize ngram weights. In this case the maximal score
        for each time slice is 1.0. Choose the score limit accordingly.
        """
        slices = self._createSlices(windowsize)
        scores = sorted([x for x in os.listdir(scorePath) if x.endswith('_score.tsv')])
        tfidfs = sorted([x for x in os.listdir(scorePath) if x.endswith('_tfidf.tsv')])
        self.createNodeRegister(scorePath, scoreLimit)
        for sl,score, tfidf in tqdm(zip(slices, scores, tfidfs), leave=False, position=0):
            self.writeLinks(sl, os.path.join(scorePath, score), scoreLimit, normalize,
                os.path.join(scorePath, tfidf), coauthorValue, authorValue, outpath=outPath, recreate=recreate
            )
