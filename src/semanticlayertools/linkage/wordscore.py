import os
import time
from collections import Counter
from itertools import islice, combinations
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np
import pandas as pd
import nltk
import spacy

try:
    nltk.pos_tag(nltk.word_tokenize('This is a test sentence.'))
except LookupError:
    print('Installing nltk perceptron tagger.')
    nltk.download('averaged_perceptron_tagger')


class CalculateScores():
    """Calculates ngram scores for documents.

    Considered parts of speech are (see `nltk` docs for details)
        - Nouns: 'NN', 'NNS', 'NNP', 'NNPS'
        - Adjectives: 'JJ', 'JJR', 'JJS'

    All texts of the corpus are tokenized and POS tags are generated.
    A global dictionary of counts of different ngrams is build in `counts`.
    The ngram relations of every text are listed in `outputDict`.

    Scoring is based on counts of occurances of different words left and right of each single
    token in each ngram, weighted by ngram size, for details see reference.

    :param sourceDataframe: Dataframe containing the basic corpus
    :type sourceDataframe: class:`pandas.DataFrame`
    :param textColumn: Column name to use for ngram calculation
    :type textColumn: str
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
        textColumn:str = "text",
        pubIDColumn:str = "pubID",
        yearColumn:str = 'year',
        ngramMax:int = 5,
        ngramMin:int = 2,
        debug:bool = False
    ):

        self.baseDF = sourceDataframe
        self.textCol = textColumn
        self.pubIDCol = pubIDColumn
        self.yearCol = yearColumn
        self.ngramEnd = ngramMax
        if ngramMin > 2:
            raise ValueError('The minimal ngram size has to be either 1 or 2!')
        self.ngramStart = ngramMin
        self.currentyear = ''
        self.scores = {}
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

    def getTermPatterns(self, year, dataframe, useSpacy=False, tokenMinLength=2):
        """Create dictionaries of occuring ngrams."""
        self.counts[year] = {}
        self.outputDict[year] = {}
        allNGrams = {x: [] for x in range(self.ngramStart, self.ngramEnd + 1, 1)}
        pos_tag = ["NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS"]
        for _, row in tqdm(dataframe.iterrows(), leave=False):
            if useSpacy is True:
                doc = nlp(row[self.textCol])
                tempNGram = []
                sentList = []
                for sent in list(doc.sents):
                    sentpos = []
                    for token in sent:
                        if token.tag_ in pos_tag:
                            sentpos.append(token.lemma_)
                    sentList.append(sentpos)

                for possent in sentList:
                    for i in range(self.ngramStart, self.ngramEnd + 1, 1):
                        val = allNGrams[i]
                        newngrams = list(nltk.ngrams(possent, i))
                        val.extend(newngrams)
                        tempNGram.extend(newngrams)
                        allNGrams.update({i: val})
            else:
                tokens = nltk.word_tokenize(row[self.textCol])
                pos = nltk.pos_tag(tokens)
                nnJJtokens = [
                    x[0].lower() for x in pos if x[1] in pos_tag and len(x[0]) > tokenMinLength
                ]
                tempNGram = []
                for i in range(self.ngramStart, self.ngramEnd + 1, 1):
                    val = allNGrams[i]
                    newngrams = list(nltk.ngrams(nnJJtokens, i))
                    val.extend(newngrams)
                    tempNGram.extend(newngrams)
                    allNGrams.update({i: val})
            self.outputDict[year][row[self.pubIDCol]] = tempNGram
        allgrams = [x for y in [y for x, y in allNGrams.items()] for x in y]
        for key, value in allNGrams.items():
            self.counts[year][key] = dict(Counter(value))

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
        recreate:bool = False, tokenMinLength:int = 2, useSpacy:bool = False, 
        limitCPUs:bool = True
    ):
        """Get score for all documents."""
        starttime = time.time()
        if useSpacy is True:
            nlp = spacy.load("en_core_web_lg")
        print(f"Got data for {self.baseDF[self.yearCol].min()} to {self.baseDF[self.yearCol].max()}.\n\tCreating slices.")
        for timeslice in self._createSlices(windowsize):
            dataframe = self.baseDF[self.baseDF[self.yearCol].isin(timeslice)]
            year = timeslice[-1]
            self.scores.update({year: {}})
            if write is True:
                filePath = f'{outpath}{str(year)}.tsv'
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
                tokenMinLength=tokenMinLength,
                useSpacy=useSpacy
            )
            uniqueNGrams = []
            for key in self.counts[year].keys():
                uniqueNGrams.extend(self.counts[year][key])
            if self.debug is True:
                print(
                    f'Found {len(uniqueNGrams)} unique {self.ngramStart} to {self.ngramEnd}-grams.')
            if limitCPUs is True:
                ncores = int(cpu_count() * 1 / 4)
            else:
                ncores = cpu_count() - 2
            if self.debug is True:
                print(f'Starting calculation of scores for {year}.')
            pool = Pool(ncores)
            chunk_size = int(len(uniqueNGrams) / ncores)
            batches = [
                list(uniqueNGrams)[i:i + chunk_size] for i in range(0, len(uniqueNGrams), chunk_size)
            ]
            self.currentyear = year
            ncoresResults = pool.map(self._calcBatch, batches)
            results = [x for y in ncoresResults for x in y]
            for elem in results:
                self.scores[year].update(elem)
            for key, val in self.outputDict[year].items():
                tmpList = []
                for elem in val:
                    try:
                        tmpList.append([elem, self.scores[year][elem]])
                    except TypeError:
                        print(elem)
                        raise
                self.outputDict[year].update({key: tmpList})
            if write is True:
                filePath = f'{outpath}{str(year)}.tsv'
                if recreate is True:
                    try:
                        os.remove(filePath)
                    except FileNotFoundError:
                        pass
                with open(filePath, 'a') as yearfile:
                    for pub in dataframe[self.pubIDCol].unique():
                        for elem in self.outputDict[year][pub]:
                            yearfile.write(f'{pub}\t{elem[0]}\t{elem[1]}\n')
                if self.debug is True:
                    print(f'Done creating scores for {year}, written to {filePath}.')
        print(f'Done in {(time.time() - starttime)/60:.2f} minutes.')
        if write is True:
            return
        return self.scores, self.outputDict


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
        scores = [x for x in os.listdir(scorePath) if x.endswith('.tsv')]
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
        coauthorValue: float = 0.0, authorValue: float = 0.0,
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

        # Sets the default value for person to person and person to publication edges
        # TODO: This should be configurable and different for paper to person, and person to person edges
        if coauthorValue is 0.0:
            coauthorValue = ngramdataframe[2].median()
        if authorValue is 0.0:
            authorValue = ngramdataframe[2].median()

        authorList = [
                    x for y in [
                        z.split(';') for z in slicedataframe[self.authorCol].values
                    ] for x in y
                ]
        authors = [x for x in set(authorList) if x]
        pubs = slicedataframe[self.pubIDCol].fillna('None').unique()
        ngrams = ngramdataframe[1].unique()

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
                ngramsList = ngramdataframe.query("@ngramdataframe[0] == @paper")
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
                        weight = ngramrow[2]
                        file.write(
                            f'2 {paperNr} 3 {ngramNr} {weight:.2f}\n'
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
        scores = sorted([x for x in os.listdir(scorePath) if x.endswith('.tsv')])
        self.createNodeRegister(scorePath, scoreLimit)
        for sl,score in tqdm(zip(slices, scores), leave=False, position=0):
            self.writeLinks(sl, os.path.join(scorePath, score), scoreLimit, normalize,
                coauthorValue, authorValue, outpath=outPath, recreate=recreate
            )
