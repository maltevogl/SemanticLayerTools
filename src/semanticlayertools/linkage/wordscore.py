import os
from collections import Counter
from itertools import islice, combinations
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np
import pandas as pd
import nltk

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
    A global dictionary of counts of different ngrams is build in `allNGrams`.
    The ngram relations of every text are listed in `outputDict`.

    Scoring counts occurance of different words left and right of each single
    token in each ngram, weighted by ngram size.

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
    """

    def __init__(
        self,
        sourceDataframe,
        textColumn="text",
        pubIDColumn="pubID",
        yearColumn='year',
        ngramsize=5,
        debug=False
    ):

        self.baseDF = sourceDataframe
        self.textCol = textColumn
        self.pubIDCol = pubIDColumn
        self.yearCol = yearColumn
        self.ngramEnd = ngramsize
        self.outputDict = {}
        self.allNGrams = []
        self.counts = {}
        self.corpussize = 1
        self.uniqueNGrams = ()
        self.debug = debug

    def getTermPatterns(self):
        """Create dictionaries of occuring ngrams."""
        allNGrams = {x: [] for x in range(1, self.ngramEnd + 1, 1)}
        pos_tag = ["NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS"]
        for _, row in tqdm(self.baseDF.iterrows()):
            tokens = nltk.word_tokenize(row[self.textCol])
            pos = nltk.pos_tag(tokens)
            nnJJtokens = [x[0].lower() for x in pos if x[1] in pos_tag]
            tempNGram = []
            for i in range(1, self.ngramEnd + 1, 1):
                val = allNGrams[i]
                newngrams = list(nltk.ngrams(nnJJtokens, i))
                val.extend(newngrams)
                tempNGram.extend(newngrams)
                allNGrams.update({i: val})
            self.outputDict[row[self.pubIDCol]] = tempNGram
        self.allNGrams = allNGrams
        allgrams = [x for y in [y for x, y in self.allNGrams.items()] for x in y]
        self.corpussize = len(allgrams)
        self.counts = Counter(allgrams)
        self.uniqueNGrams = set(allgrams)

    def getScore(self, target):
        """Calculate ngram score."""
        valueList = []
        for _, subgram in enumerate(target):
            contains = [x for x in self.allNGrams[2] if subgram in x]
            rvalue = len(set(x for x in contains if x[0] == subgram))
            lvalue = len(set(x for x in contains if x[1] == subgram))
            valueList.append((lvalue + 1) * (rvalue + 1))
        return {
            target: 1 / self.counts[target] * (np.prod(valueList)) ** (1 / (2.0 * len(target)))
        }

    def _calcBatch(self, batch):
        res = []
        for elem in tqdm(batch):
            res.append(self.getScore(elem))
        return res

    def run(self, write=False, outpath='./', recreate=False, limitCPUs=True):
        """Get score for all documents."""
        scores = {}
        self.getTermPatterns()
        if self.debug is True:
            print(f'Found {len(self.uniqueNGrams)} unique {self.ngramEnd}-grams.')
        if limitCPUs is True:
            ncores = int(cpu_count() * 1 / 4)
        else:
            ncores = cpu_count() - 2
        pool = Pool(ncores)
        chunk_size = int(len(self.uniqueNGrams) / ncores)
        batches = [
            list(self.uniqueNGrams)[i:i + chunk_size] for i in range(0, len(self.uniqueNGrams), chunk_size)
        ]
        ncoresResults = pool.map(self._calcBatch, batches)
        results = [x for y in ncoresResults for x in y]
        for elem in results:
            scores.update(elem)
        for key, val in self.outputDict.items():
            tmpList = []
            for elem in val:
                tmpList.append([elem, scores[elem]])
            self.outputDict.update({key: tmpList})
        if write is True:
            for year, df in self.baseDF.groupby(self.yearCol):
                filePath = f'{outpath}{str(year)}.tsv'
                if os.path.isfile(filePath):
                    if recreate is False:
                        raise IOError(
                            f'File at {filePath} exists. Set recreate = True to rewrite file.'
                        )
                    if recreate is True:
                        os.remove(filePath)
                with open(filePath, 'a') as yearfile:
                    for pub in df[self.pubIDCol].unique():
                        for elem in self.outputDict[pub]:
                            yearfile.write(f'{pub}\t{elem[0]}\t{elem[1]}\n')
        return scores, self.outputDict


class LinksOverTime():
    """Create multilayer pajek files for corpus.

    To keep track of nodes over time, we need a global register of node names.
    This class takes care of this, by adding new keys of authors, papers or
    ngrams to the register.

    :param dataframe: Source dataframe containing metadata of texts (authors, publicationID and year)
    :type dataframe: class:`pandas.DataFrame`
    :param authorColumn: Column name for author information
    :param pubIDColumn: Column name to identify publications
    :param yearColumn: Column name with year information
    """

    def __init__(
        self,
        dataframe,
        authorColumn='authors',
        pubIDColumn="pubID",
        yearColumn='year',
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

    def createNodeRegister(self, sl, scorePath, scoreLimit):
        """Create multilayer node register for time slice."""
        if self.debug is True:
            print(f'Slice: {sl[0]}')
        dataframe = self.dataframe[self.dataframe[self.yearColumn].isin(sl)]
        dfNgramsList = [pd.read_csv(
            scorePath + str(slN) + '.tsv',
            sep='\t',
            header=None
        ) for slN in sl]
        ngramdataframe = pd.concat(dfNgramsList)
        ngramdataframe = ngramdataframe[ngramdataframe[2] > scoreLimit]

        authorList = [x for y in [x.split(';') for x in dataframe[self.authorCol].values] for x in y]
        authors = [x for x in set(authorList) if x]
        pubs = dataframe[self.pubIDCol].fillna('None').unique()
        ngrams = ngramdataframe[1].unique()

        for authorval in authors:
            if not self.nodeMap.values():
                self.nodeMap.update({authorval: 1})
            else:
                if authorval not in self.nodeMap.keys():
                    self.nodeMap.update(
                        {authorval: max(self.nodeMap.values()) + 1}
                    )
        for pubval in list(pubs):
            if pubval not in self.nodeMap.keys():
                self.nodeMap.update({pubval: max(self.nodeMap.values()) + 1})
        for ngramval in list(ngrams):
            if ngramval not in self.nodeMap.keys():
                self.nodeMap.update({ngramval: max(self.nodeMap.values()) + 1})

        if self.debug is True:
            print(
                '\tNumber of vertices (authors, papers and ngrams) {0}'.format(
                    max(self.nodeMap.values())
                )
            )

    def writeLinks(self, sl, scorePath, scoreLimit, outpath='./', recreate=False):
        """Write multilayer links to file in Pajek format."""
        dataframe = self.dataframe[self.dataframe[self.yearColumn].isin(sl)]
        filePath = outpath + 'multilayerPajek_{0}.net'.format(sl[0])

        if os.path.isfile(filePath):
            if recreate is False:
                raise IOError(
                    f'File at {filePath} exists. Set recreate = True to rewrite file.'
                )
            if recreate is True:
                os.remove(filePath)

        dfNgramsList = [pd.read_csv(
            scorePath + str(slN) + '.tsv',
            sep='\t',
            header=None
        ) for slN in sl]
        ngramdataframe = pd.concat(dfNgramsList)
        ngramdataframe = ngramdataframe[ngramdataframe[2] > scoreLimit]

        with open(filePath, 'a') as file:
            file.write("# A network in a general multiplex format\n")
            file.write("*Vertices {0}\n".format(max(self.nodeMap.values())))
            for x, y in self.nodeMap.items():
                tmpStr = '{0} "{1}"\n'.format(y, x)
                if tmpStr:
                    file.write(tmpStr)
            file.write("*Multiplex\n")
            file.write("# layer node layer node [weight]\n")
            if self.debug is True:
                print('\tWriting inter-layer links to file.')
            for _, row in dataframe.fillna('').iterrows():
                authors = row[self.authorCol].split(';')
                paper = row[self.pubIDCol]
                if paper not in self.nodeMap.keys():
                    print(f'Cannot find {paper}')
                ngramsList = ngramdataframe[ngramdataframe[0] == paper]
                paperNr = self.nodeMap[paper]
                if len(authors) >= 2:
                    # pairs = [x for x in combinations(authors, 2)]
                    for pair in combinations(authors, 2):  # pairs:
                        file.write(
                            '{0} {1} {2} {3} 1\n'.format(
                                1,
                                self.nodeMap[pair[0]],
                                1,
                                self.nodeMap[pair[1]]
                            )
                        )
                for author in authors:
                    try:
                        authNr = self.nodeMap[author]
                        file.write(
                            '{0} {1} {2} {3} 1\n'.format(
                                1,
                                authNr,
                                2,
                                paperNr
                            )
                        )
                    except KeyError:
                        pass
                for _, ngramrow in ngramsList.iterrows():
                    try:
                        ngramNr = self.nodeMap[ngramrow[1]]
                        weight = ngramrow[2]
                        file.write(
                            '{0} {1} {2} {3} {4}\n'.format(
                                2,
                                paperNr,
                                3,
                                ngramNr,
                                weight
                            )
                        )
                    except KeyError:
                        pass

    def run(self, recreate=False, windowsize=1, scorePath='./', outPath='./', scoreLimit=1.0):
        """Create data for all slices."""
        for sl in tqdm(self._createSlices(windowsize)):
            self.createNodeRegister(sl, scorePath, scoreLimit)
            self.writeLinks(sl, scorePath, scoreLimit, outpath=outPath, recreate=recreate)
