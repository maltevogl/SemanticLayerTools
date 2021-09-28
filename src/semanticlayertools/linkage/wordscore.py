import os
import re
from collections import Counter, defaultdict
from itertools import islice, combinations

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

    Considered parts of speech are (see NLTK docs for details)
        - Nouns: 'NN', 'NNS', 'NNP', 'NNPS'
        - Adjectives: 'JJ', 'JJR', 'JJS'
    """

    def __init__(self, sourceDataframe, textColumn="text", pubIDColumn="pubID", yearColumn='year',  ngramsize=5,):

        self.baseDF = sourceDataframe
        self.textCol = textColumn
        self.pubIDCol = pubIDColumn
        self.yearCol = yearColumn
        self.ngramEnd = ngramsize
        self.outputDict = {}
        self.allNGrams = []
        self.counts = {}
        self.allgramslist = []
        self.uniqueNGrams = ()

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
        self.allgramslist = allgrams
        self.counts = Counter(allgrams)
        self.uniqueNGrams = set(allgrams)

    def getScore(self, target):
        """Calculate ngram score."""
        meta = {
            "target": target,
            "counts": self.counts[target],
            "corpusL": len(self.allgramslist),
            "maxL": len(target),
        }

        res = defaultdict(list)

        for idx, subgram in enumerate(target):
            key = idx + 1
            for tup in self.allNGrams[2]:
                if tup[1:][0] == subgram:
                    res[f"l_{key}"].append(tup[:1][0])
                elif tup[:-1][0] == subgram:
                    res[f"r_{key}"].append(tup[1:][0])
        valueList = []
        for L in range(1, meta["maxL"] + 1, 1):
            leftkey = f"l_{L}"
            rightkey = f"r_{L}"
            if rightkey not in res.keys():
                rvalue = 0
            else:
                rvalue = len(list(set(res[rightkey])))
            if leftkey not in res.keys():
                lvalue = 0
            else:
                lvalue = len(list(set(res[leftkey])))
            valueList.append((lvalue + 1) * (rvalue + 1))
        return {
            target: 1/meta["counts"] * (np.prod(valueList)) ** (1 / (2.0 * meta["maxL"]))
        }

    def run(self, write=False, outpath='./'):
        """Get score for all documents."""
        scores = {}
        self.getTermPatterns()
        for target in tqdm(self.uniqueNGrams):
            scores.update(self.getScore(target))
        for key, val in self.outputDict.items():
            tmpList = []
            for elem in val:
                tmpList.append([elem, scores[elem]])
            self.outputDict.update({key: tmpList})
        if write is True:
            for year, df in self.baseDF.groupby(self.yearCol):
                with open(f'{outpath}{str(year)}.csv', 'a') as yearfile:
                    for pub in df[self.pubIDCol].unique():
                        for elem in self.outputDict[pub]:
                            yearfile.write(f'{pub},{elem[0]},{elem[1]}')
        return scores, self.outputDict


class LinksOverTime():
    """To keep track of nodes over time, we need a global register of node names.

    Input:
    """

    def __init__(self, outputPath, scorePath, dataframe, authorColumn='authors', pubIDColumn="pubID", yearColumn='year', scoreLimit=1.0, debug=False, windowSize=1):
        self.dataframe = dataframe
        self.authorCol = authorColumn
        self.pubIDCol = pubIDColumn
        self.yearColumn = yearColumn
        self.scoreLimit = scoreLimit
        self.outpath = outputPath
        self.scorepath = scorePath
        self.nodeMap = {}
        self.debug = debug
        self.windowSize = windowSize

    def _window(self, seq):
        """Return a sliding window (of width n) over data from the iterable.

        s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...
        """
        n = self.windowSize
        it = iter(seq)
        result = tuple(islice(it, n))
        if len(result) == n:
            yield result
        for elem in it:
            result = result[1:] + (elem,)
            yield result

    def _createSlices(self):
        slices = []
        years = sorted(self.dataframe[self.yearColumn].unique())
        for x in self._window(years):
            slices.append(x)
        return slices

    def createNodeRegister(self, sl):
        """Create multilayer node register for time slice."""
        if self.debug is True:
            print(f'Slice: {sl[0]}')
        dataframe = self.dataframe[self.dataframe[self.yearColumn].isin(sl)]
        dfNgramsList = [pd.read_csv(
            self.scorepath + str(slN) + '.tsv',
            sep='\t',
            header=None
        ) for slN in sl]
        ngramdataframe = pd.concat(dfNgramsList)
        ngramdataframe = ngramdataframe[ngramdataframe[2] > self.scoreLimit]

        authorList = [x for y in dataframe[self.authorCol].values for x in y]

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

    def writeLinks(self, sl, recreate=False):
        """Write links to file."""
        dataframe = self.dataframe[self.dataframe[self.yearColumn].isin(sl)]
        filePath = self.outpath + 'multilayerPajek_{0}.net'.format(sl[0])

        if os.path.isfile(filePath):
            if recreate is False:
                raise IOError(
                    f'File at {filePath} exists. Set recreate = True to rewrite file.'
                    )
            if recreate is True:
                os.remove(filePath)

        dfNgramsList = [pd.read_csv(
            self.scorepath + str(slN) + '.tsv',
            sep='\t',
            header=None
        ) for slN in sl]
        ngramdataframe = pd.concat(dfNgramsList)
        ngramdataframe = ngramdataframe[ngramdataframe[2] > self.scoreLimit]

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
                authors = row[self.authorCol]
                paper = row[self.pubIDCol]
                if paper not in self.nodeMap.keys():
                    print(f'Cannot find {paper}')
                ngramsList = ngramdataframe[ngramdataframe[0] == paper]
                paperNr = self.nodeMap[paper]
                if len(authors) >= 2:
                    # pairs = [x for x in combinations(authors, 2)]
                    for pair in combinations(authors, 2):  # pairs:
                        file.write('{0} {1} {2} {3} 1\n'.format(
                            1,
                            self.nodeMap[pair[0]],
                            1,
                            self.nodeMap[pair[1]]
                            )
                        )
                for author in authors:
                    try:
                        authNr = self.nodeMap[author]
                        file.write('{0} {1} {2} {3} 1\n'.format(
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
                        file.write('{0} {1} {2} {3} {4}\n'.format(
                            2,
                            paperNr,
                            3,
                            ngramNr,
                            weight
                            )
                        )
                    except KeyError:
                        pass

    def run(self, recreate=False):
        """Create all data for slices."""
        for sl in tqdm(self._createSlices()):
            self.createNodeRegister(sl)
            self.writeLinks(sl, recreate=recreate)
