import os
import re
from collections import Counter, defaultdict

from tqdm import tqdm
import numpy as np
import nltk

try:
    nltk.pos_tag(nltk.word_tokenize('This is a test sentence.'))
except LookupError:
    print('Installing nltk perceptron tagger.')
    nltk.download('averaged_perceptron_tagger')


class CalculateScores(object):
    """Calculates ngram scores for documents.

    Considered parts of speech are (see NLTK docs for details)
        - Nouns: 'NN', 'NNS', 'NNP', 'NNPS'
        - Adjectives: 'JJ', 'JJR', 'JJS'
    """

    def __init__(self, sourceDataframe, textCol="text", pubIDCol="pubID", ngramsize=5,):

        self.baseDF = sourceDataframe
        self.textCol = textCol
        self.pubIDCol = pubIDCol
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

        res = defaultdict(list())

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
            target: meta["counts"] * (np.prod(valueList)) ** (1 / (2.0 * meta["maxL"]))
        }

    def run(self):
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
        return scores, self.outputDict
