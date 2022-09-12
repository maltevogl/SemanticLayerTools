import pandas as pd
import spacy
import math
import time

from tqdm import tqdm

from itertools import groupby, islice
from operator import itemgetter
from collections import Counter


def _window(seq, n):
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


def _createSlices(dataframe, column, windowsize):
    slices = []
    years = sorted(dataframe[column].unique())
    for x in _window(years, windowsize):
        slices.append(x)
    return slices


def corpusSurprise(corpusPath, yearColumn="date", languageModel="en_core_web_lg", maxLength=2000000,
    debug=False
):
    """Calculate surprise for time slices of corpus.""" 
    
    data = pd.read_csv(corpusPath, index_col=0)
    nlp = spacy.load(languageModel)
    nlp.max_length = maxLength

    timeSurpriseList = []

    for timeslice in _createSlices(data, yearColumn, 3):
        starttime = time.time()
        dataframe = data[data[yearColumn].isin(timeslice)]
        year = timeslice[-1]
        if debug is True:
            print(f"Working on {year}")
        allFulltext = ' '.join(dataframe.fullText.values)
        doc = nlp(allFulltext)
        
        all4grams = []
        all5grams = []
        all1grams = []
        all2grams = []
        
        starttime0 = time.time()
        if debug is True:
            print(f"\tFinished loading corpus in {starttime0 - starttime:.3f} seconds.")
        for sent in doc.sents:
            lemmasent = []
            for token in sent:
                if len(token.lemma_) < 2:
                    continue
                if token.is_digit:
                    continue
                if token.is_punct:
                    continue
                if not token.is_alpha:
                    continue
                lemmasent.append(token.lemma_.lower())
            all1grams.extend(lemmasent)
            
            snt2grams = zip(*[lemmasent[i:] for i in range(2)])
            all2grams.extend(
                list(snt2grams)
            )
            
            snt4grams = zip(*[lemmasent[i:] for i in range(4)])
            all4grams.extend(
                list(snt4grams)
            )
            snt5grams = zip(*[lemmasent[i:] for i in range(5)])
            all5grams.extend(
                list(snt5grams)
            )
        OneGramCounts = dict(Counter(all1grams).most_common())
        TwoGramCounts = dict(Counter(all2grams).most_common())
        
        starttime2 = time.time()
        if debug is True:
            print(f"\tFinished ngram creation in {starttime2 - starttime0:.3f} seconds.")
        
        all4grams.sort(key=lambda x: x[3])
        all5grams.sort(key=lambda x: (x[3],x[4]))
        
        sorted4grams = [list(group) for key, group in groupby(all4grams, itemgetter(3))]
        sorted5grams = [list(group) for key, group in groupby(all5grams, itemgetter(3,4))]
        
        for elem in sorted4grams:
            tokName = elem[0][3]
            if OneGramCounts[tokName] < 5:
                continue
            basisLen = len(set(elem))
            counts = dict(Counter(elem).most_common())
            probList = []
            for key, val in counts.items():
                probList.append(
                    -math.log(val/basisLen, 2)
                )
            surpriseVal = 1/basisLen * sum(probList)
            if surpriseVal > 0:
                timeSurpriseList.append((year, "1gram", tokName, surpriseVal))
        if debug is True:
            print(f"\tFinished surprise 1-gram calculation in {time.time() - starttime2:.3f} seconds..")
        starttime3 = time.time()
        for elem in sorted5grams:
            tokList = [elem[0][3], elem[0][4]]
            tokName = ' '.join(tokList)
            if TwoGramCounts[tuple(tokList)] < 5:
                continue
            basisLen = len(set(elem))
            counts = dict(Counter(elem).most_common())
            probList = []
            for key, val in counts.items():
                probList.append(
                    -math.log(val/basisLen, 2)
                )
            surpriseVal = 1/basisLen * sum(probList)
            if surpriseVal > 0:
                timeSurpriseList.append((year, "2gram", tokName, surpriseVal))
        if debug is True:
            print(f"\tFinished surprise 2-gram calculation in {time.time() - starttime3:.3f} seconds..")
    return timeSurpriseList


def corpusKDL(corpusPath, direction='post', yearColumn="date", languageModel="en_core_web_lg", maxLength=2000000,
    debug=False
):
    """Calculate Kullback-Leibler divergence between current and next (direction = "post") or previous (direction = "pre") time slice of corpus.""" 
    
    data = pd.read_csv(corpusPath, index_col=0)
    nlp = spacy.load(languageModel)
    nlp.max_length = maxLength

    slicePairs = list(_window([sl for sl in _createSlices(data, yearColumn, 3)],2))

    timeKDLList = []

    for timeslices in slicePairs:
        starttime = time.time()
        if direction == "post":
            dataframe1 = data[data[yearColumn].isin(timeslices[0])]
            dataframe2 = data[data[yearColumn].isin(timeslices[1])]
            year = timeslices[0][-1]
        elif direction == "pre":
            dataframe1 = data[data[yearColumn].isin(timeslices[1])]
            dataframe2 = data[data[yearColumn].isin(timeslices[0])]
            year = timeslices[1][-1]
        
        
        if debug is True:
            print(f"Working on {year}")
        allFulltext1 = ' '.join(dataframe1.fullText.values)
        allFulltext2 = ' '.join(dataframe2.fullText.values)

        doc1 = nlp(allFulltext1)
        doc2 = nlp(allFulltext2)
        
        all4grams1 = []
        all4grams2 = []
        
        all1grams1 = []
        all1grams2 = []
        
        starttime0 = time.time()
        if debug is True:
            print(f"\tFinished loading corpus in {starttime0 - starttime:.3f} seconds.")
        for sent in doc1.sents:
            lemmasent = []
            for token in sent:
                if len(token.lemma_) < 2:
                    continue
                if token.is_digit:
                    continue
                if token.is_punct:
                    continue
                if not token.is_alpha:
                    continue
                lemmasent.append(token.lemma_.lower())
            all1grams1.extend(lemmasent)
            
            snt4grams = zip(*[lemmasent[i:] for i in range(4)])
            all4grams1.extend(
                list(snt4grams)
            )

        for sent in doc2.sents:
            lemmasent = []
            for token in sent:
                if len(token.lemma_) < 2:
                    continue
                if token.is_digit:
                    continue
                if token.is_punct:
                    continue
                if not token.is_alpha:
                    continue
                lemmasent.append(token.lemma_.lower())
            all1grams2.extend(lemmasent)
            
            snt4grams = zip(*[lemmasent[i:] for i in range(4)])
            all4grams2.extend(
                list(snt4grams)
            )

        OneGramCounts1 = dict(Counter(all1grams1).most_common())
        
        starttime2 = time.time()
        if debug is True:
            print(f"\tFinished ngram creation in {starttime2 - starttime0:.3f} seconds.")
        
        all4grams1.sort(key=lambda x: x[3])
        all4grams2.sort(key=lambda x: x[3])
        
        sorted4grams1 = {key:list(group) for key, group in groupby(all4grams1, itemgetter(3))}
        sorted4grams2 = {key:list(group) for key, group in groupby(all4grams2, itemgetter(3))}
                    
        for key, elem1 in sorted4grams1.items():
                
            tokName = elem1[0][3]
                
            if OneGramCounts1[tokName] < 5:
                continue
            if not key in sorted4grams2.keys():
                continue
                
            elem2 = sorted4grams2[key]
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
            kdlval = sum(probList)
            timeKDLList.append((year, "1gram", tokName, kdlval))
        if debug is True:
            print(f"\tFinished surprise 1-gram calculation in {time.time() - starttime2:.3f} seconds..")
    return timeKDLList  