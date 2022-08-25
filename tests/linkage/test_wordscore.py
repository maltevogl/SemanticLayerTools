import unittest
import os
import pandas as pd
import spacy
from semanticlayertools.linkage.wordscore import CalculateScores
from semanticlayertools.cleaning.text import tokenize

try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    nlp = spacy.load("en_core_web_sm")


basePath = os.path.dirname(os.path.abspath(__file__ + "/../"))
filePath = f'{basePath}/testdata/cocite/'

df = pd.concat(
    [
        pd.read_json(
            filePath + x, lines=True
        ) for x in os.listdir(filePath) if x.endswith('.json')
    ]
)

year = df['date'].apply(lambda x: x[0][:4])
df.insert(0,'year', year)
text = df['title'].apply(lambda x: x[0])
df.insert(0, 'text', text)
tokens = df.text.apply(lambda x: tokenize(x, languageModel=nlp))
df.insert(0, 'tokens', tokens)



class TestCalculateScores(unittest.TestCase):

    def setUp(self):
        self.scoreinit = CalculateScores(
         df, tokenColumn='tokens', pubIDColumn='nodeID',
         yearColumn="year")
        # self.scorePattern = self.scoreinit.getTermPatterns(1955, df)
        self.tfidfOut, self.scoreOut, _ = self.scoreinit.run()
        # print(self.scoreOut)

    def test_scoring(self):
        scoreVal = self.scoreOut['1952'][('remark', 'composition')]
        assert(1.0 < scoreVal < 2.0)
