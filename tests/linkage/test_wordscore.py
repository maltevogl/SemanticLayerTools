import unittest
import os
import pandas as pd
from semanticlayertools.linkage.wordscore import CalculateScores

basePath = os.path.dirname(os.path.abspath(__file__ + "/../"))
filePath = f'{basePath}/testdata/testdata.json'

df = pd.read_json(filePath)


class TestCalculateScores(unittest.TestCase):

    def setUp(self):
        self.scoreinit = CalculateScores(df, textColumn='clean', pubIDColumn='pubIDs')
        self.scorePattern = self.scoreinit.getTermPatterns()
        self.scoreOut = self.scoreinit.run()

    def test_scoring(self):
        self.assertLessEqual(self.scoreOut[0][('theory',)], 1)
