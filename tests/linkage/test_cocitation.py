import unittest
import os
import tempfile
import pandas as pd

outPath = tempfile.TemporaryDirectory().name

from semanticlayertools.linkage.citation import Cocitations


basePath = os.path.dirname(os.path.abspath(__file__ + "/../"))
filePath = f'{basePath}/testdata/cocite/'

filename = [x for x in os.listdir(filePath) if x.endswith('.json')]

testchunk = pd.read_json(filePath + filename[0], lines=True)

class TestCocitationCreation(unittest.TestCase):

    def setUp(self):
        self.cociteinit = Cocitations(
            filePath, outPath, 'reference',
            numberProc=2
        )


    def test_getCombinations(self):
        res = self.cociteinit.getCombinations(testchunk)
        assert(type(res[0]) == tuple)

