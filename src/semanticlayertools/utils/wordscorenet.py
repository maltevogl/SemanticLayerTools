"""Runs all steps to create a multilayer network."""
import tempfile
from datetime import datetime
import os

from ..cleaning.text import htmlTags, lemmaSpacy
from ..linkage.wordscore import CalculateScores, LinksOverTime
from ..clustering.infomap import Clustering


def run(
    dataframe,
    tempFiles=True,
    outPath='./',
    textColumn='text',
    authorColumn='author',
    pubIDColumn='publicationID',
    scoreLimit=1.0
):
    """Run all steps for multilayer network generation using wordscoring."""
    clean = dataframe[textColumn].apply(lambda row: lemmaSpacy(htmlTags(row)))

    dataframe.insert(0, 'clean', clean)

    score = CalculateScores(
        dataframe,
        textColumn='clean',
        pubIDColumn=pubIDColumn
    )
    links = LinksOverTime(
        dataframe,
        authorColumn=authorColumn,
        pubIDColumn=pubIDColumn
    )
    clusters = Clustering()
    if tempFiles is True:
        basedir = tempfile.TemporaryDirectory().name
    else:
        timestamp = datetime.now().strftime("_%Y_%m_%d")
        basedir = outPath + timestamp
    for subdir in ['scores', 'links', 'clusters']:
        os.makedirs(os.path.join(basedir, subdir))
    sc, outDict = score.run(
        write=True, outpath=f'{basedir}/scores/', recreate=True
    )
    links.run(
        recreate=True,
        scorePath=f'{basedir}/scores/',
        outPath=f'{basedir}/links/',
        scoreLimit=scoreLimit
    )
    clusters.run(
        pajekPath=f'{basedir}/links/',
        outPath=f'{outPath}',
    )
