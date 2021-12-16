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
    yearColumn='year',
    authorColumn='author',
    pubIDColumn='publicationID',
    ngramsize=5,
    scoreLimit=1.0
):
    """Run all steps for multilayer network generation using wordscoring."""

    if tempFiles is True:
        basedir = tempfile.TemporaryDirectory().name
        clusterout = outPath
    else:
        timestamp = datetime.now().strftime("_%Y_%m_%d")
        basedir = outPath + 'Clustering' + timestamp
        clusterout = f'{basedir}/clusters/'
    for subdir in ['scores', 'links', 'clusters']:
        os.makedirs(os.path.join(basedir, subdir))
    print(f'Start cleaning {textColumn} column.')
    clean = dataframe[textColumn].apply(lambda row: lemmaSpacy(htmlTags(row)))

    dataframe.insert(0, 'clean', clean)

    if tempFiles is False:
        dataframe.to_json(f'{basedir}/sourceDFcleaned.json', orient='records', lines=True)
    print('\tDone.')
    score = CalculateScores(
        dataframe,
        textColumn='clean',
        pubIDColumn=pubIDColumn,
        ngramsize=ngramsize
    )
    links = LinksOverTime(
        dataframe,
        authorColumn=authorColumn,
        pubIDColumn=pubIDColumn,
        yearColumn=yearColumn
    )
    clusters = Clustering()

    print(f'Start calculating scores for {dataframe.shape[0]} texts.')
    score.run(
        write=True, outpath=f'{basedir}/scores/', recreate=True
    )
    print('\tDone.')
    print(f'Start creating links with scoreLimit > {scoreLimit}.')
    links.run(
        recreate=True,
        scorePath=f'{basedir}/scores/',
        outPath=f'{basedir}/links/',
        scoreLimit=scoreLimit
    )
    print('\tDone.')
    print('Start calculating infomap clusters.')
    clusters.run(
        pajekPath=f'{basedir}/links/',
        outPath=clusterout,
    )
    print('\tDone.')
    with open(f'{basedir}/README.txt', 'w+') as file:
        file.write(
            f"""Run of clustering {datetime.now().strftime("%Y_%m_%d")}

            Text cleaned in column: {textColumn} (html tags removed and lemmatized)
            Authors information from column: {authorColumn}
            Unique publication IDs from columns: {pubIDColumn}
            Ngram scores greater {scoreLimit} were considered for link creation.
            Clustering result in folder: {clusterout}
            """
        )
        if tempFiles is True:
            file.write(
                'Temporay files for wordscores and multilayer networks were deleted.'
            )
    print(f"""Results in {clusterout}.\n
    Head over to https://www.mapequation.org/alluvial/ to visualize the ftree files.
        """)
