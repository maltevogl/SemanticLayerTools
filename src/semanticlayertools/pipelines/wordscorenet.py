import tempfile
from datetime import datetime
import os
import time

from ..cleaning.text import htmlTags, lemmaSpacy
from ..linkage.wordscore import CalculateScores, LinksOverTime
from ..clustering.infomap import Clustering


def run(
    dataframe,
    tempFiles: str = True,
    outPath: str = './',
    windowsize: int = 3,
    textColumn: str = 'text',
    yearColumn: str = 'year',
    authorColumn: str = 'author',
    pubIDColumn: str = 'publicationID',
    ngramRange: tuple = (2,5),
    tokenMinLength: int = 2,
    normalize: bool = True,
    scoreLimit: float = 0.1,
    numTrials: int = 5,
    flowModel: str = 'undirected',
    recreate:bool = True,
    skipClean:bool = False,
):
    """Run all steps for multilayer network generation using wordscoring.

    Calculates word scoring for corpus documents, creates multilayer network
    by linking co-authors and authors, their publications and used ngrams and
    calculates clusters for each timeslice using the infomap algorithm.

    By default, temmporal folders are used such that only the found clusters
    are returned.

    For details of the ngram method refere to the module documentation.

    :param dataframe: The input corpus dataframe.
    :type dataframe: class:`pandas.DataFrame`
    :param tempFiles: Use temporal files during the pipeline run.
    :type tempFiles: bool
    :param outpath: Path for writing resulting cluster data, or all temporary data
    :type outpath: str
    :param windowsize: Length of year window in which text corpus is joint and network files are created
    :type windowsize: int
    :param textColumn: Column name to use for ngram calculation
    :type textColumn: str
    :param authorColumn: Column name to use for author names, assumes a string with coauthors joined by a semicolon (;)
    :type authorColumn: str
    :param pubIDColumn: Column name to use for publication identification (assumend to be unique)
    :type pubIDColumn: str
    :param yearColumn: Column name for temporal ordering publications, used during writing the scoring files
    :type yearColumn: str
    :param ngramRange: Range of considered ngrams (default: (2,5), i.e. 2- to 5-grams)
    :type ngramRange: tuple
    :param tokenMinLength: Minimal token, i.e. word, length to consider in analysis, default 2
    :type tokenMinLength: int
    :param normalize: Trigger normalization of ngram scores for each year slice. Default is True, the maximal score in year slice is then 1.0
    :type normalize: bool
    :param scoreLimit: Minimal weight in each slice corpus to consider an ngram score (default: 0.1)
    :type scoreLimit: float
    :param numTrials: Number of iterations of the infomap algorithm, default is 5
    :type numTrials: int
    :param flowModel: Flow model for the infomap algorithm, defaults to "undirected"
    :type flowModel: str
    :param recreate: Set the recreate parameter for all parts of the pipeline, i.e. existing files are overwritten, defaults to True
    :type recreate: bool
    :param skipClean: Skip the text cleaning part of the pipeline.
    :type skipClean: bool
    """
    starttime = time.time()
    if tempFiles is True:
        basedir = tempfile.TemporaryDirectory().name
        clusterout = outPath
    else:
        timestamp = datetime.now().strftime("_%Y_%m_%d")
        basedir = outPath + 'Clustering' + timestamp
        clusterout = f'{basedir}/clusters/'
    for subdir in ['scores', 'links', 'clusters']:
        os.makedirs(os.path.join(basedir, subdir))
    if skipClean is False:
        print(f'Start cleaning {textColumn} column.')
        clean = dataframe[textColumn].apply(lambda row: lemmaSpacy(htmlTags(row)))
        dataframe.insert(0, 'clean', clean)
        print('\tDone.')
    else:
        dataframe = dataframe.rename(columns={textColumn:'clean'})

    if tempFiles is False:
        dataframe.to_json(f'{basedir}/sourceDFcleaned.json', orient='records', lines=True)
    score = CalculateScores(
        dataframe,
        textColumn='clean',
        pubIDColumn=pubIDColumn,
        yearColumn=yearColumn,
        ngramMin=ngramRange[0],
        ngramMax=ngramRange[1],
    )
    links = LinksOverTime(
        dataframe,
        authorColumn=authorColumn,
        pubIDColumn=pubIDColumn,
        yearColumn=yearColumn
    )
    clusters = Clustering(
        inpath=f'{basedir}/links/',
        outpath=clusterout,
        num_trials=numTrials,
        flow_model=flowModel,
        recreate=recreate
    
    )

    print(f'Start calculating scores for {dataframe.shape[0]} texts.')
    score.run(
        windowsize=windowsize, tokenMinLength=tokenMinLength,
        write=True, outpath=f'{basedir}/scores/', recreate=True
    )
    print('\tDone.')
    print(f'Start creating links with scoreLimit > {scoreLimit}.')
    links.run(
        windowsize=windowsize,
        normalize=normalize,
        recreate=True,
        scorePath=f'{basedir}/scores/',
        outPath=f'{basedir}/links/',
        scoreLimit=scoreLimit
    )
    print('\tDone.')
    print('Start calculating infomap clusters.')
    clusters.run()
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
    print(f"""Finished in {(time.time() - starttime)/60:.2f} minutes. Results in {clusterout}.\n
    Head over to https://www.mapequation.org/alluvial/ to visualize the ftree files.
        """)
