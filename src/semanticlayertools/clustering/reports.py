import re
import os
import time
import multiprocessing
from collections import Counter
from tqdm import tqdm

import spacy
import textacy
import textacy.tm
import pandas as pd
import numpy as np
import warnings

num_processes = multiprocessing.cpu_count()

mainLanguageCorp = 'en_core_web_lg'
nlp = spacy.load(mainLanguageCorp)


class ClusterReports():

    def __init__(
        self, infile: str, metadatapath: str, outpath: str,
        numberProc: int = num_processes, minClusterSize: int = 1000,
        timerange: tuple = (1945, 2005)
    ):
        self.numberProc = numberProc
        self.minClusterSize = minClusterSize
        self.metadatapath = metadatapath
        clusterdf = pd.read_csv(infile)
        basedata = clusterdf.groupby(['year', 'cluster']).size().to_frame('counts').reset_index()
        self.largeClusterList = list(
            basedata.groupby('cluster').sum().query(f'counts > {self.minClusterSize}').index
        )
        self.clusternodes = clusterdf.query(
            'cluster in @self.largeClusterList'
        )
        outfolder = infile.split(os.path.sep)[-1][:-4]
        self.timerange = timerange
        self.outpath = os.path.join(outpath, outfolder)
        if os.path.isdir(self.outpath):
            raise OSError(f'Output folder {self.outpath} exists. Aborting.')
        else:
            os.mkdir(self.outpath)
            for clu in self.largeClusterList:
                os.mkdir(os.path.join(self.outpath, f'Cluster_{clu}'))

    def create_corpus(self, dataframe):
        """Create corpus out of dataframe."""
        docs = []
        titles = [x[0] for x in dataframe.title.values if type(x) == list]
        for title in tqdm(titles, leave=False):
            try:
                # text pre-processing
                title = re.sub("\n", " ", title)
                title = re.sub("[\r|\t|\x0c|\d+]", "", title)
                title = re.sub("[.,]", "", title)
                title = re.sub("\\\'s", "'s", title)
                title = title.lower()

                doc = nlp(title)

                tokens_without_sw = ' '.join([t.lemma_ for t in doc if not t.is_stop])

                docs.append(tokens_without_sw)
            except:
                print(title)
                raise

        corpus_titles = textacy.Corpus(mainLanguageCorp, data=docs)
        return corpus_titles

    def find_topics(
        self, corpus_titles: list, n_topics: int, top_words: int,
    ):
        """Calculate topics in corpus."""
        vectorizer = textacy.representations.vectorizers.Vectorizer(
            tf_type="linear",
            idf_type="smooth",
            norm="l2",
            min_df=2,
            max_df=0.95
        )
        tokenized_docs = (
            (
                term.lemma_ for term in textacy.extract.terms(doc, ngs=1, ents=True)
            ) for doc in corpus_titles
        )
        doc_term_matrix = vectorizer.fit_transform(tokenized_docs)

        model = textacy.tm.TopicModel("nmf", n_topics)
        model.fit(doc_term_matrix)

        topics = []
        for topic_idx, top_terms in model.top_topic_terms(vectorizer.id_to_term, top_n=top_words):
            topics.append("topic " + str(topic_idx) + ": " + "   ".join(top_terms))
        outtext = f'\n\n\tTopics in cluster for {n_topics} topics:\n'
        for topic in topics:
            outtext += f'\t\t{topic}\n'
        return outtext

    def fullReport(self, cluster):
        """Generate full cluster report."""
        starttime = time.time()
        clusterpath = os.path.join(self.outpath, f'Cluster_{cluster}')
        clusterfiles = os.listdir(clusterpath)
        clusterdf = []
        for x in clusterfiles:
            try:
                clusterdf.append(
                    pd.read_json(os.path.join(clusterpath, x), lines=True)
                )
            except ValueError:
                raise
        dfCluster = pd.concat(clusterdf, ignore_index=True)
        basedf = self.clusternodes.query('cluster == @cluster')
        inputnodes = basedf.node.values
        foundNodes = [x[0] for x in dfCluster.bibcode.values]
        notFound = [x for x in inputnodes if x not in foundNodes]
        topAuthors = Counter(
            [x for y in [x for x in dfCluster.author.values if type(x) == list] for x in y]
        ).most_common(20)
        authortext = ''
        for x in topAuthors:
            authortext += f'\t{x[0]}: {x[1]}\n'
        topAffils = Counter(
            [x for y in [x for x in dfCluster.aff.values if type(x) == list] for x in y]
        ).most_common(21)
        affiltext = ''
        for x in topAffils[1:]:
            affiltext += f'\t{x[0]}: {x[1]}\n'
        corpus = self.create_corpus(dfCluster)
        warnings.simplefilter(action='ignore', category=FutureWarning)
        topics_15 = self.find_topics(corpus, n_topics=15, top_words=20)
        topics_50 = self.find_topics(corpus, n_topics=50, top_words=20)
        outtext = f"""Report for Cluster {cluster}

Got {len(inputnodes)} unique publications in time range: {basedf.year.min()} to {basedf.year.max()}.
    Found metadata for {dfCluster.shape[0]} publications.
    There are {len(notFound)} publications without metadata.

    The top 20 authors of this cluster are:
    {authortext}

    The top 20 affiliations of this cluster are:
    {affiltext}

    {topics_15}

    {topics_50}

Finished analysis of cluster {cluster} in {time.time()- starttime} seconds."""
        return outtext

    def _mergeData(self, filename, publicationIDcolumn: str = 'nodeID'):
        filepath = os.path.join(self.metadatapath, filename)
        data = pd.read_json(filepath, lines=True)
        selectMerge = data.merge(
            self.clusternodes,
            left_on=publicationIDcolumn,
            right_on='node',
            how='inner'
        )
        if selectMerge.shape[0] > 0:
            for clu, g0 in selectMerge.groupby('cluster'):
                g0.to_json(
                    os.path.join(
                        self.outpath,
                        f'Cluster_{clu}',
                        'merged_' + filename
                    ), orient='records', lines=True
                )
        return ''

    def gatherClusterMetadata(self):
        filenames = os.listdir(self.metadatapath)
        yearFiles = []
        for x in filenames:
            try:
                year = int(re.findall(r'\d{4}', x)[0])
            except:
                raise
            if self.timerange[0] <= year <= self.timerange[1]:
                yearFiles.append(x)
        with multiprocessing.Pool(self.numberProc) as pool:
            _ = pool.map(self._mergeData, tqdm(yearFiles, leave=False))
        return

    def writeReports(self):
        for cluster in tqdm(self.largeClusterList):
            outtext = self.fullReport(cluster)
            with open(f'{self.outpath}Cluster_{cluster}.txt', 'w') as file:
                file.write(outtext)
