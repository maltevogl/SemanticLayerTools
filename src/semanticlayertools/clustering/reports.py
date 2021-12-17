import re
import os
from tqdm import tqdm

import spacy
import textacy
import textacy.tm
import pandas as pd
import numpy as np
import multiprocessing

num_processes = multiprocessing.cpu_count()

mainLanguageCorp = 'en_core_web_lg'
nlp = spacy.load(mainLanguageCorp)


class ClusterReports():

    def __init__(
        self, infile:str, metadatapath:str, outpath:str,
        numberProc: int=num_processes, minClusterSize: int=1000
    ):
        self.numberProc = numberProc
        self.minClusterSize = minClusterSize
        self.metadatapath = metadatapath
        self.clusterdf = pd.read_csv(infile)
        basedata = self.clusterdf.groupby(['year', 'cluster']).size().to_frame('counts').reset_index()
        largeClusterList = list(
            basedata.groupby('cluster').sum().query(f'counts > {self.minClusterSize}').index
        )
        self.clusternodes = self.clusterdf.query(
            'cluster in @largeClusterList'
        )
        outfolder = infile.split(os.path.sep)[-1].split('.')[0]
        self.outpath = os.path.join(outpath, outfolder)
        if os.path.isdir(self.outpath):
            raise OSError(f'Output folder {self.outpath} exists. Aborting.')
        else:
            os.mkdir(self.outpath)
            for clu in largeClusterList:
                os.mkdir(os.path.join(self.outpath, f'Cluster_{clu}'))

    def create_corpus(self, dataframe):
        """Create corpus out of dataframe."""
        docs = []
        titles = [x[0] for x in dataframe.title.values if type(x) == list]
        for title in tqdm(titles):
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


    def find_topics(self, corpus_titles: list, n_topics: int, top_words: int, outpath:str='./', writeReport: bool=False):
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

        doc_topic_matrix = model.transform(doc_term_matrix)

        topics = []
        for topic_idx, top_terms in model.top_topic_terms(vectorizer.id_to_term, top_n=top_words):
            topics.append("topic " + str(topic_idx) + ": " + "   ".join(top_terms))
        if writeReport is True:
            outfile.write(f'\n\n\tTopics in cluster for {n_topics} topics:\n')
            for topic in topics:
                outfile.write(f'\t\t{topic}\n')
        else:
            print("\nTopics in the cluster:\n")
            return topics

    def fullReport(self, cluster):
        """Generate full cluster report."""
        with open(f'{outFolderReports}Report_{cluster}.txt', 'a') as outfile:
            selection = self.clusterdf.query('cluster == @cluster')
            nodeList = list(set(selection.node.values))
            starttime = time.time()
            result = {}
            resultMeta = []
            for key, vals in groupby(sorted(nodeList), lambda x: x[:4]):
                result[int(key)] = list(vals)

            for key in result.keys():
                if key > 1949 and key < 2005 and key != 1996:
                    yeardata = pd.read_json(f'{inFolderMetadata}{key}_meta.json', lines=True)
                    selectNodedata = yeardata[yeardata.nodeID.isin(result[key])]
                    resultMeta.append(selectNodedata)

            metadata = pd.concat(resultMeta)
            metadata.to_json(
                f'{outFolderReports}meta/cluster_{cluster}_meta.json',
                orient='records',
                lines=True
            )
            foundNodes = [x[0] for x in metadata.bibcode.values]
            notFound = [x for x in nodeList if x not in foundNodes]

            outfile.write(
                f'\tGot {len(nodeList)} unique publications in time range:\
                 {selection.year.min()} to {selection.year.max()}.\n'
            )
            outfile.write(
                f'\t\tFound metadata for {metadata.shape[0]} publications.\n'
            )
            outfile.write(
                f'\t\tThere are {len([x for x in foundNodes if x not in nodeList])}\
                found publications which where NOT in the query list.\n'
            )
            outfile.write(
                f'\t\tThere are {len(notFound)} publication(s) which where NOT found:\n'
            )

            topAuthors = Counter(
                [x for y in [x for x in metadata.author.values if type(x) == list] for x in y]
            ).most_common(20)
            outfile.write('\n\tThe top authors of this cluster are:\n')
            for elem in topAuthors:
                outfile.write(f'\t\t{elem[0]}: {elem[1]} pubs\n.')
            topAffils = Counter(
                [x for y in [x for x in metadata.aff.values if type(x) == list] for x in y]
            ).most_common(20)
            outfile.write('\n\tThe top 20 affiliations of this cluster are:\n')
            for elem in topAffils:
                outfile.write(f'\t\t{elem[0]}: {elem[1]} authors.\n')
            outfile.write(
                f'\n\n\tFinished analysis of cluster {cluster} with {len(nodeList)}\
                 unique publications in {time.time()- starttime} seconds.\n\n'
            )
            corpus = create_corpus(metadata)
            find_topics(
                corpus, n_topics=15, top_words=10, writeReport=True, outfile=outfile
            )
            find_topics(
                corpus, n_topics=50, top_words=10, writeReport=True, outfile=outfile
            )
            outfile.write(
                f'\n\n\tFinished analysis of topics in {cluster} in {time.time()- starttime} seconds.\n\n'
            )
            return cluster

    def _mergeData(self, filename, publicationIDcolumn: str='nodeID'):
        filepath = os.path.join(self.metadatapath, filename)
        data = pd.read_json(filepath, lines=True)
        selectMerge = data.merge(self.clusternodes, left_on=publicationIDcolumn, right_on='node', how='inner')
        if selectMerge.shape[0]>0:
            for clu, g0 in selectMerge.groupby('cluster'):
                g0.to_json(os.path.join(self.outpath, f'Cluster_{clu}', 'merged_' + filename) , orient='records', lines=True)
        self.pbar.update(1)
        return

    def gatherClusterMetadata(self):
        filenames = os.listdir(self.metadatapath)
        #chunk_size = int(len(filenames) / self.numberProc)
        #chunks = np.array_split(filenames, chunk_size)
        self.pbar = tqdm(len(filenames))
        pool = multiprocessing.Pool(self.numberProc)
        result = pool.map(self._mergeData, filenames, chunksize=int(len(filenames) / self.numberProc))
        return

            # filepath = os.path.join(self.metadatapath, filename)
            # data = pd.read_json(filepath, lines=True)
            # selectMerge = data.merge(self.clusternodes, left_on=publicationIDcolumn, right_on='node', how='inner')
            # if selectMerge.shape[0]>0:
            #     for clu, g0 in selectMerge.groupby('cluster'):
            #         g0.to_json(os.path.join(self.outpath, f'Cluster_{clu}', 'merged_' + filename) , orient='records', lines=True)
