import pandas as pd
import time

from tqdm import tqdm

from sentence_transformers import SentenceTransformer
import umap

from langdetect import detect

import textacy
import textacy.tm

from ..cleaning.text import htmlTags


def createTopicEmbedding(
    dataframePath: str, year: str, textColumn: str = 'title', idColumn: str = "nodeID", yearColumn: str = 'year', targetLanguage: str = "en",
    device: str = "cuda", nNei: int = 20, nComp: int = 2, metric: str = "cosine", useStopList: bool = False, denseMap: bool = False, debug: bool = True
):
    starttime = time.time()
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    dfTemp = pd.read_json(dataframePath, lines=True)
    dfTemp.insert(0, 'year', year)
    texts = []
    for idx, row in tqdm(dfTemp.iterrows()):
        if type(row[textColumn])==list:
            textelem = row[textColumn][0]
            try:
                lang = detect(textelem)
            except:
                continue
            if type(textelem) == str:
                if lang == targetLanguage:
                    outtext = htmlTags(textelem)
                    texts.append(
                        (row[idColumn], row[yearColumn], outtext)
                    )
    dfTexts = pd.DataFrame(texts, columns=['idx', 'year', 'text'])
    basedata = dfTexts.drop_duplicates(subset="text", keep="last")
    embedded = model.encode(
        basedata.text.values,
        convert_to_numpy=True,
        device=device,
        normalize_embeddings=True
    )
    time2 = time.time()
    if debug is True:
        print(f"Done embedding in {time2-starttime} sec.")
    embedded_2D = umap.UMAP(
        densmap=denseMap,
        n_neighbors=nNei,
        n_components=nComp,
        metric=metric
    ).fit_transform(embedded)
    time3 = time.time()
    if debug is True:
        print(f"Done mapping to 2D in {time3-time2} sec.")
    basedata.insert(0, 'x', embedded_2D[:, 0])
    basedata.insert(0, 'y', embedded_2D[:, 1])

    records = []
    for idx, row in tqdm(basedata.iterrows()):
        doc = textacy.make_spacy_doc((row["text"], {"idx": row["idx"]}), 'en_core_web_lg')
        records.append(doc)
    corpus = textacy.Corpus("en_core_web_lg", data=records)
    time4 = time.time()
    if debug is True:
        print(f"Done building corpus in {time4-time3} sec.")
    if isinstance(useStopList, list):
        STOPLIST = useStopList
    else:
        STOPLIST = []

    vectorizer = textacy.representations.vectorizers.Vectorizer(
        tf_type="linear",
        idf_type="smooth",
        norm="l2",
        min_df=20,
        max_df=0.9
    )
    tokenized_docs = (
        (
            term.lemma_ for term in textacy.extract.terms(doc, ngs=3, ents=True, ncs=True) if term.lemma_ not in STOPLIST
        ) for doc in corpus
    )
    doc_term_matrix = vectorizer.fit_transform(tokenized_docs)

    topicmodel = textacy.tm.TopicModel("nmf", 15)
    topicmodel.fit(doc_term_matrix)
    time5 = time.time()
    if debug is True:
        print(f"Done fitting topics in {time5-time4} sec.")

    topics = {}
    for topic_idx, top_terms in topicmodel.top_topic_terms(
        vectorizer.id_to_term, top_n=20
    ):
        topics[str(topic_idx)] = " ".join(top_terms)
  
    topTopic = [(x[0], x[1][0]) for x in topicmodel.top_doc_topics(topicmodel.get_doc_topic_matrix(doc_term_matrix), top_n=1)]

    dfTopic = pd.DataFrame(topTopic)
    basedata.insert(3, 'topic', dfTopic[1].values)
    sizeDict = {key: 0.5 + 100 / val for key, val in basedata.topic.value_counts().to_dict().items()}
    sizes = basedata.topic.apply(lambda x: sizeDict[x])
    basedata.insert(4, 'sizes', sizes)
    if debug is True:
        print(f"Done in {time5-starttime} sec total.")
    return topics, basedata  
