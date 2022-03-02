import os
from typing import TypeVar

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

try:
    from sentence_transformers import SentenceTransformer
    import umap
    import hdbscan
    import torch
except ModuleNotFoundError as e:
    print('Please install the dependencies for the visualization routines, using `pip install semanticlayertools[embeddml]`.')
    raise e

smoothing = TypeVar('smoothing', bool, float)


def gaussian_smooth(x, y, grid, sd):
    weights = np.transpose([stats.norm.pdf(grid, m, sd) for m in x])
    weights = weights / weights.sum(0)
    return (weights * y).sum(1)


def streamgraph(
    filepath: str, smooth: smoothing = False,
    minClusterSize: int = 1000, showNthGrid: int = 5
):
    """Plot streamgraph of cluster sizes vs years.

    Based on https://www.python-graph-gallery.com/streamchart-basic-matplotlib
    """
    basedf = pd.read_csv(filepath)
    basedata = basedf.groupby(['year', 'cluster']).size().to_frame('counts').reset_index()
    yearbase = [
        x for x in range(
            int(basedata.year.min()), int(basedata.year.max()) + 1
        )
    ]
    largeclu = list(basedata.groupby('cluster').sum().query(f'counts > {minClusterSize}').index)
    cluDict = {}
    for clu in basedata.cluster.unique():
        if clu in largeclu:
            cluvec = []
            basedf = basedata.query('cluster == @clu')
            baseyears = list(basedf.year.unique())
            for year in yearbase:
                if year in baseyears:
                    cluvec.append(basedf.query('year == @year').counts.iloc[0])
                else:
                    cluvec.append(0)
            cluDict[clu] = cluvec

    fig, ax = plt.subplots(figsize=(16, 9))
    if type(smooth) is float:
        grid = np.linspace(yearbase[0], yearbase[-1], num=100)
        y = [np.array(x) for x in cluDict.values()]
        y_smoothed = [gaussian_smooth(yearbase, y_, grid, smooth) for y_ in y]
        ax.stackplot(
            grid,
            y_smoothed,
            labels=cluDict.keys(),
            baseline="sym",
            colors=plt.get_cmap('tab20').colors
        )

        pass
    else:
        ax.stackplot(
            yearbase,
            cluDict.values(),
            labels=cluDict.keys(),
            baseline='sym',
            colors=plt.get_cmap('tab20').colors
        )
    ax.legend()
    ax.set_title('Cluster sizes')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of publications')
    ax.yaxis.set_ticklabels([])
    ax.xaxis.grid(color='gray')
    temp = ax.xaxis.get_ticklabels()
    temp = list(set(temp) - set(temp[::showNthGrid]))
    for label in temp:
        label.set_visible(False)
    ax.set_axisbelow(True)
    return fig


def embeddedTextPlotting(
    infolderpath: str, columnName: str, outpath: str,
    umapNeighors: int = 200,
):
    """Create embedding for corpus text."""
    print('Initializing embedder model.')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    clusterfiles = os.listdir(infolderpath)
    clusterdf = []
    for x in clusterfiles:
        try:
            clusterdf.append(
                pd.read_json(os.path.join(infolderpath, x), lines=True)
            )
        except ValueError:
            raise
    dataframe = pd.concat(clusterdf, ignore_index=True)
    dataframe = dataframe.dropna(subset=[columnName], axis=0).reset_index(drop=True)
    corpus = [x[0] for x in dataframe[columnName].values]
    print('Start embedding.')
    corpus_embeddings = model.encode(
        corpus,
        convert_to_tensor=True
    )
    torch.save(
        corpus_embeddings,
        f'{os.path.join(outpath, "embeddedCorpus.pt")}'
    )
    print('\tDone\nStarting mapping to 2D.')
    corpus_embeddings_2D = umap.UMAP(
        n_neighbors=umapNeighors,
        n_components=2,
        metric='cosine'
    ).fit_transform(corpus_embeddings)
    np.savetxt(
        os.path.join(outpath, "embeddedCorpus_2d.csv"),
        corpus_embeddings_2D,
        delimiter=',',
        newline='\n'
    )
    print('\tDone.')
    dataframe.insert(0, 'x', corpus_embeddings_2D[:, 0])
    dataframe.insert(0, 'y', corpus_embeddings_2D[:, 1])
    return dataframe


def embeddedTextClustering(
    infolderpath: str, columnName: str, emdeddingspath: str, outpath: str,
    umapNeighors: int = 200, umapComponents: int = 50,
    hdbscanMinCluster: int = 500,
):
    """Create clustering based on embedding for corpus texts."""
    print('Initializing embedder model.')
    clusterfiles = os.listdir(infolderpath)
    clusterdf = []
    for x in clusterfiles:
        try:
            clusterdf.append(
                pd.read_json(os.path.join(infolderpath, x), lines=True)
            )
        except ValueError:
            raise
    dataframe = pd.concat(clusterdf, ignore_index=True)
    dataframe = dataframe.dropna(subset=[columnName], axis=0).reset_index(drop=True)
    print('Loading embedding.')
    corpus_embeddings = torch.load(emdeddingspath)
    print('\tDone\nStarting mapping to lower dimensions.')
    corpus_embeddings_50D = umap.UMAP(
        n_neighbors=umapNeighors,
        n_components=umapComponents,
        min_dist=0.0,
        metric='cosine'
    ).fit_transform(corpus_embeddings)
    np.savetxt(
        os.path.join(outpath, "embeddedCorpus_50d.csv"),
        corpus_embeddings_50D,
        delimiter=',',
        newline='\n'
    )
    print('\tDone.\nStarting clustering.')
    cluster = hdbscan.HDBSCAN(
        min_cluster_size=hdbscanMinCluster,
        metric='euclidean',
        cluster_selection_method='eom'
    ).fit(corpus_embeddings_50D)
    dataframe.insert(0, 'label', cluster.labels_)
    return dataframe
