import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from typing import TypeVar

smoothing = TypeVar('smoothing', bool, float)


def gaussian_smooth(x, y, grid, sd):
    weights = np.transpose([stats.norm.pdf(grid, m, sd) for m in x])
    weights = weights / weights.sum(0)
    return (weights * y).sum(1)


def streamgraph(filepath: str, smooth: smoothing=False, minClusterSize: int=1000, showNthGrid: int=5):
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
            baseline="sym"
            ,colors=plt.get_cmap('tab20').colors
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
    #plt.show()
    return fig
