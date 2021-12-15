import matplotlib.pyplot as plt
import pandas as pd


def streamgraph(filepath):
    """Plot streamgraph of cluster sizes vs years."""
    basedf = pd.read_csv(filepath)
    basedata = basedf.groupby(['year', 'cluster']).size().to_frame('counts').reset_index()
    yearbase = [
        str(x) for x in range(
            int(basedata.year.min()), int(basedata.year.max()) + 1
        )
    ]
    cluDict = {}
    for clu in basedata.cluster.unique():
        cluvec = []
        basedf = basedata.query('cluster == @clu')
        baseyears = basedf.year.unique()
        for year in yearbase:
            if year in baseyears:
                cluvec.append(basedf.query('year == @year').counts.iloc[0])
            else:
                cluvec.append(0)
        cluDict[clu] = cluvec

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.stackplot(
        yearbase,
        cluDict.values(),
        labels=cluDict.keys(),
        baseline='sym'
    )
    ax.set_title('Cluster sizes')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of publications')
    ax.axhline(0, color="black", ls="--")

    return fig
