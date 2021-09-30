import os
from tqdm import tqdm
import infomap


class Clustering():
    """Cluster using infomap."""

    def __init__(
        self,
        infomapSettings="-N5 -imultilayer -fundirected --silent"
    ):
        self.infomult = infomap.Infomap(infomapSettings)

    def calcInfomap(self, inFilePath, outPath, recreate=False, debug=False):
        """Calc clusters for one pajekt file."""
        year = inFilePath.split(os.path.sep)[-1].split('_')[1].split('.')[0]
        cluFilePath = f'{outPath}slice_{year}.clu'
        ftreeFilePath = f'{outPath}slice_{year}.ftree'
        if os.path.isfile(cluFilePath) or os.path.isfile(ftreeFilePath):
            if recreate is False:
                raise IOError(
                    f'Files at {cluFilePath} or {ftreeFilePath} exists. Set recreate = True to rewrite files.'
                    )
            if recreate is True:
                os.remove(cluFilePath)
                os.remove(ftreeFilePath)
        self.infomult.readInputData(inFilePath)
        self.infomult.run()
        self.infomult.writeClu(cluFilePath)
        self.infomult.writeFlowTree(ftreeFilePath)
        if debug:
            print(
                f"Clustered in {self.infomult.maxTreeDepth()} levels with codelength {self.infomult.codelength}"
            )
            print("\tDone: Slice {0}!".format(year))
        return

    def run(self, pajekPath='./', outPath='./', recreate=False, debug=False):
        """Calculate infomap clustering for all pajek files in path."""
        pajekFiles = sorted(
            [pajekPath + x for x in os.listdir(pajekPath) if x.endswith('.net')]
        )
        for file in tqdm(pajekFiles):
            self.calcInfomap(inFilePath=file, outPath=outPath, debug=debug)
