import os
import re
from tqdm import tqdm
import infomap


class Clustering():
    """Cluster mulitlayer time-dependent networks using the infomap algorithm.

    Calculates clusters using the infomap algorithm. Input files are assumed
    to have multilayer Pajek format and contain the year in four digit format.
    The default settings for running the method assume an undirected multilayer
    network and will use at most 5 optimization runs.

    :param inpath: Path to input pajek files
    :type inpath: str
    :param outpath: Path for writing resulting cluster data
    :type outpath: str
    :param recreate: Toggle recreation of already exisiting files
    :type recreate: bool
    :param infomapSettings: Initializing arguments for the infomap algorithm.
    :type infomapSettings: str
    :param debug: Toggle writing of debug info to standard output.
    :type debug: bool

    .. seealso::
       Martin Rosvall and Carl T. Bergstrom (2008).
       Maps of information flow reveal community structure in complex networks.
       PNAS, 105, 1118. 10.1073/pnas.0706851105
    """

    def __init__(
        self,
        inpath: str,
        outpath: str,
        recreate: bool = False,
        infomapSettings: str = "-N5 -imultilayer -fundirected --silent",
        debug: bool = False
    ):
        self.inpath = inpath
        self.outpath = outpath
        self.infomult = infomap.Infomap(infomapSettings)
        self.recreate = recreate
        self.debug = debug

    def calcInfomap(self, inFilePath):
        """Calculate clusters for one pajek file.

        Writes found cluster (i.e. module) information in CLU and FlowTree file
        format to output path.

        :param inFilePath: Path to input pajek file
        :type inFilePath: str
        :raises OSError: If one of the output files for this year already exists.
        :returns: Writes two files with found cluster information, method return value is empty
        :rtype: None

        .. seealso::
          Infomap python documentation on mapequation
          `Infomap module <https://mapequation.github.io/infomap/python/infomap.html>`_
        """
        filename = inFilePath.split(os.pathsep)[-1]
        year = re.findall(r'\d{4}', filename)[0]
        cluFilePath = f'{self.outpath}slice_{year}.clu'
        ftreeFilePath = f'{self.outpath}slice_{year}.ftree'
        if os.path.isfile(cluFilePath) or os.path.isfile(ftreeFilePath):
            if self.recreate is False:
                raise OSError(
                    f'Files at {cluFilePath} or {ftreeFilePath} exists. Set recreate = True to rewrite files.'
                )
            if self.recreate is True:
                os.remove(cluFilePath)
                os.remove(ftreeFilePath)
        self.infomult.readInputData(inFilePath)
        self.infomult.run()
        self.infomult.writeClu(cluFilePath)
        self.infomult.writeFlowTree(ftreeFilePath)
        if self.debug is True:
            print(
                f"Clustered in {self.infomult.maxTreeDepth()} levels with codelength {self.infomult.codelength}"
            )
            print("\tDone: Slice {0}!".format(year))
        return

    def run(self):
        """Calculate infomap clustering for all pajek files in input path."""
        pajekFiles = sorted(
            [self.inpath + x for x in os.listdir(self.inpath) if x.endswith('.net')]
        )
        for file in tqdm(pajekFiles):
            self.calcInfomap(inFilePath=file)
