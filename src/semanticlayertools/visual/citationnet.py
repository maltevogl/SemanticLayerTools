import dimcli
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import re
import os
from collections import Counter
from requests.exceptions import HTTPError


class GenerateTree:
    """Generate tree for citationent visualization.

    For a given input document, its references and citations are evaluated. In
    a second step, citations of citations and references of references are
    extracted. This information is used to generate a tree like network for
    visualization.
    """

    def __init__(self, verbose: bool = False, api_key=""):
        """Init module."""
        while not dimcli.login_status():
            try:
                dimcli.login(key=api_key)
            except HTTPError as e:
                if e.response.status_code == 401:
                    raise
                time.sleep(5)
                pass

        self.dsl: dimcli.Dsl = dimcli.Dsl()
        self._verbose = verbose
        self.startDoi: str = ""
        self.firstAuthor: str = "NoAuthor"
        self.citationLimit: int = 100
        self.dataframeList = []

        self.stringClean = {
            r"\s": "__",
            "/": "_slash_",
            ":": "_colon_",
            r"\.": "_dot_"
        }

    def _cleanTitleString(self, row):
        """Clean non-JSON characters from titles.

        Removes newline characters, double backslashes and quoted '"'.
        """
        try:
            title = row
            for pair in [('\n', ' '), (r':?\\+', ''), ('"', '')]:
                title = re.sub(pair[0], pair[1], title)
            return title
        except Exception:
            return 'Can not process title.'

    def _formatFOR(self, row):
        """Format existing FOR codes.

        Each publication has a total value of one. Only first level parts of
        codes are counted. If no FOR code exist, return '00:1'.

        Example: "02, 0201, 0204, 06" yields "02:0.75;06:025"
        """
        try:
            inputForcodes = [x['name'][:2] for x in row]
            forcodes = ';'.join(
                [f'{x[0]}:{x[1]/len(inputForcodes):.2f}' for x in Counter(
                    inputForcodes
                ).most_common()]
            )
        except TypeError:
            forcodes = '00:1'
        return forcodes

    def _editDF(self, inputdf, dftype='cite_l1', level2List=None):
        """Return reformated dataframe. """
        retCols = ['source', 'target', 'doi', 'year', 'title', 'times_cited', 'forcodes', 'level', 'is_input']
        formatedFOR = inputdf.category_for.apply(lambda row: self._formatFOR(row))
        inputdf.insert(0, 'forcodes', formatedFOR)
        inputdf.drop(['category_for'], axis=1, inplace=True)
        inputdf.rename(columns={'id': 'source'}, inplace=True)
        if dftype in ['ref_l1', 'cite_l2', 'ref_l2']:
            outdf = inputdf.explode('reference_ids')
            outdf.rename(columns={'reference_ids': 'target'}, inplace=True)
            if dftype == 'cite_l2':
                outdf = outdf.query('target.isin(@level2List)')
        elif dftype == 'cite_l1':
            inputdf.insert(0, 'target', self.pubids)
            outdf = inputdf.copy()
        outdf.insert(0, 'level', dftype)
        outdf = outdf.dropna(subset=['source', 'target'])
        outdf.insert(
            0,
            'is_input',
            outdf.source.apply(lambda x: x == self.pubids)
        )
        cleantitle = outdf.title.apply(lambda row: self._cleanTitleString(row))
        outdf.drop('title', axis=1, inplace=True)
        outdf.insert(0, 'title', cleantitle)
        return outdf[retCols]

    def _getMissing(self, idlist):
        """Get metadata for second level reference nodes."""
        retCols = ['source', 'doi', 'year', 'title', 'times_cited', 'forcodes', 'level', 'is_input']
        dfList = []
        if len(idlist) > 512:
            for partlist in tqdm(np.array_split(idlist, round(len(idlist)/400))):
                res = self.dsl.query_iterative(
                    f"""search
                          publications
                        where
                          id in {json.dumps(list(partlist))}
                        return
                          publications[id+doi+times_cited+category_for+title+year]
                    """,
                    verbose=self._verbose
                )
                dfList.append(res.as_dataframe())
            retDF = pd.concat(dfList)
        else:
            res = self.dsl.query_iterative(
                f"""search
                      publications
                    where
                      id in {json.dumps(list(idlist))}
                    return
                      publications[id+doi+times_cited+category_for+title+year]
                """,
                verbose=self._verbose
            )
            retDF = res.as_dataframe()
        formatedFOR = retDF.category_for.apply(lambda row: self._formatFOR(row))
        retDF.insert(0, 'forcodes', formatedFOR)
        retDF.drop(['category_for'], axis=1, inplace=True)
        retDF.rename(columns={'id': 'source'}, inplace=True)
        retDF.insert(0, 'level', 'ref_l3')
        retDF.insert(0, 'is_input', False)
        cleantitle = retDF.title.apply(lambda row: self._cleanTitleString(row))
        retDF.drop('title', axis=1, inplace=True)
        retDF.insert(0, 'title', cleantitle)
        return retDF[retCols]

    def query(self, startDoi='', citationLimit=100):
        self.startDoi = startDoi
        self.citationLimit = citationLimit
        self.dataframeList = []
        starttime = time.time()
        doi2id = self.dsl.query(
            f"""search
                  publications
                where
                  doi = "{startDoi}"
                return
                  publications[id+authors+doi+times_cited+category_for+title+year+reference_ids]
            """,
            verbose=self._verbose
        )
        querydf = doi2id.as_dataframe()
        if querydf.shape[0] == 0:
            return f"The dataset contains no entry for {startDoi}."
        elif querydf['times_cited'].iloc[0] >= self.citationLimit:
            return f"{startDoi} is cited {querydf['times_cited'].iloc[0]} times. You can try to change the limit, if possible."
        try:
            self.firstAuthor = doi2id.as_dataframe_authors()['last_name'].iloc[0]
        except KeyError:
            pass
        self.pubids = querydf['id'].values[0]
        self.pubrefs = list(
            [x for y in querydf['reference_ids'].values for x in y]
        )
        self.dataframeList.append(
            self._editDF(querydf, dftype="ref_l1")
        )
        ref1trgtList = list(self.dataframeList[0].target.values)
        cit1df = self.dsl.query_iterative(
            f"""search
                  publications
                where
                  reference_ids = "{self.pubids}"
                return
                  publications[id+doi+times_cited+category_for+title+year+reference_ids]
            """,
            verbose=self._verbose)
        self.dataframeList.append(
            self._editDF(cit1df.as_dataframe(), dftype='cite_l1')
        )
        cit1SrcList = list(self.dataframeList[1].source.values)
        cit2df = self.dsl.query_iterative(
            f"""search
                  publications
                where
                  reference_ids in {json.dumps(cit1SrcList)}
                return
                  publications[id+doi+times_cited+category_for+title+year+reference_ids]""",
            verbose=self._verbose
        )
        self.dataframeList.append(
            self._editDF(cit2df.as_dataframe(), dftype='cite_l2', level2List=cit1SrcList)
        )
        ref2df = self.dsl.query_iterative(
            f"""search
                  publications
                where
                  id in {json.dumps(ref1trgtList)}
                return
                  publications[id+doi+times_cited+category_for+title+year+reference_ids]""",
            verbose=self._verbose
        )
        self.dataframeList.append(
            self._editDF(ref2df.as_dataframe(), dftype='ref_l2')
        )
        print(f'Finished queries in {time.time() - starttime} seconds.')
        return self

    def returnLinks(self):
        """Return all links as dataframe."""
        return pd.concat(self.dataframeList)

    def generateNetworkFiles(self, outfolder):
        """Generates JSON with nodes and edges lists."""
        starttime = time.time()
        outformat = {'nodes': [], 'edges': []}
        dflinks = pd.concat(self.dataframeList)
        srcNodes = dflinks.source.unique()
        trgNodes = [x for x in dflinks.target.unique() if x not in srcNodes]
        nodeMetadata = pd.concat(
            [
                dflinks.drop('target', axis=1).drop_duplicates(),
                self._getMissing(trgNodes)
            ]
        )
        for idx, row in nodeMetadata.fillna('').iterrows():
            outformat['nodes'].append(
                {
                    'id': row['source'],
                    'attributes':
                        {
                            "title": row['title'],
                            "doi": row["doi"],
                            "nodeyear": row["year"],
                            "ref-by-count": row["times_cited"],
                            "is_input_DOI": row['is_input'],
                            "category_for": row["forcodes"],
                            'level': row['level']
                        }
                }
            )
        for idx, row in dflinks.fillna('').iterrows():
            outformat['edges'].append(
                {
                    'source': row['source'],
                    'target': row['target'],
                    'attributes':
                        {
                            'year': row['year'],
                            'level': row['level']
                        }
                }
            )
        doiname = self.startDoi
        firstauthor = self.firstAuthor
        for key, val in self.stringClean.items():
            doiname = re.sub(key, val, doiname)
            firstauthor = re.sub(key, val, firstauthor)

        outfile = os.path.join(outfolder, f'{firstauthor}_{doiname}.json')
        with open(outfile, 'w', encoding="utf8") as ofile:
            json.dump(outformat, ofile, ensure_ascii=True)
        return time.time() - starttime, f'{firstauthor}_{doiname}.json'
