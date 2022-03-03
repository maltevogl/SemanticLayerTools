"""
generateCitationTree contains classes for generating citation trees for a single DOI.

https://app.dimensions.ai is used in favor of CrossRef, as it contains richer information.
Requires dimensions.ai API access.
Results are compatible to the existing GenerateCitationNetwork module.
"""
import os
import dimcli
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import math
from typing import Dict, Tuple, List
import datetime
import time
import re
from requests.exceptions import HTTPError

# type aliases
Doi = str
PubID = str
FilePath = str


class generate:
    """GenerateCitationNet makes citation/reference networks for a document.

    For a given input document, its references and citations are evaluated. In
    a second step, citations of citations and references of references are extracted.
    This information is used to generate a tree like network.
    """

    def __init__(
        self,
        verbose: bool = False,
        api_key="",
        use_expanded_target_references: bool = False,
    ):
        """
        __init__ instantiates citation network generator.

        :param verbose: forwarded to dimcli queries, defaults to False
        :type verbose: bool, optional
        :param api_key: dimensions.ai API key, tries to use dsl.ini if not existent, defaults to ""
        :type api_key: str, optional
        :param use_expanded_target_references: whether or not to use indirect connections
            (not through input node) to make network edges
        :type use_expanded_target_references: bool, optional
        """
        while not dimcli.login_status():
            try:
                dimcli.login(key=api_key)
            except HTTPError as e:
                if e.response.status_code == 401:
                    raise
                time.sleep(5)
                pass

        self.dsl: dimcli.Dsl = dimcli.Dsl()
        self._verbose: bool = verbose
        self.startDoi: Doi = ""
        self.stringClean = {r"\s": "__", "/": "_slash_", ":": "_colon_", r"\.": "_dot_"}
        self._make_hairball = use_expanded_target_references

    def fetchPubsByIDs(
        self, pubIDs: List[PubID], authors: bool = True
    ) -> Tuple[bool, pd.DataFrame]:
        """
        Fetch publications from dimcli using PubIDs defined by dimensions.ai.

        :param pubIDs: list of PubIDs (string type alias)
        :type pubIDs: List[PubID]
        :param authors: whether to fetch author information, defaults to True
        :type authors: bool, optional
        :return: status-bool (True if everything is okay), dataframe containing
            required information for input publications
        :rtype: Tuple[bool, pd.DataFrame]
        """
        if self._verbose:
            print(f"hi, this is fetchPubsByIDs() for pubs = {pubIDs}")

        if authors:
            query = f"""
                    search publications
                        where id in {json.dumps(pubIDs)}
                    return publications[id+doi+title+category_for+year
                        +reference_ids+authors+journal_title_raw+times_cited]
                    limit {len(pubIDs)}
                """
        else:
            query = f"""
                    search publications
                        where id in {json.dumps(pubIDs)}
                    return publications[id+doi+title+category_for+year
                        +reference_ids+journal_title_raw+times_cited]
                    limit {len(pubIDs)}
                """

        dsl_data = self.dsl.query(query, verbose=self._verbose)

        df = dsl_data.as_dataframe()

        try:
            df["target_refs"] = df["reference_ids"]
        except (TypeError, KeyError):
            return False, pd.DataFrame()

        # replace NaN with empty list
        df["target_refs"] = df["target_refs"].apply(
            lambda target_ref: [] if type(target_ref) == float else target_ref
        )

        if not authors:
            df["authors"] = [np.nan] * len(df)

        return True, df

    def fetchPubsByDois(
        self, dois: List[Doi], authors: bool = True
    ) -> Tuple[bool, pd.DataFrame]:
        """
        Fetch publications from dimcli using DOIs.

        :param dois: list of DOIs (string type alias)
        :type dois: List[Doi]
        :param authors: whether to fetch author information, defaults to True
        :type authors: bool, optional
        :return: status-bool (True if everything is okay), dataframe containing
            required information for input publications
        :rtype: Tuple[bool, pd.DataFrame]
        """
        if self._verbose:
            print(f"hi, this is fetchOrigin() for doi = {dois}")

        if authors:
            query = f"""
                    search publications
                        where doi in {json.dumps(dois)}
                    return publications[id+doi+title+category_for+year
                        +reference_ids+authors+journal_title_raw+times_cited]
                """
        else:
            query = f"""
                    search publications
                        where doi in {json.dumps(dois)}
                    return publications[id+doi+title+category_for+year
                        +reference_ids+journal_title_raw+times_cited]
                """

        dsl_data = self.dsl.query(query, verbose=self._verbose)

        df = dsl_data.as_dataframe()
        try:
            df["target_refs"] = df["reference_ids"]
        except (TypeError, KeyError):
            return False, pd.DataFrame()

        # replace NaN with empty list
        df["target_refs"] = df["target_refs"].apply(
            lambda target_ref: [] if type(target_ref) == float else target_ref
        )

        if not authors:
            df["authors"] = [np.nan] * len(df)

        return True, df

    def fetchCitations(
        self, pubIDs: List[PubID], authors: bool = False
    ) -> Tuple[bool, pd.DataFrame]:
        """
        Fetch citing publications for a list of publications using their PubIDs.

        :param pubIDs: list of PubIDs (string type alias)
        :type pubIDs: List[PubID]
        :param authors: whether to fetch author information, defaults to False
        :type authors: bool, optional
        :return: status-bool (True if everything is okay), dataframe containing
            required information for citing publications
        :rtype: Tuple[bool, pd.DataFrame]
        """
        if self._verbose:
            print(f"hi, this is fetchCitations() for pubs = {pubIDs}")

        dfs = []
        if math.ceil(len(pubIDs) / 512) > 1:
            __range__ = tqdm(range(math.ceil(len(pubIDs) / 512)))
        else:
            __range__ = range(math.ceil(len(pubIDs) / 512))

        for i in __range__:
            # dimcli queries are limited to 512 entites per list for `in` filtering
            offset = i * 512

            if authors:
                query = f"""
                        search publications
                            where reference_ids in {json.dumps(pubIDs[offset:offset+512])}
                        return publications[id+doi+title+category_for+year
                            +reference_ids+authors+journal_title_raw+times_cited]
                    """
            else:
                query = f"""
                        search publications
                            where reference_ids in {json.dumps(pubIDs[offset:offset+512])}
                        return publications[id+doi+title+category_for+year
                            +reference_ids+journal_title_raw+times_cited]
                    """

            dsl_data = self.dsl.query_iterative(query, verbose=self._verbose)
            tmp = dsl_data.as_dataframe()

            try:
                _ = tmp["reference_ids"]
            except (TypeError, KeyError):
                return False, pd.DataFrame()

            dfs.append(tmp)

        df = pd.concat(dfs)
        # intersection of input pubIDs and references of each publication
        df["target_refs"] = df["reference_ids"].apply(
            lambda row_refs: list(set(pubIDs) & set(row_refs))
        )

        if not authors:
            df["authors"] = [np.nan] * len(df)

        return True, df

    def fetchReferences(
        self, pubIDs: List[PubID], authors: bool = True
    ) -> Tuple[bool, pd.DataFrame]:
        """
        Fetch references for a list of publications using their PubIDs as defined by dimensions.ai.

        :param pubIDs: list of PubIDs (string type alias)
        :type pubIDs: List[PubID]
        :param authors: whether to fetch author information, defaults to True
        :type authors: bool, optional
        :return: status-bool (True if everything is okay), dataframe containing
            required information for references
        :rtype: Tuple[bool, pd.DataFrame]
        """
        if self._verbose:
            print(f"hi, this is fetchReferences() for pubs = {pubIDs}")

        dfs = []

        if math.ceil(len(pubIDs) / 512) > 1:
            __range__ = tqdm(range(math.ceil(len(pubIDs) / 512)))
        else:
            __range__ = range(math.ceil(len(pubIDs) / 512))

        # get references (PubID) of given PubIDs
        for i in __range__:
            # dimcli queries are limited to 512 entites per list for `in` filtering
            offset = i * 512

            query = f"""
                    search publications
                        where id in {json.dumps(pubIDs[offset:offset + 512])}
                    return publications[id+reference_ids]
                    limit 512
                """

            dsl_data = self.dsl.query(query, verbose=self._verbose)
            tmp = dsl_data.as_dataframe()

            try:
                _ = tmp["reference_ids"]
            except (TypeError, KeyError):
                return False, pd.DataFrame()

            dfs.append(tmp)

        df0 = pd.concat(dfs)

        # flatten list of references
        # List[List[PubID]] -> List[PubID]
        refs = [x for x in df0["reference_ids"].dropna().to_list() for x in x]

        # drop duplicates
        refs = list(set(refs))

        dfs = []
        if math.ceil(len(refs) / 512) > 1:
            __range__ = tqdm(range(math.ceil(len(refs) / 512)))
        else:
            __range__ = range(math.ceil(len(refs) / 512))

        for i in __range__:
            # dimcli queries are limited to 512 entites per list for `in` filtering
            offset = i * 512
            ok, df = self.fetchPubsByIDs(refs[offset : offset + 512], authors=authors)
            if ok:
                dfs.append(df)
            else:  # pragma: no cover
                # cannot be reached unless dimensions database is malicious
                return False, pd.DataFrame()

        return True, pd.concat(dfs)

    def run(
        self, doi: Doi, levels_ref: int = 2, levels_cite: int = 2, authors: bool = False
    ) -> Tuple[bool, pd.DataFrame]:
        """
        Generate citation network for a publication using its DOI.

        :param doi: input DOI (string type alias)
        :type doi: Doi
        :param levels_ref: number of levels for references, defaults to 2
        :type levels_ref: int, optional
        :param levels_cite: number of levels for citing publications, defaults to 2
        :type levels_cite: int, optional
        :param authors: whether to include author information, defaults to False
        :type authors: bool, optional
        :return: status-bool (True if everything is okay), dataframe containing
            required information for input publications, references and citing publications
        :rtype: Tuple[bool, pd.DataFrame]
        """
        if hasattr(self, "result_df"):
            return True, self.result_df

        self.startDoi = doi

        dfs = []

        print("level 0")
        ok, df_origin = self.fetchPubsByDois([doi], authors)
        if not ok:
            print(f"could not fetch publication for DOI {doi}")
            return False, pd.DataFrame()

        df_origin["level"] = 0
        dfs.append(df_origin)

        ok, dfs_cite = self._fetchCite(df_origin, levels_cite, authors)
        ok, dfs_ref = self._fetchRef(df_origin, levels_ref, authors)

        dfs.extend(dfs_cite + dfs_ref)

        self.result_df: pd.DataFrame = pd.concat(dfs).reset_index(drop=True)

        # cleaning
        # self.result_df = self.dropDuplicates(self.result_df)
        self.result_df["first_author"] = self.result_df["authors"].apply(
            lambda authors: authors[0]["last_name"] if type(authors) == list else ""
        )
        self.result_df["ref_count"] = self.result_df["reference_ids"].apply(
            lambda refs: len(refs) if type(refs) == list else None
        )
        self.result_df.index = self.result_df["id"]

        self.main_node = df_origin.iloc[0].copy()
        if type(self.main_node["authors"]) == list:
            self.main_node["first_author"] = self.main_node["authors"][0]["last_name"]
        else:
            self.main_node["first_author"] = ""

        self.result_df["main_category_for"] = self.result_df["category_for"].apply(
            lambda c: [
                x["name"]
                for x in filter(lambda dict_: re.match(r"^\d\d\s", dict_["name"]), c)
            ][0]
            if type(c) == list
            else ""
        )

        # replace NaN in reference_ids with empty list
        self.result_df["reference_ids"] = self.result_df["reference_ids"].apply(
            lambda target_ref: [] if type(target_ref) == float else target_ref
        )

        # include expanded target refs
        # (intersection of references of a paper and all listed publications,
        #  e.g. main_node cites A, Y cites main_node and A
        #    -> target_refs does not contain connection from Y to A)
        all_pubs = set(self.result_df.index)
        self.result_df["expanded_target_refs"] = self.result_df["reference_ids"].apply(
            lambda reference_ids: list(all_pubs.intersection(reference_ids))
        )

        return True, self.result_df

    def _fetchCite(
        self, df_origin: pd.DataFrame, levels: int, authors: bool
    ) -> Tuple[bool, List[pd.DataFrame]]:
        dfs_cite = []

        pubIDs = df_origin["id"].to_list()
        for i in range(levels):
            print(f"level {i + 1}, fetching citations for {len(pubIDs)} publications")
            ok, tmp = self.fetchCitations(pubIDs, authors)
            if ok:
                pubIDs = tmp["id"].to_list()
                tmp["level"] = i + 1
                dfs_cite.append(tmp)
            else:  # pragma: no cover
                # cannot be reached unless dimensions database is malicious
                return False, pd.DataFrame()
        return True, dfs_cite

    def _fetchRef(
        self, df_origin: pd.DataFrame, levels: int, authors: bool
    ) -> Tuple[bool, List[pd.DataFrame]]:
        dfs_ref = []
        pubIDs = df_origin["id"].to_list()
        for i in range(levels):
            print(
                f"level {(i + 1) * (-1)}, fetching references for {len(pubIDs)} publications"
            )
            ok, tmp = self.fetchReferences(pubIDs, authors)
            if ok:
                pubIDs = tmp["id"].to_list()
                tmp["level"] = (i + 1) * (-1)
                dfs_ref.append(tmp)
            else:  # pragma: no cover
                # cannot be reached unless dimensions database is malicious
                return False, pd.DataFrame()
        return True, dfs_ref

    def _makeCompatibleRefDf(
        self, df: pd.DataFrame, use_expanded: bool = False
    ) -> pd.DataFrame:
        """
        Reformat references dataframe to match prior versions formatting.

        :param df: dataframe as generated by .fetchReferences()
        :type df: pd.DataFrame
        :return: compatible dataframe
        :rtype: pd.DataFrame
        """
        levels_ref = min(df["level"])

        # flatten references
        if use_expanded:
            target_ref_type = "expanded_target_refs"
        else:
            target_ref_type = "target_refs"

        ref_tuples = {
            (row["id"], ref)
            for _, row in df.query(f"{levels_ref} < level <= 0").iterrows()
            for ref in row[target_ref_type]
        }

        df = df[~df.index.duplicated(keep="first")]

        refs = []
        for (source_id, target_id) in ref_tuples:
            source = df.loc[source_id]
            target = df.loc[target_id]

            refs.append(
                {
                    "type": "reference",
                    "sourceYear": source["year"],
                    "sourceDOI": source["doi"],
                    "sourcePubID": source["id"],
                    "sourceJournal": source["journal_title_raw"],
                    "targetFull": "",
                    "targetYear": target["year"],
                    "targetDOI": target["doi"],
                    "targetPubID": target["id"],
                    "targetrefCount": target["ref_count"],
                    "targetis_ref_byCount": target["times_cited"],
                    "targettitleStr": target["title"],
                    "targetFirstAuthor": target["first_author"],
                    "targetJournal": target["journal_title_raw"],
                    "targetSubject": target["category_for"],
                }
            )

        return pd.DataFrame(refs)

    def _makeCompatibleCiteDf(
        self, df: pd.DataFrame, use_expanded: bool = False
    ) -> pd.DataFrame:
        """
        Reformat citation dataframe to match prior versions formatting.

        :param df: dataframe as generated by .fetchCitations()
        :type df: pd.DataFrame
        :return: compatible dataframe
        :rtype: pd.DataFrame
        """
        levels_cite = max(df["level"])

        # flatten citations
        if use_expanded:
            target_ref_type = "expanded_target_refs"
        else:
            target_ref_type = "target_refs"

        cite_tuples = {
            (row["id"], ref)
            for _, row in df.query(f"{levels_cite} >= level > 0").iterrows()
            for ref in row[target_ref_type]
        }

        df = df[~df.index.duplicated(keep="first")]

        cites = []
        for (source_id, target_id) in cite_tuples:
            source = df.loc[source_id]
            target = df.loc[target_id]

            cites.append(
                {
                    "type": "citation",
                    "targetPubID": target["id"],
                    "targetYear": target["year"],
                    "targetDOI": target["doi"],
                    "targetJournal": target["journal_title_raw"],
                    "sourceYear": source["year"],
                    "sourceDOI": source["doi"],
                    "sourcePubID": source["id"],
                    "sourcerefCount": source["ref_count"],
                    "sourceis_ref_byCount": source["times_cited"],
                    "sourcetitleStr": source["title"],
                    "sourceFirstAuthor": source["first_author"],
                    "sourceJournal": source["journal_title_raw"],
                    "sourceSubject": source["category_for"],
                }
            )

        return pd.DataFrame(cites)

    def makeCompatibleDf(self) -> Tuple[bool, pd.DataFrame]:
        """
        Reformat dataframe to match prior versions formatting.

        :param df: dataframe as generated by .run()
        :type df: pd.DataFrame
        :return: compatible dataframe
        :rtype: pd.DataFrame
        """
        if not hasattr(self, "result_df"):
            print("you gotta run .run() first")
            return False, pd.DataFrame()

        if hasattr(self, "compatible_result_df"):
            return True, self.compatible_result_df

        df_ref = self._makeCompatibleRefDf(self.result_df, self._make_hairball)
        df_cite = self._makeCompatibleCiteDf(self.result_df, self._make_hairball)

        self.compatible_result_df: pd.DataFrame = pd.concat(
            [df_cite, df_ref], ignore_index=True
        ).fillna("")

        return True, self.compatible_result_df

    def runCompatible(
        self,
        doi: Doi,
        level: int = 2,
        direct: str = "both",
        debug: bool = False,
    ) -> Tuple[bool, pd.DataFrame]:
        """
        Wrap .run() with same parameters and outputs as prior versions.

        :param doi: input DOI (string type alias)
        :type doi: Doi
        :param level: number of levels to fetch, defaults to 2
        :type level: int, optional
        :param direct: direction of search (either "ref", "cite" or "both"), defaults to "both"
        :type direct: str, optional
        :param debug: [description], defaults to False
        :type debug: bool, optional
        :return: [description]
        :rtype: Tuple[bool, pd.DataFrame]
        """
        if direct == "ref":
            ok, df = self.run(doi, levels_ref=level, levels_cite=0)
        elif direct == "cite":
            ok, df = self.run(doi, levels_ref=0, levels_cite=level)
        elif direct == "both":
            ok, df = self.run(doi, levels_ref=level, levels_cite=level)
        else:
            print("provide proper direction of search (either `ref`, `cite` or `both`)")
            return False, pd.DataFrame()

        if not ok:
            return False, pd.DataFrame()

        ok, comp_df = self.makeCompatibleDf()
        if ok:
            return True, comp_df
        else:
            return False, pd.DataFrame()

    def _nodeDict(self, row: pd.Series) -> Dict:
        # row = row.fillna("")

        if row["doi"].lower() == self.startDoi.lower():
            inputDOI = "True"
        else:
            inputDOI = "False"
        res = {
            #   "label": nodeName,
            #   "x": 0,
            #   "y": 0,
            "id": row["id"],
            "attributes": {
                # "name": nodeName,
                "title": row["title"],
                "doi": row["doi"],
                "nodeyear": row["year"],
                "ref-by-count": row["times_cited"],
                "is_input_DOI": inputDOI,
                "category_for": row["main_category_for"],
                "level": row["level"],
            },
            #   "color": "rgb(0,0,0)",
            #   "size": 10
        }
        return res

    def _edgeDict(self, row: pd.Series) -> Dict:
        # row = row.fillna("")

        res = {
            "source": row["sourcePubID"],
            "target": row["targetPubID"],
            #   "id": idx,
            "attributes": {"year": row["sourceYear"], "type": row["type"]},
            #   "color": "rgb(0,0,0)",
            #   "size": 1
        }
        return res

    def _createFilename(self, ext: str = "json") -> FilePath:
        filename = self.startDoi
        date = datetime.datetime.now().strftime("%Y-%m-%d")
        for key, val in self.stringClean.items():
            filename = re.sub(key, val, filename)
        if self._make_hairball:
            path = f"{self.main_node['first_author']}_{filename}_date_{date}_hairball.{ext}"
        else:
            path = f"{self.main_node['first_author']}_{filename}_date_{date}.{ext}"
        return path

    def createJSON(self, outputPath: FilePath = "./out") -> Tuple[bool, FilePath]:
        """
        Create JSON file on disk containing network as lists of nodes and edges for visualization.

        :param outputPath: output directory, defaults to "./out"
        :type outputPath: FilePath, optional
        :return: status-bool (True if everything is okay), path of JSON file
        :rtype: Tuple[bool, FilePath]
        """
        if not hasattr(self, "result_df"):
            print("You need to use .run() first to create some data to write.")
            return False, ""

        if not hasattr(self, "compatible_result_df"):
            self.makeCompatibleDf()

        allNodes = [
            x
            for _, x in self.result_df[~self.result_df.index.duplicated()]
            .fillna("")
            .iterrows()
        ]
        allRows = [x for x in self.compatible_result_df.fillna("").iterrows()]

        outputPath = os.path.abspath(outputPath)
        if not os.path.exists(outputPath):
            os.mkdir(outputPath)

        with open(f"{outputPath}/{self._createFilename()}", "w") as outFile:
            # write nodes
            outFile.write('{\n  "nodes": [\n')

            # write nodes from compatible_result_df/allNodes
            while allNodes:
                node = allNodes.pop()
                if len(allNodes) == 0:
                    outFile.write(json.dumps(self._nodeDict(node)) + "\n")
                else:
                    outFile.write(json.dumps(self._nodeDict(node)) + ",\n")

            # write edges
            outFile.write('  ],\n  "edges":[')
            while allRows:
                idx, edge = allRows.pop()
                if len(allRows) == 0:
                    outFile.write(json.dumps(self._edgeDict(edge)) + "\n")
                else:
                    x = self._edgeDict(edge)
                    outFile.write(json.dumps(x) + ",\n")
            outFile.write("  ]\n}")

        return True, f"{outputPath}/{self._createFilename()}"

    def logout(self) -> None:
        """
        Dimcli logout.
        """
        dimcli.logout()
