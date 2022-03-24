Utility functions for visualizations
====================================

The usage of some of these methods requires installing the package with
the extra requirements for text embedding and clustering

.. code-block:: bash
   :linenos:

   pip install semanticlayertools[embeddml]


Representing temporal cluster evolution with a streamgraph
**********************************************************

This utility function is meant to support the visualization of calculated
temporal clusters. Parameters to vary are the smoothing (bool) and the minimal
cluster size to consider (default=1000).

.. code-block:: python
   :linenos:

   streamgraph(file, smooth, minClusterSize)


Embedding a text corpus in 2 dimensions
***************************************

Meant to be used to visualize a corpus on 2D by embedding a text column using
the SentenceTransformer approach of SBERT and UMAP. Time consuming method!

.. code-block:: python
   :linenos:

   embeddedTextPlotting(infolderpath, columnName, outpath, umapNeighors)

.. seealso ::
    `SBERT docs <https://www.sbert.net/index.html>`_

    `UMAP docs <https://umap-learn.readthedocs.io/en/latest/index.html>`_


Clustering texts using SentenceEmbedding
****************************************

Similar to the above method but extended to help finding large scale structures
of a given text corpus. Similar to topic modelling, in addition makes use of
HDBSCAN clustering. Reuses previously generated embedding of corpus.

.. code-block:: python
   :linenos:

   embeddedTextClustering(
       infolderpath, columnName, embeddingspath, outpath,
       umapNeighors, umapComponents, hdbscanMinCluster
   )

.. seealso ::
    `HDBSCAN docs <https://hdbscan.readthedocs.io>`_


Generate citation and reference tree graph
******************************************

Using the Dimensions AI dataset, this routine generates a structure
starting from a source publications, that represents its references and their
references as well as its citations and their citations. With this means,
visualizations of it show academic roots and conduits and can display
disciplinary pathways.

.. automodule:: semanticlayertools.visual.citationnet
  :members:
  :private-members:
  :undoc-members:
