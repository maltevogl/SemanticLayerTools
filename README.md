## SemanticLayerTools

Collects tools to create semantic layers in the socio-epistemic networks framework. Source material can be any structured corpus with metadata of authors, time, and at least one text column.

Documentation is available on [ReadTheDocs](https://semanticlayertools.readthedocs.io/).

## Installation

tl;dr Use pip

~~~bash
pip install semanticlayertools
~~~

Consider using a clean virtual environment to keep your main packages separated.
Create a new virtual environment and install the package

~~~bash
python3 -m venv env
source env/bin/activate
pip install semanticlayertools
~~~

To use some sentence embedding utility functions please install with the
`embeddml` option

~~~bash
pip install semanticlayertools[embeddml]
~~~

## Testing

Tests can be run by installing the _dev_ requirements and running `tox`.

~~~bash
pip install semanticlayertools[dev]
tox
~~~

## Building documentation

The documentation is build using _sphinx_. Install with the _dev_ option and run

~~~bash
pip install semanticlayertools[dev]
tox -e docs
~~~

## Funding information

The development is part of the research project [ModelSEN](https://modelsen.mpiwg-berlin.mpg.de)

> Socio-epistemic networks: Modelling Historical Knowledge Processes,

in Department I of the Max Planck Institute for the History of Science 
and funded by the Federal Ministry of Education and Research, Germany (Grant No. 01 UG2131).
