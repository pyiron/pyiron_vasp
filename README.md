# vaspparser

[![Pipeline](https://github.com/pyiron/vaspparser/actions/workflows/pipeline.yml/badge.svg)](https://github.com/pyiron/vaspparser/actions/workflows/pipeline.yml)
[![codecov](https://codecov.io/gh/pyiron/vaspparser/graph/badge.svg?token=PWWLjnbDJz)](https://codecov.io/gh/pyiron/vaspparser)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pyiron/vaspparser/HEAD)

Parser for the Vienna Ab initio Simulation Package (VASP)

## Installation 
Via pip
```
pip install vaspparser
```

Via conda
```
conda install -c conda-forge vaspparser
```

## Usage
Parse an directory with VASP output files 
```python
from vaspparser.vasp.output import parse_vasp_output

output_dict = parse_vasp_output(working_directory="path/to/calculation")
```

## Documentation 

* [vaspparser](https://vaspparser.readthedocs.io/en/latest/README.html)
  * [Installation](https://vaspparser.readthedocs.io/en/latest/README.html#installation)
  * [Usage](https://vaspparser.readthedocs.io/en/latest/README.html#usage)
* [Demo](https://vaspparser.readthedocs.io/en/latest/demo.html)
  * [Parsing a full VASP run](https://vaspparser.readthedocs.io/en/latest/demo.html#reading-and-writing-structures)
  * [](https://vaspparser.readthedocs.io/en/latest/demo.html#parsing-a-full-vasp-run)
  * [Parsing individual output files](https://vaspparser.readthedocs.io/en/latest/demo.html#parsing-individual-output-files)
  * [Electronic structure: DOS and band gap](https://vaspparser.readthedocs.io/en/latest/demo.html#electronic-structure-dos-and-band-gap)
  * [Volumetric data: charge density](https://vaspparser.readthedocs.io/en/latest/demo.html#volumetric-data-charge-density)
  * [Bader charge analysis](https://vaspparser.readthedocs.io/en/latest/demo.html#bader-charge-analysis)
* [Interface](https://vaspparser.readthedocs.io/en/latest/api.html)
