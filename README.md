# vaspparser

[![Pipeline](https://github.com/pyiron/vaspparser/actions/workflows/pipeline.yml/badge.svg)](https://github.com/pyiron/vaspparser/actions/workflows/pipeline.yml)
[![codecov](https://codecov.io/gh/pyiron/vaspparser/graph/badge.svg?token=PWWLjnbDJz)](https://codecov.io/gh/pyiron/vaspparser)

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
