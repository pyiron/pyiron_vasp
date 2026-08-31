# vaspparser

[![Pipeline](https://github.com/pyiron/vaspparser/actions/workflows/pipeline.yml/badge.svg)](https://github.com/pyiron/vaspparser/actions/workflows/pipeline.yml)
[![codecov](https://codecov.io/gh/pyiron/vaspparser/graph/badge.svg?token=PWWLjnbDJz)](https://codecov.io/gh/pyiron/vaspparser)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pyiron/vaspparser/HEAD)

`vaspparser` reads the output files of a [VASP](https://www.vasp.at) calculation -
`OUTCAR`, `vasprun.xml`, `OSZICAR`, `CONTCAR`/`POSCAR`, `PROCAR`, `CHGCAR`, `LOCPOT`,
`AECCAR0`/`AECCAR2` and `REPORT` - and turns them into plain Python data: an
[ASE](https://wiki.fysik.dtu.dk/ase/) `Atoms` object for the structure, numpy arrays for
energies/forces/charge densities, and nested dictionaries for everything else. There is no
VASP-specific object model to learn - if you already know what a `vasprun.xml` or a
`CHGCAR` contains, `vaspparser` gets it into your Python session with minimal ceremony.

## Installation

Via pip
```
pip install vaspparser
```

Via conda
```
conda install -c conda-forge vaspparser
```

## Quick start

For a finished calculation, `parse_vasp_output()` collects everything that is available
in the working directory into a single hierarchical dictionary:

```python
from vaspparser.vasp.output import parse_vasp_output

output_dict = parse_vasp_output(working_directory="path/to/calculation")
list(output_dict.keys())
# ['description', 'generic', 'structure', 'electronic_structure', 'outcar', ...]

output_dict["generic"]["energy_tot"][-1]  # final total energy in eV
```

`vasprun.xml` is preferred when present (it is the most complete and most precise
source), with `OUTCAR` used as a fallback or to fill in quantities `vasprun.xml` does not
report (stresses, elastic constants, ...). `LOCPOT`/`CHGCAR` are parsed when found, and
Bader charges are computed automatically when `AECCAR0`/`AECCAR2` are present.

If you only need one piece of information, or want more control, the individual parsers
below can be used directly instead.

## Features

### Structure I/O

Read a `POSCAR`/`CONTCAR` into an ASE `Atoms` object, and write one back out:

```python
from vaspparser.vasp.structure import read_atoms, write_poscar, vasp_sorter

structure = read_atoms("POSCAR", species_list=["Fe"])
write_poscar(structure, filename="POSCAR_new")

# VASP groups atoms by species; vasp_sorter() gives the permutation that does that
vasp_sorter(structure)
```

### Individual output files

Each file also has a parser that can be used on its own, e.g. for scripting against a
single quantity without parsing the full directory:

```python
from vaspparser.vasp.parser.oszicar import Oszicar
from vaspparser.vasp.parser.outcar import Outcar
from vaspparser.vasp.vasprun import Vasprun

oszicar = Oszicar()
oszicar.from_file(filename="OSZICAR")
oszicar.parse_dict["energy_pot"]  # energy at every electronic/ionic step

outcar = Outcar()
outcar.from_file(filename="OUTCAR")
outcar.parse_dict["fermi_level"]

vasprun = Vasprun()
vasprun.from_file(filename="vasprun.xml")
vasprun.vasprun_dict["total_fr_energies"]
```

### Electronic structure: DOS and band gap

`Vasprun.get_electronic_structure()` (or the `PROCAR` parser below) returns an
`ElectronicStructure` object holding eigenvalues and occupancies for every k-point and
spin channel, from which the band gap and density of states (DOS) can be computed:

```python
electronic_structure = vasprun.get_electronic_structure()
electronic_structure.get_band_gap()
# {0: {'band_gap': 0.62, 'vbm': {...}, 'cbm': {...}}}

dos = electronic_structure.get_dos(n_bins=200)
dos.plot_total_dos()  # requires matplotlib
```

### Atom- and orbital-projected DOS (`PROCAR`)

The `PROCAR` file holds atom- and orbital-projected band/DOS information, useful for
analysing the chemical character of bands:

```python
from vaspparser.vasp.procar import Procar

procar_es = Procar().from_file(filename="PROCAR")
band = procar_es.kpoints[0].bands[0][0]
band.atom_resolved_dos  # contribution of each atom to this band
band.orbital_resolved_dos  # contribution of each orbital (s, p, d, ...)
```

### Volumetric data: charge density and electrostatic potential

`CHGCAR`/`LOCPOT`-type files describe a quantity on a real-space grid.
`VaspVolumetricData` parses them and offers helpers such as averaging along one lattice
direction (handy for work-function or interface analysis):

```python
from vaspparser.vasp.volumetric_data import VaspVolumetricData

charge_density = VaspVolumetricData()
charge_density.from_file(filename="CHGCAR", normalize=True)
charge_density.total_data.shape  # (Nx, Ny, Nz)
charge_density.get_average_along_axis(
    ind=2
)  # average over the ab-plane, as a function of c
```

### Bader charge analysis

Bader analysis partitions the charge density into atomic basins. Running it requires the
external [`bader`](http://theory.cm.utexas.edu/henkelman/code/bader) executable; `vaspparser`
builds its `CHGCAR`-style inputs from `AECCAR0`/`AECCAR2` and parses the result:

```python
from vaspparser.dft.bader import Bader

bader = Bader(structure=structure, working_directory="path/to/calculation")
charges, volumes = bader.compute_bader_charges()
```

## Documentation

* [vaspparser](https://vaspparser.readthedocs.io/en/latest/README.html)
  * [Installation](https://vaspparser.readthedocs.io/en/latest/README.html#installation)
  * [Quick start](https://vaspparser.readthedocs.io/en/latest/README.html#quick-start)
  * [Features](https://vaspparser.readthedocs.io/en/latest/README.html#features)
* [Demo](https://vaspparser.readthedocs.io/en/latest/demo.html)
  * [Reading and writing structures](https://vaspparser.readthedocs.io/en/latest/demo.html#reading-and-writing-structures)
  * [Parsing a full VASP run](https://vaspparser.readthedocs.io/en/latest/demo.html#parsing-a-full-vasp-run)
  * [Parsing individual output files](https://vaspparser.readthedocs.io/en/latest/demo.html#parsing-individual-output-files)
  * [Electronic structure: DOS and band gap](https://vaspparser.readthedocs.io/en/latest/demo.html#electronic-structure-dos-and-band-gap)
  * [Volumetric data: charge density](https://vaspparser.readthedocs.io/en/latest/demo.html#volumetric-data-charge-density)
  * [Bader charge analysis](https://vaspparser.readthedocs.io/en/latest/demo.html#bader-charge-analysis)
* [Interface](https://vaspparser.readthedocs.io/en/latest/api.html)
