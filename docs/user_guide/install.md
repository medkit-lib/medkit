# Installation

## Supported Python versions

`medkit` requires a distribution of Python with a minimum version of 3.8.

:::{note}
It is recommended to install `medkit` in a virtual or conda environment.
:::

## Install an official version

Releases of `medkit` are published on [PyPI](https://pypi.org/project/medkit-lib/)
under the name **medkit-lib**.

To install `medkit` with basic functionalities:

```console
python -m pip install 'medkit-lib'
```

To install `medkit` with all functionalities:

```console
python -m pip install 'medkit-lib[all]'
```

Using `conda`, `mamba` or `micromamba`:

```console
conda create -n medkit python=3.8
conda activate medkit
pip install 'medkit-lib[all]'
```

## Install a development version

To start contributing, first clone the `medkit` [repository](https://github.com/medkit-lib/medkit.git) locally:

Using Git:

```console
git clone https://github.com/medkit-lib/medkit.git
```

or the GitHub CLI:

```console
gh repo clone medkit-lib/medkit.git
```

This project uses [Hatch](https://hatch.pypa.io/) to manage its dependencies.
Please follow its [installation instructions](https://hatch.pypa.io/latest/install/).

The project can be deployed in a virtual environment and tested with:

```console
hatch run test:no-cov tests/unit
```

The corresponding documentation can be built with:

```console
hatch run docs:build
```

Or served with interactive reloading with:

```console
hatch run docs:serve
```

Code linting and formatting can be applied with:

```console
hatch fmt
```

Additional checks may be run using [pre-commit](https://pre-commit.com/):

```console
pre-commit run --all-files
```
