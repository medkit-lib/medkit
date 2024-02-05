# Installation

The medkit package supports a version of python >= 3.8.

## Install an official version

:::{important}
From 0.4.1 version, medkit package (named **medkit-lib**) is available on
[PyPi](https://pypi.org/project/medkit-lib/).
:::

Releases are published on <https://github.com/medkit-lib/medkit/releases>.

To install medkit :

```
# Install medkit with required dependencies
python -m pip install 'medkit-lib'

# Install medkit with all extra dependencies
python -m pip install 'medkit-lib[all]'
```

:::{note}
We recommend to install the medkit package in a virtual or conda environment.
:::

Here is an example with conda:
```
conda create -n medkit-tuto python=3.8
conda activate medkit-tuto
pip install 'medkit-lib[all]'
```

## Install a development version

If you want to contribute, clone the `medkit` repository locally:
  - SSH: `git clone  git@github.com:medkit-lib/medkit.git`
  - HTTPS: `git clone https://github.com/medkit-lib/medkit.git`

[Poetry](https://python-poetry.org) is used for managing dependencies and
packaging medkit.

```shell
cd medkit
poetry install
```

If you want to also install the extras dependencies, you may use:
```shell
poetry install --all-extras
```

For documentation:
```shell
poetry install --with docs

make docs # for generating documentation
```

Then, a `.venv` folder is created at the root of the project. To activate the
virtual environment:
```shell
source .venv/bin/activate
```

To make sure everything is set up properly, you may run the tests :

```
# For unit/small tests
pytest -v tests/unit
```

## Troubleshooting

Sometimes, for documentation and/or testing, you may need some additional packages:

```
sudo apt-get install -y gcc g++ libsndfile1 graphviz
```
You may also refer to CI file (e.g., .gitlab-ci.yml) for up-to-date information.
