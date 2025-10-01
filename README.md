https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder

Write a pyproject.toml: https://packaging.python.org/en/latest/guides/writing-pyproject-toml/

https://packaging.python.org/en/latest/tutorials/packaging-projects/#choosing-build-backend

https://packaging.python.org/en/latest/guides/packaging-namespace-packages/

## Installation

Toying around with the experiments requires installing the package, following a [setuptools tutorial](https://setuptools.pypa.io/en/latest/userguide/development_mode.html), or our instructions below.

We suggest limiting the damage to an aptly named virtual environment:
```sh
python -m venv .venv --prompt symplearn
```
creating a `.venv` folder containing a local Python executable.
This must then be activated using either 
```sh
source .venv/bin/activate
```
on Unix systems, or `.venv\Scripts\activate` on Windows.
No Windows system was involved in the development of this project: some unexpected behaviour may appear.

For a "read-only" use in the `experiments` folder, install the package with the line
```sh
pip install .
```
If you wish to toy around in the source files, install it in editable mode, with the line
```sh
pip install -e .
```

## Dependencies

Be warned that this code uses `torch.func` of version `2.5.1`.
As mentionned by [the corresponding documentation](https://pytorch.org/docs/2.5/func.html), this library is still in beta, therefore using the wrong version might result in unexpected behaviour.
