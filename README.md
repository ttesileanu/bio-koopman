# Grid and place cells from Koopman and DMD

Some experiments with understanding grid cells from the viewpoint of transfer operators. Notes can be found here: https://www.overleaf.com/project/6000b373a7589fa06914fda9. If you don't have access, contact me.

## Getting started

To get started with the code, you need to install the `neurodmd` package. To do so, make sure to switch to the main repo folder. The first step is to make a virtual environment and activate it.

**Using conda.**

```sh
conda create -n bio-koop python=3.9
conda activate bio-koop
```

**Using venv.**

```sh
python -m venv env
source venv/bin/activate
```

**Install (either conda or venv).**

```sh
pip install -e .
```

This makes an "editable" install so that any changes to the code take effect instantly.

**Test install.** Switch to the `test` folder and run

```sh
pytest .
```

All tests should run without errors. If there are errors, double check your environment. If there are still errors, make a new environment following the instructions above. If there are still errors after that, file a bug, including as much information as possible.

## Examples

Find example code in the `sandbox` folder.

## Issues?

If you run into any trouble, file a bug.

