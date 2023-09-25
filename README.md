# Grid and place cells from Koopman and DMD

Some experiments with understanding grid cells from the viewpoint of transfer operators. Notes can be found here: https://www.overleaf.com/project/6000b373a7589fa06914fda9. If you don't have access, contact me.

## Getting started

To get started with the code, you need to install the `neurodmd` package. To do so, make sure to switch to the main repo folder. The first step is to make a virtual environment and activate it.

#### Using conda

```sh
conda create -n bio-koop python=3.9
conda activate bio-koop
```

#### Using venv

```sh
python -m venv env
source venv/bin/activate
```

#### Install (either conda or venv)

```sh
pip install -e .
```

This makes an "editable" install so that any changes to the code take effect instantly.

#### Test install

```sh
pytest test
```

All tests should run without errors. If there are errors, double check your environment. If there are still errors, make a new environment following the instructions above. If there are still errors after that, file a bug, including as much information as possible.

## Examples

Find example code in the `sandbox` folder.

## Components

### One-dimensional place cell simulator

The [`PlaceGridMotionSimulator`](src/neurodmd/bump_simulator.py) can be used to simulate motion across a discrete set of place cells, potentially with periodic boundary conditions. It uses a Gaussian "bump" of activity in the non-periodic case, and a von Mises in the periodic case.

### Place+grid simulations with single controller

Two classes implement the interplay between place and grid cells based on transfer operators, as outlined in [the Overleaf](https://www.overleaf.com/project/6000b373a7589fa06914fda9).

One, [`PlaceGridSystemNonBioCplx`](src/neurodmd/naive_model_cplx.py), uses complex numbers, greatly simplifying parts of the implementation, but adding some complexity around the constraints needed to keep outputs real.

There is also an implementation based on real numbers, [`PlaceGridSystemNon`](src/neurodmd/naive_model.py), which is typically more stable.

In both of these cases, there is only one dynamics that is being simulated, for instance, translation. The direction of translation (left vs. right) is controlled using the sign of the magnitude parameter.

### Multi-operator simulations

In real environments, non-linear translation operators are not necessarily invertible (e.g., attempting to move right into a wall products no motion, and the inverse of "no motion" is not move left away from the wall). Thus, it is more reasonable to use at least two operators, one for leftward motion and one for rightward.

The [`PlaceGridMultiNonBio`](src/neurodmd/multi_model.py) class handles such "multi-operator" simulations.

### Attempt at using similarity matching

An attempt to use similarity matching to do the training (as opposed to backtracking, which is what all the others models do) is in [`EquivariantSM`](src/neurodmd/sm_model.py). This has not really been successful.


### Scripts for facilitating training and testing

These are defined in [training.py](src/neurodmd/training.py).

### Step-wise scheduler

A simple learning-rate scheduler that allows for stepwise changes in learning rate is the [`StepwiseScheduler`](src/neurodmd/utils.py).

## Issues?

If you run into any trouble, file a bug.

