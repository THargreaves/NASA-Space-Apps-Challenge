# NASA-Space-Apps-Challenge

## Setup

### Environment Management

To create an environment from the requirements file, run `conda env create -f environment.yml`.

To update this environment use `conda env update -f environment.yml`, adding the `--prune` flag if packages have been removed.

To use the environment with Jupyter, activate the environment with `conda activate space-app-env` and run `python -m ipykernel install --user --name space-app-env --display-name "Space App Environment"`
