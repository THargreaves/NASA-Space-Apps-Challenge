# NASA-Space-Apps-Challenge
## Prerequisities
- A Conda installation: https://docs.conda.io/en/latest/miniconda.html

## Our challenge
- Determine changes in rainforest cover over time
	- Think this is right? I was semi-in the call when it was being discussed

## Presenting the project
- A [Jupyter Notebook](https://jupyter.org/) to demonstrate our model

## Components
- Machine Learning model to determine rainforest cover, given a satellite image
	- It must also identify that clouds do not contribute to the overall cover
- Use of [SciPy](https://www.scipy.org) or [MatLab](https://www.mathworks.com/products/matlab.html) to help with image processing
- Use of [SNAP and its Python Snappy API](https://towardsdatascience.com/getting-started-with-snap-toolbox-in-python-89e33594fa04) to help explore and find relevant satellite data
- Our data - I'm not too sure where this is coming from, but my understanding is that we have all relevant and sufficient data to work on this model

## Code environments and libraries
- Using [conda](https://docs.conda.io/en/latest/) (Miniconda is sufficient) to help manage our Python environment, and list our dependencies and Python version
	- Have an `environment.yml`
	- Create the local environment using `conda env create -f environment.yml`
	- Update if need-be by using `conda env update -f environment.yml`, adding `--prune` flag if packages have been removed
	- Activate this environment using `conda activate space-app-env` and run `python -m ipykernel install --user --name space-app-env --display-name "Space App Environment"`
- Connecting this with a Jupyter Notebook
	- Use `jupyter notebook` in the terminal to start this
	- Since it is a dependency in our `conda` environment, it will have already been installed

## The `git` environment
- If unfamiliar with `git`, read the `GIT_BEST_PRACTICE.md` file, which will give enough information for the day's use of `git`
- Branching is probably unnecessary for this task, since it is quite sequential
