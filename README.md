# Codebase for Drone Team

Template from tbp.monty paper. Uses tbp.monty version 0.5.0. 

## Getting Started

- Install Tello app from iOS or Android (Optional, but good to get a sense of what the Drone can do). 
- Python Interfaces to DJI Tello
	- [DJITelloPy](https://github.com/damiafuentes/DJITelloPy)
	- Python 2 Project: [Tello-Python](https://github.com/dji-sdk/Tello-Python)
		- Will likely not use due to Python 2 but may be helpful to look at scripts like `Tello_Video(With_Pose_Recognition)`
	- Drone Programming with Python: [Youtube 3.5 hours](https://www.youtube.com/watch?v=LmEcyQnfpDA)
- Very handy resources
    - Tello User Manual (In drone channel on Slack)
- [Tello SDK Manual](https://dl-cdn.ryzerobotics.com/downloads/Tello/Tello%20SDK%202.0%20User%20Guide.pdf)

## Working with Monty
- Setup something similar to the Paper Repo Template (to use a certain version of Monty for reproducibility).
- **Notes**: 
	- When working with drone, computer must be connected to Wifi: `TELLO-xxxx` (there will be no internet access)
	- Bluetooth must be off (at least true for my Mac)
- Update dependencies:
	- `pip install djitellopy`

### DJITelloPy Notes
```
tello = Tello()

# Output
[INFO] tello.py - 129 - Tello instance was initialized. Host: '192.168.10.1'. Port: '8889'.
```

## Make it yours

After copying the template, you need to address the following TODOs.

### `environment.yml`

- Update project `name`.
- Update `thousandbrainsproject::tbp.monty` version.
- Add any other dependencies.

### `pyproject.toml`

- Update the project `description`
- Update the project `name`
- Update the `Repository` and `Issues` URLs

### Delete template images

- Delete `delete_me.png`
- Delete `delete_me_too.png`

### `README.md`

- Update for your project

### Recommendations

For a cleaner project commit history, go to your repository settings and in the Pull Requests section, only "Allow squash merging". It also helps to set your default commit message to the "Pull request title" option.

![Pull Request Settings](./delete_me_too.png)

## Installation

The environment for this project is managed with [conda](https://www.anaconda.com/download/success).

To create the environment, run:

### ARM64 (Apple Silicon) (zsh shell)
```
conda env create -f environment.yml --subdir=osx-64
conda init zsh
conda activate paper # TODO: Update to your paper's name
conda config --env --set subdir osx-64
```

### ARM64 (Apple Silicon) (bash shell)
```
conda env create -f environment.yml --subdir=osx-64
conda init
conda activate paper # TODO: Update to your paper's name
conda config --env --set subdir osx-64
```

### Intel (zsh shell)
```
conda env create -f environment.yml
conda init zsh
conda activate paper # TODO: Update to your paper's name
```

### Intel (bash shell)
```
conda env create -f environment.yml
conda init
conda activate paper # TODO: Update to your paper's name
```

## Experiments

Experiments are defined in the `configs` directory.

After installing the environment, to run an experiment, run:

```bash
python run.py -e <experiment_name>
```

To run an experiment where episodes are executed in parallel, run:

```bash
python run_parallel.py -e <experiment_name> -n <num_parallel>
```

## Development

After installing the environment, you can run the following commands to check your code.

### Run formatter

```bash
ruff format
```

### Run style checks

```bash
ruff check
```

### Run dependency checks

```bash
deptry .
```

### Run static type checks

```bash
mypy .
```

### Run tests

```bash
pytest
```
