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

## Installation

The environment for this project is managed with [conda](https://www.anaconda.com/download/success).

To create the environment, run:

### ARM64 (Apple Silicon) (zsh shell)
```
conda env create -f environment.yml --subdir=osx-64
conda init zsh
conda activate drone
conda config --env --set subdir osx-64
```

### ARM64 (Apple Silicon) (bash shell)
```
conda env create -f environment.yml --subdir=osx-64
conda init
conda activate drone
conda config --env --set subdir osx-64
```

### Intel (zsh shell)
```
conda env create -f environment.yml
conda init zsh
conda activate drone 
```

### Intel (bash shell)
```
conda env create -f environment.yml
conda init
conda activate drone 
```

## Estimating Depth using Depth-Anything-v2

[Depth-Anythin-v2 Github Repo](https://github.com/DepthAnything/Depth-Anything-V2)

1. Get the code via `git clone`
```bash
cd ~/tbp
git clone https://github.com/DepthAnything/Depth-Anything-V2
# Note: tbp.drone is at ~/tbp/tbp.drone
```
2. Download the Depth-Anything-V2-Base model from [here](https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true) 
**Note**: There are bigger models, but we have to run on CPU, so `Base` is a good starting point. 
**Note**: You will get error for the line in `dpt.py`:
```python
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
# Replace with the following line
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
```
This is because `torch.backends.mps.is_available()` is not available in our `torch` version (v1.11.0) because of our Python 3.8 dependency. So we cannot take advantage of `mps` in our Macbook Pros. 

I have already copied and pasted their requirements into our `environment.yml` so you don't need to install anything. Do update your `drone` conda environment if you have already installed it. 

## Object Segmentation using Segment Anything Model (SAM)

[Segment Anything Model Github Repo](https://github.com/facebookresearch/segment-anything)

1. Get the code via `git clone`
```bash
cd ~/tbp
git clone https://github.com/facebookresearch/segment-anything
# Note: tbp.drone is at ~/tbp/tbp.drone
```

2. Download the SAM ViT-B model from [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
**Note**: We're using the ViT-B model as it provides a good balance between performance and resource usage when running on CPU.

The model should be placed in the `~/tbp/tbp.drone/models/` directory as `sam_vit_b_01ec64.pth`.

The requirements for SAM have already been added to our `environment.yml`, so you don't need to install anything additional. Just make sure your `drone` conda environment is up to date.