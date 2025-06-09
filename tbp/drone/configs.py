import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Union,
)

import numpy as np

from tbp.drone.src.dataloader import DroneImageDataLoader, DroneStreamDataLoader
from tbp.drone.src.dataset import DroneEnvironmentDataset
from tbp.drone.src.environment import (
    DroneEnvironment,
    DroneImageEnvironment,
    DroneStreamEnvironment,
)
from tbp.monty.frameworks.config_utils.config_args import (
    LoggingConfig,
    MotorSystemConfigNaiveScanSpiral,
    PatchAndViewMontyConfig,
)
from tbp.monty.frameworks.config_utils.policy_setup_utils import (
    make_naive_scan_policy_config,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.models.displacement_matching import DisplacementGraphLM
from tbp.monty.frameworks.models.evidence_matching import (
    MontyForEvidenceGraphMatching,
)
from tbp.monty.frameworks.models.motor_policies import (
    NaiveScanPolicy,
)
from tbp.monty.frameworks.models.sensor_modules import (
    DetailedLoggingSM,
    FeatureChangeSM,
)

DATA_DIR = Path("~/tbp/results/drone").expanduser()


@dataclass
class DroneEvalLoggingConfig(LoggingConfig):
    monty_log_level: str = "BASIC"
    monty_handlers: List = field(default_factory=list)
    wandb_handlers: List = field(default_factory=list)
    python_log_level: str = "INFO"
    output_dir: str = str(DATA_DIR)
    run_name: str = ""


@dataclass
class DroneEvalMontyConfig(PatchAndViewMontyConfig):
    """The best existing combination of sensor module and policy attributes.

    Uses the best existing combination of sensor module and policy attributes,
    including the feature-change sensor module, and the hypothesis-testing action
    policy.
    """

    monty_class: Callable = MontyForEvidenceGraphMatching
    sensor_module_configs: Union[dataclass, Dict] = field(
        default_factory=lambda: dict(
            sensor_module_0=dict(
                sensor_module_class=FeatureChangeSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=[
                        # morphological features (nescessarry)
                        "pose_vectors",
                        "pose_fully_defined",
                        "on_object",
                        # non-morphological features (optional)
                        "object_coverage",
                        "hsv",
                        "principal_curvatures_log",
                    ],
                    delta_thresholds={
                        "on_object": 0,
                        "n_steps": 20,
                        "hsv": [0.1, 0.1, 0.1],
                        "pose_vectors": [np.pi / 4, np.pi * 2, np.pi * 2],
                        "principal_curvatures_log": [2, 2],
                        "distance": 0.01,
                    },
                    save_raw_obs=False,
                ),
            ),
            sensor_module_1=dict(
                # No need to extract features from the view finder since it is not
                # connected to a learning module (just used at beginning of episode)
                sensor_module_class=DetailedLoggingSM,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=False,
                ),
            ),
        )
    )
    motor_system_config = MotorSystemConfigNaiveScanSpiral(
        motor_system_args=dict(
            policy_class=NaiveScanPolicy,
            policy_args=make_naive_scan_policy_config(
                step_size=5, agent_id="agent_id_0"
            ),
        )
    )


@dataclass
class DroneStreamEnvInitArgs:
    """Args for DroneStreamEnvironment"""

    patch_size: int = field(default=64)


@dataclass
class DroneImageEnvInitArgs:
    """Args for DroneImageEnvironment"""

    patch_size: int = field(default=64)
    data_path: Optional[os.PathLike] = field(default=None)


@dataclass
class DroneStreamDatasetArgs:
    env_init_func: Callable = field(default=DroneStreamEnvironment)
    env_init_args: Dict = field(
        default_factory=lambda: DroneStreamEnvInitArgs().__dict__
    )
    transform = None
    rng = None

    def __post_init__(self):
        self.transform = []
        #     DepthTo3DLocations(
        #         agent
        #         resolution=(720, 960),  # DJI Tello camera resolution
        #         focal_length_pixels=1825.1,  # Calculated from physical parameters
        #         optical_center=(480.0, 360.0),  # Half of resolution
        #         get_all_points=False,  # Only get object points
        #     )
        # ]


@dataclass
class DroneImageDatasetArgs:
    env_init_func: Callable = field(default=DroneImageEnvironment)
    env_init_args: Dict = field(
        default_factory=lambda: DroneImageEnvInitArgs().__dict__
    )
    transform = None
    rng = None

    def __post_init__(self):
        self.transform = []
        #     DepthTo3DLocations(
        #         agent
        #         resolution=(720, 960),  # DJI Tello camera resolution
        #         focal_length_pixels=1825.1,  # Calculated from physical parameters
        #         optical_center=(480.0, 360.0),  # Half of resolution
        #         get_all_points=False,  # Only get object points
        #     )
        # ]
