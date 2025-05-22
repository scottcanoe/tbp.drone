# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Config to just get Drone flying."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Callable,
    Dict,
    List,
    Union,
)

import numpy as np

from tbp.drone.src.actions import (
    Action,
    Land,
    MoveBackward,
    MoveDown,
    MoveForward,
    MoveLeft,
    MoveRight,
    MoveUp,
    SetHeight,
    SetYaw,
    TakeOff,
    TurnLeft,
    TurnRight,
)
from tbp.drone.src.dataloader import DroneDataLoader
from tbp.drone.src.dataset import DroneDatasetArgs, DroneEnvironmentDataset
from tbp.monty.frameworks.config_utils.config_args import (
    LoggingConfig,
    MontyArgs,
    MotorSystemConfigNaiveScanSpiral,
    PatchAndViewMontyConfig,
    PretrainLoggingConfig,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    EvalExperimentArgs,
    ExperimentArgs,
    PredefinedObjectInitializer,
    get_object_names_by_idx,
)
from tbp.monty.frameworks.config_utils.policy_setup_utils import (
    make_naive_scan_policy_config,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.experiments import (
    MontyObjectRecognitionExperiment,
    MontySupervisedObjectPretrainingExperiment,
)
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

# ------------------------------------------------------------------------------
# Drone
# ------------------------------------------------------------------------------

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
class DroneEvalDataLoaderArgs:
    object_name: str
    actions: List[Action]


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


# The below is modeled after `supervised_pre_training_base from benchmarks/configs/pretraining_experiments.py`
pretrain_drone_config = dict(
    experiment_class=MontySupervisedObjectPretrainingExperiment,
    experiment_args=ExperimentArgs(
        do_eval=False,
        n_train_epochs=1,
    ),
    logging_config=PretrainLoggingConfig(
        output_dir=Path("~/tbp/tbp.drone/pretraining").expanduser(),
    ),
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyArgs(num_exploratory_steps=500),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=DisplacementGraphLM,
                learning_module_args=dict(
                    k=10,
                    match_attribute="displacement",
                    tolerance=np.ones(3) * 0.0001,
                    graph_delta_thresholds=dict(
                        patch=dict(
                            distance=0.001,
                            # Only first pose vector (point normal) is currently used
                            pose_vectors=[np.pi / 8, np.pi * 2, np.pi * 2],
                            principal_curvatures_log=[1, 1],
                            hsv=[0.1, 1, 1],
                        )
                    ),
                ),
            ),
        ),
        motor_system_config=MotorSystemConfigNaiveScanSpiral(
            motor_system_args=dict(
                policy_class=NaiveScanPolicy,
                policy_args=make_naive_scan_policy_config(step_size=5, agent_id="agent_id_0"),
            )
        ),
    ),
    dataset_class=DroneEnvironmentDataset,
    dataset_args=DroneDatasetArgs(),
    train_dataloader_class=ED.EnvironmentDataLoaderPerObject,
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 1, object_list=["potted_meat_can"]),
        object_init_sampler=PredefinedObjectInitializer(rotations=[np.array([0,0,0])]),
    ),
)

# The below is modeled after `supervised_pre_training_base from benchmarks/configs/pretraining_experiments.py`
drone_test = dict(
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=EvalExperimentArgs(),
    logging_config=DroneEvalLoggingConfig(run_name="drone_test"),
    monty_config=DroneEvalMontyConfig(),
    dataset_class=DroneEnvironmentDataset,
    dataset_args=DroneDatasetArgs(),
    eval_dataloader_class=DroneDataLoader,
    eval_dataloader_args=DroneEvalDataLoaderArgs(
        object_name="potted_meat_can",
        actions=[TakeOff(), MoveLeft(0.2), MoveRight(0.2)],
    ),
)


CONFIGS = {
    "pretrain_drone": pretrain_drone_config,
    "drone_test": drone_test,
}
