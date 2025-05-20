# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Config to just get Drone flying."""
from pathlib import Path
import numpy as np
from tbp.monty.frameworks.config_utils.config_args import (
    MontyArgs,
    PatchAndViewMontyConfig,
)
from tbp.monty.frameworks.experiments import (
    MontySupervisedObjectPretrainingExperiment,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    ExperimentArgs,
    PredefinedObjectInitializer,
    get_object_names_by_idx,
)
from tbp.monty.frameworks.config_utils.policy_setup_utils import (
    make_naive_scan_policy_config,
)
from tbp.monty.frameworks.models.displacement_matching import DisplacementGraphLM
from tbp.monty.frameworks.models.sensor_modules import (
    DetailedLoggingSM,
    HabitatSurfacePatchSM
)
from tbp.monty.frameworks.config_utils.config_args import (
    FiveLMMontyConfig,
    MontyArgs,
    MontyFeatureGraphArgs,
    MotorSystemConfigCurvatureInformedSurface,
    MotorSystemConfigNaiveScanSpiral,
    PatchAndViewMontyConfig,
    PretrainLoggingConfig,
    SurfaceAndViewMontyConfig,
    get_cube_face_and_corner_views_rotations,
)
from tbp.monty.frameworks.models.motor_policies import NaiveScanPolicy
from tbp.drone.src.dataset_args import DroneDatasetArgs, DroneEnvironmentDataset

# ------------------------------------------------------------------------------
# Drone
# ------------------------------------------------------------------------------



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
                policy_args=make_naive_scan_policy_config(step_size=5),
            )
        ),
    ),
    dataset_class=DroneEnvironmentDataset,
    dataset_args=DroneDatasetArgs(),
    train_dataloader_class=ED.EnvironmentDataLoaderPerObject,
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 10, object_list=["potted_meat_can"]),
        object_init_sampler=PredefinedObjectInitializer(rotations=[np.array([0,0,0])]),
    ),
)


CONFIGS = {
    "pretrain_drone": pretrain_drone_config,
}