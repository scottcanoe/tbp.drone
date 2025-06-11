# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Module for the tbp.drone's experiment configurations.

Note: The config(s) defined here will not yet work until further development of
the environment/dataset/dataloader pipeline.
"""

from dataclasses import asdict

from configs.drone import pretrain_drone_config
from configs.names import Experiments

# Add your experiment configurations here
# e.g.: my_experiment_config = dict(...)

experiments = Experiments(
    # For each experiment name in Experiments, add its corresponding
    # configuration here.
    # e.g.: my_experiment=my_experiment_config
    pretrain_drone=pretrain_drone_config,
)
CONFIGS = asdict(experiments)
