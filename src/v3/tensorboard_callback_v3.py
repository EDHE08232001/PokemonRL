"""
TensorBoard callback for V3 training.

Logs per-episode environment statistics (mean/max/distributions),
exploration map images, event flag snapshots, and V3-specific metrics
(dialogue count, graph nodes/edges, maps discovered) at episode boundaries.
"""

import os
import json

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Image
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from einops import rearrange, reduce


def merge_dicts(dicts):
    """Merge a list of stat dicts into mean and distribution dicts."""
    sum_dict = {}
    count_dict = {}
    distrib_dict = {}

    for d in dicts:
        for k, v in d.items():
            if isinstance(v, (int, float)):
                sum_dict[k] = sum_dict.get(k, 0) + v
                count_dict[k] = count_dict.get(k, 0) + 1
                distrib_dict.setdefault(k, []).append(v)

    mean_dict = {}
    for k in sum_dict:
        mean_dict[k] = sum_dict[k] / count_dict[k]
        distrib_dict[k] = np.array(distrib_dict[k])

    return mean_dict, distrib_dict


class TensorboardCallback(BaseCallback):
    """Logs env stats, exploration maps, event flags, and V3 metrics to TensorBoard."""

    def __init__(self, log_dir, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.writer = None

    def _on_training_start(self):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, 'histogram'))

    def _on_step(self) -> bool:
        # Only log at episode boundaries (when env 0 is done)
        if self.training_env.env_method("check_if_done", indices=[0])[0]:
            all_infos = self.training_env.get_attr("agent_stats")
            all_final_infos = [stats[-1] for stats in all_infos]
            mean_infos, distributions = merge_dicts(all_final_infos)

            # Log mean and max stats across all parallel envs
            for key, val in mean_infos.items():
                self.logger.record(f"env_stats/{key}", val)
            for key, distrib in distributions.items():
                self.writer.add_histogram(f"env_stats_distribs/{key}", distrib, self.n_calls)
                self.logger.record(f"env_stats_max/{key}", max(distrib))

            # Log aggregated exploration map as an image
            explore_map = np.array(self.training_env.get_attr("explore_map"))
            map_sum = reduce(explore_map, "f h w -> h w", "max")
            self.logger.record("trajectory/explore_sum", Image(map_sum, "HW"), exclude=("stdout", "log", "json", "csv"))

            map_row = rearrange(explore_map, "(r f) h w -> (r h) (f w)", r=2)
            self.logger.record("trajectory/explore_map", Image(map_row, "HW"), exclude=("stdout", "log", "json", "csv"))

            # Log all event flags that have been set
            list_of_flag_dicts = self.training_env.get_attr("current_event_flags_set")
            merged_flags = {k: v for d in list_of_flag_dicts for k, v in d.items()}
            self.logger.record("trajectory/all_flags", json.dumps(merged_flags))

            # V3: Log semantic and graph metrics
            dialogue_counts = [info.get("dialogue_count", 0) for info in all_final_infos]
            graph_nodes = [info.get("graph_nodes", 0) for info in all_final_infos]
            maps_discovered = [info.get("maps_discovered", 0) for info in all_final_infos]
            self.logger.record("v3/mean_dialogue_count", np.mean(dialogue_counts))
            self.logger.record("v3/max_dialogue_count", max(dialogue_counts))
            self.logger.record("v3/mean_graph_nodes", np.mean(graph_nodes))
            self.logger.record("v3/max_graph_nodes", max(graph_nodes))
            self.logger.record("v3/mean_maps_discovered", np.mean(maps_discovered))
            self.logger.record("v3/max_maps_discovered", max(maps_discovered))

        return True

    def _on_training_end(self):
        if self.writer:
            self.writer.close()
