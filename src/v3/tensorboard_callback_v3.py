"""
TensorBoard callback for V3 training.

Logs per-episode environment statistics (mean/max/distributions),
exploration map images, event flag snapshots, and V3-specific metrics
(dialogue count, graph nodes/edges, maps discovered) at episode boundaries.

Also prints a clean per-rollout summary table to stdout for SLURM .out files.
"""

import os
import json
import time

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


def _format_env_summary(all_stats, timestep, elapsed_seconds):
    """Build a box-drawn summary table of all environment stats.

    Parameters
    ----------
    all_stats : list[dict]
        One stats dict per environment (from get_latest_stats).
    timestep : int
        Current global training timestep.
    elapsed_seconds : float
        Wall-clock seconds since training started.
    """
    # Filter out empty dicts (envs that haven't started yet)
    valid = [(i, s) for i, s in enumerate(all_stats) if s]
    if not valid:
        return ""

    # Columns: env_id, step, map, event, level, badge, explore, deaths, dialogue, graph_dist, faints, t_wins, oak, party, reward_sum
    headers = ["env", "step", "map", "event", "level", "badge", "explore", "deaths", "dialogue", "g_dist", "faints", "t_win", "oak", "party", "rew_sum"]
    col_widths = [5, 7, 5, 7, 7, 5, 7, 6, 8, 6, 6, 5, 3, 5, 10]

    def _fmt_row(vals):
        cells = []
        for v, w in zip(vals, col_widths):
            s = str(v)
            cells.append(s.rjust(w))
        return " │ ".join(cells)

    # Sort by reward_sum descending (event is used as proxy for reward progress)
    rows = []
    reward_sums = []
    for env_id, s in valid:
        reward_sum = sum(v for k, v in s.items() if k in (
            "event", "healr", "coord_count", "deaths", "dialogue_count", "graph_distance"
        ) and isinstance(v, (int, float)))
        # Use the actual fields we have
        row_vals = [
            env_id,
            s.get("step", 0),
            s.get("map", 0),
            f'{s.get("event", 0):.2f}',
            s.get("levels_sum", 0),
            s.get("badge", 0),
            s.get("coord_count", 0),
            s.get("deaths", 0),
            s.get("dialogue_count", 0),
            s.get("graph_distance", 0),
            s.get("enemy_faint_count", 0),
            s.get("trainer_win_count", 0),
            s.get("oak_milestones", 0),
            s.get("max_party_size", 1),
            f'{s.get("event", 0) + s.get("healr", 0):.2f}',
        ]
        rows.append((s.get("event", 0) + s.get("healr", 0), row_vals))
        reward_sums.append(s.get("event", 0) + s.get("healr", 0))

    rows.sort(key=lambda r: r[0], reverse=True)

    # Build table
    elapsed_h = int(elapsed_seconds // 3600)
    elapsed_m = int((elapsed_seconds % 3600) // 60)
    elapsed_s = int(elapsed_seconds % 60)

    header_line = _fmt_row(headers)
    table_width = len(header_line)
    sep = "─" * table_width

    lines = []
    lines.append(f"┌{sep}┐")
    lines.append(f"│ Timestep: {timestep:,}  Wall-clock: {elapsed_h:02d}:{elapsed_m:02d}:{elapsed_s:02d}".ljust(table_width) + "│")
    lines.append(f"├{sep}┤")
    lines.append(f"│{header_line}│")
    lines.append(f"├{sep}┤")
    for _, row_vals in rows:
        lines.append(f"│{_fmt_row(row_vals)}│")
    lines.append(f"├{sep}┤")

    # Aggregate stats
    if reward_sums:
        mean_reward = np.mean(reward_sums)
        max_reward = np.max(reward_sums)
        mean_explore = np.mean([s.get("coord_count", 0) for _, s in valid])
        max_explore = np.max([s.get("coord_count", 0) for _, s in valid])
        mean_badge = np.mean([s.get("badge", 0) for _, s in valid])
        max_badge = int(np.max([s.get("badge", 0) for _, s in valid]))
        agg_line = (
            f" mean reward: {mean_reward:.2f}  max reward: {max_reward:.2f}"
            f"  mean explore: {mean_explore:.0f}  max explore: {max_explore:.0f}"
            f"  mean badge: {mean_badge:.1f}  max badge: {max_badge}"
        )
        lines.append(f"│{agg_line.ljust(table_width)}│")

    lines.append(f"└{sep}┘")
    return "\n".join(lines)


class TensorboardCallback(BaseCallback):
    """Logs env stats, exploration maps, event flags, and V3 metrics to TensorBoard."""

    def __init__(self, log_dir, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.writer = None
        self._start_time = None
        self._rollout_count = 0
        self._rollouts_per_episode = None

    def _on_training_start(self):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, 'histogram'))
        self._start_time = time.time()
        # Compute how many rollouts fit in one episode: ep_length / n_steps
        ep_length = self.training_env.get_attr("max_steps", indices=[0])[0]
        n_steps = self.model.n_steps
        self._rollouts_per_episode = max(ep_length // n_steps, 1)

    def _on_step(self) -> bool:
        return True

    def _log_episode_metrics(self, all_stats):
        """Log episode-level metrics to TensorBoard."""
        mean_infos, distributions = merge_dicts(all_stats)

        # Log mean and max stats across all parallel envs
        for key, val in mean_infos.items():
            self.logger.record(f"env_stats/{key}", val)
        for key, distrib in distributions.items():
            self.writer.add_histogram(f"env_stats_distribs/{key}", distrib, self.num_timesteps)
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
        dialogue_counts = [info.get("dialogue_count", 0) for info in all_stats]
        graph_nodes = [info.get("graph_nodes", 0) for info in all_stats]
        maps_discovered = [info.get("maps_discovered", 0) for info in all_stats]
        self.logger.record("v3/mean_dialogue_count", np.mean(dialogue_counts))
        self.logger.record("v3/max_dialogue_count", max(dialogue_counts))
        self.logger.record("v3/mean_graph_nodes", np.mean(graph_nodes))
        self.logger.record("v3/max_graph_nodes", max(graph_nodes))
        self.logger.record("v3/mean_maps_discovered", np.mean(maps_discovered))
        self.logger.record("v3/max_maps_discovered", max(maps_discovered))

        op_levels = [info.get("max_opponent_level", 0) for info in all_stats]
        self.logger.record("v3/mean_op_level", np.mean(op_levels))
        self.logger.record("v3/max_op_level", max(op_levels))

        # V3: Battle progression metrics
        enemy_faints = [info.get("enemy_faint_count", 0) for info in all_stats]
        trainer_wins = [info.get("trainer_win_count", 0) for info in all_stats]
        battle_entries = [info.get("battle_entries", 0) for info in all_stats]
        self.logger.record("v3/mean_enemy_faints", np.mean(enemy_faints))
        self.logger.record("v3/max_enemy_faints", max(enemy_faints))
        self.logger.record("v3/mean_trainer_wins", np.mean(trainer_wins))
        self.logger.record("v3/max_trainer_wins", max(trainer_wins))
        self.logger.record("v3/mean_battle_entries", np.mean(battle_entries))
        self.logger.record("v3/max_battle_entries", max(battle_entries))

        # V3: Quest progression metrics
        oak_milestones = [info.get("oak_milestones", 0) for info in all_stats]
        self.logger.record("v3/mean_oak_milestones", np.mean(oak_milestones))
        self.logger.record("v3/max_oak_milestones", max(oak_milestones))

        # V3: Party growth metrics
        max_party = [info.get("max_party_size", 1) for info in all_stats]
        self.logger.record("v3/mean_max_party_size", np.mean(max_party))
        self.logger.record("v3/max_max_party_size", max(max_party))

        # V3: Per-component reward instrumentation
        reward_comp_lists = {}
        for info in all_stats:
            comps = info.get("reward_components", {})
            for k, v in comps.items():
                if isinstance(v, (int, float)):
                    reward_comp_lists.setdefault(k, []).append(v)
        for comp_name, values in reward_comp_lists.items():
            arr = np.array(values)
            self.logger.record(f"reward_components/{comp_name}_mean", np.mean(arr))
            self.logger.record(f"reward_components/{comp_name}_max", np.max(arr))

    def _on_rollout_end(self) -> None:
        """Print env summary table and log episode-level TensorBoard metrics."""
        all_stats = self.training_env.env_method("get_latest_stats")
        elapsed = time.time() - self._start_time if self._start_time else 0.0

        # Print SLURM summary table every rollout
        summary = _format_env_summary(all_stats, self.num_timesteps, elapsed)
        if summary:
            print(summary, flush=True)

        # Log episode-level TensorBoard metrics at episode boundaries
        self._rollout_count += 1
        if self._rollouts_per_episode and self._rollout_count % self._rollouts_per_episode == 0:
            self._log_episode_metrics(all_stats)

    def _on_training_end(self):
        if self.writer:
            self.writer.close()
