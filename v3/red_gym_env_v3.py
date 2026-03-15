"""
V3 Gym environment for Pokemon Red.

Builds on V2's coordinate-based exploration with three new systems:
1. Recurrent Memory (LSTM) — removes recent_actions from obs (handled natively by LSTM)
2. Semantic Text Rewards — reads and decodes Gen 1 text from WRAM, rewards new dialogue
3. Topological Graph Navigation — builds a directed graph of (map_id, r, c) nodes,
   detects warp edges, and feeds graph distance into the observation
"""

import uuid
import json
import hashlib
from pathlib import Path

import numpy as np
import networkx as nx
from skimage.transform import downscale_local_mean
import matplotlib.pyplot as plt
from pyboy import PyBoy
import mediapy as media
from einops import repeat

from gymnasium import Env, spaces
from pyboy.utils import WindowEvent

from v3.global_map import local_to_global, GLOBAL_MAP_SHAPE

# Event flag memory range
event_flags_start = 0xD747
event_flags_end = 0xD87E  # expanded to cover SS Anne events
museum_ticket = (0xD754, 0)

# WRAM text buffer address and length (Gen 1)
TEXT_BUFFER_ADDR = 0xCF4B
TEXT_BUFFER_LEN = 20

# Gen 1 character map (hex → character)
GEN1_CHAR_MAP = {0x7F: " ", 0x50: "\x00"}  # 0x50 = string terminator
# A–Z: 0x80–0x99
for _i, _c in enumerate(range(0x80, 0x9A)):
    GEN1_CHAR_MAP[_c] = chr(ord("A") + _i)
# a–z: 0x9A–0xB3
for _i, _c in enumerate(range(0x9A, 0xB4)):
    GEN1_CHAR_MAP[_c] = chr(ord("a") + _i)
# 0–9: 0xF6–0xFF
for _i, _c in enumerate(range(0xF6, 0x100)):
    GEN1_CHAR_MAP[_c] = str(_i)


def decode_text(hex_array):
    """Decode a raw hex byte array into a string using the Gen 1 character map.

    Stops at the string terminator (0x50) or end of array.
    Unknown bytes are skipped.
    """
    chars = []
    for b in hex_array:
        if b == 0x50:
            break
        ch = GEN1_CHAR_MAP.get(b)
        if ch is not None:
            chars.append(ch)
    return "".join(chars)


def text_to_obs_hash(text, size=8):
    """Hash a text string into a normalized uint8 vector of given size."""
    h = hashlib.md5(text.encode()).digest()
    return np.array([b for b in h[:size]], dtype=np.uint8)


class RedGymEnv(Env):
    """
    V3 Pokemon Red environment with LSTM-ready observations,
    semantic text rewards, and topological graph navigation.
    """

    def __init__(self, config=None):
        self.s_path = config["session_path"]
        self.save_final_state = config["save_final_state"]
        self.print_rewards = config["print_rewards"]
        self.headless = config["headless"]
        self.init_state = config["init_state"]
        self.act_freq = config["action_freq"]
        self.max_steps = config["max_steps"]
        self.save_video = config["save_video"]
        self.fast_video = config["fast_video"]
        self.frame_stacks = 3  # grayscale frames stacked along channel dim
        self.explore_weight = (
            1 if "explore_weight" not in config else config["explore_weight"]
        )
        self.reward_scale = (
            1 if "reward_scale" not in config else config["reward_scale"]
        )
        self.instance_id = (
            str(uuid.uuid4())[:8]
            if "instance_id" not in config
            else config["instance_id"]
        )
        self.s_path.mkdir(exist_ok=True)
        self.full_frame_writer = None
        self.model_frame_writer = None
        self.map_frame_writer = None
        self.reset_count = 0
        self.all_runs = []

        # Map IDs for key locations, ordered by game progress
        self.essential_map_locations = {
            v: i for i, v in enumerate([
                40, 0, 12, 1, 13, 51, 2, 54, 14, 59, 60, 61, 15, 3, 65
            ])
        }

        self.metadata = {"render.modes": []}
        self.reward_range = (0, 15000)

        # 7 actions: 4 directions + A + B + Start
        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START,
        ]

        self.release_actions = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_START
        ]

        # Load event flag names for logging
        with open(Path(__file__).parent / "events.json") as f:
            event_names = json.load(f)
        self.event_names = event_names

        self.output_shape = (72, 80, self.frame_stacks)  # grayscale, 3 stacked
        self.coords_pad = 12  # half-size of local exploration map window

        self.action_space = spaces.Discrete(len(self.valid_actions))

        self.enc_freqs = 8  # Fourier encoding frequencies for level
        self.text_hash_size = 8  # size of hashed text observation vector

        # Dict observation: LSTM-ready (no recent_actions — handled by LSTM natively)
        # New: text_hash for semantic content, graph_distance for topological progress
        self.observation_space = spaces.Dict(
            {
                "screens": spaces.Box(low=0, high=255, shape=self.output_shape, dtype=np.uint8),
                "health": spaces.Box(low=0, high=1),
                "level": spaces.Box(low=-1, high=1, shape=(self.enc_freqs,)),
                "badges": spaces.MultiBinary(8),
                "events": spaces.MultiBinary((event_flags_end - event_flags_start) * 8),
                "map": spaces.Box(low=0, high=255, shape=(
                    self.coords_pad * 4, self.coords_pad * 4, 1), dtype=np.uint8),
                "text_hash": spaces.Box(low=0, high=255, shape=(self.text_hash_size,), dtype=np.uint8),
                "graph_distance": spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8),
            }
        )

        head = "null" if config["headless"] else "SDL2"

        self.pyboy = PyBoy(
            config["gb_path"],
            window=head,
        )

        if not config["headless"]:
            self.pyboy.set_emulation_speed(6)

    def reset(self, seed=None, options={}):
        """Reset emulator state and all episode tracking variables."""
        self.seed = seed
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)

        self.init_map_mem()

        self.agent_stats = []

        # Global exploration map: marks visited tiles on the stitched world map
        self.explore_map_dim = GLOBAL_MAP_SHAPE
        self.explore_map = np.zeros(self.explore_map_dim, dtype=np.uint8)

        self.recent_screens = np.zeros(self.output_shape, dtype=np.uint8)

        self.levels_satisfied = False
        self.base_explore = 0
        self.max_opponent_level = 0
        self.max_event_rew = 0
        self.max_level_rew = 0
        self.last_health = 1
        self.total_healing_rew = 0
        self.died_count = 0
        self.party_size = 0
        self.step_count = 0

        # Baseline event flag count (to only reward newly triggered flags)
        self.base_event_flags = sum([
            self.bit_count(self.read_m(i))
            for i in range(event_flags_start, event_flags_end)
        ])

        self.current_event_flags_set = {}

        self.max_map_progress = 0

        # --- V3: Semantic text tracking ---
        self.seen_dialogue = set()
        self.current_text_hash = np.zeros(self.text_hash_size, dtype=np.uint8)

        # --- V3: Topological graph ---
        self.world_graph = nx.DiGraph()
        x, y, m = self.get_game_coords()
        self.previous_node = (m, y, x)
        self.world_graph.add_node(self.previous_node)
        self.discovered_maps = {m}
        self.root_node = self.previous_node  # Pallet Town starting position
        self.graph_distance = 0

        # --- V3: Semantic reward accumulator (for step delta tracking) ---
        self.semantic_reward = 0.0

        self.progress_reward = self.get_game_state_reward()
        self.total_reward = sum([val for _, val in self.progress_reward.items()])
        self.reset_count += 1
        return self._get_obs(), {}

    def init_map_mem(self):
        """Initialize coordinate visit counter."""
        self.seen_coords = {}

    def render(self, reduce_res=True):
        """Render current game screen as grayscale (single channel)."""
        game_pixels_render = self.pyboy.screen.ndarray[:, :, 0:1]  # (144, 160, 1)
        if reduce_res:
            game_pixels_render = (
                downscale_local_mean(game_pixels_render, (2, 2, 1))
            ).astype(np.uint8)
        return game_pixels_render

    def _get_obs(self):
        """Build the dict observation from game state."""
        screen = self.render()

        self.update_recent_screens(screen)

        # Fourier-encode total party level (normalized by 0.02)
        level_sum = 0.02 * sum([
            self.read_m(a) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
        ])

        observation = {
            "screens": self.recent_screens,
            "health": np.array([self.read_hp_fraction()]),
            "level": self.fourier_encode(level_sum),
            "badges": np.array([int(bit) for bit in f"{self.read_m(0xD356):08b}"], dtype=np.int8),
            "events": np.array(self.read_event_bits(), dtype=np.int8),
            "map": self.get_explore_map()[:, :, None],  # local map crop
            "text_hash": self.current_text_hash,
            "graph_distance": np.array([min(self.graph_distance, 255)], dtype=np.uint8),
        }

        return observation

    def step(self, action):
        """Execute one agent step: act, observe, compute reward."""
        if self.save_video and self.step_count == 0:
            self.start_video()

        self.run_action_on_emulator(action)
        self.append_agent_stats(action)

        self.update_seen_coords()

        self.update_explore_map()

        self.update_heal_reward()

        self.party_size = self.read_m(0xD163)

        # --- V3: Semantic text processing ---
        self.process_text_buffer()

        # --- V3: Topological graph update ---
        self.update_world_graph()

        new_reward = self.update_reward()

        self.last_health = self.read_hp_fraction()

        self.update_map_progress()

        step_limit_reached = self.check_if_done()

        obs = self._get_obs()

        # Periodically snapshot event flags for logging
        if self.step_count % 100 == 0:
            for address in range(event_flags_start, event_flags_end):
                val = self.read_m(address)
                for idx, bit in enumerate(f"{val:08b}"):
                    if bit == "1":
                        key = f"0x{address:X}-{idx}"
                        if key in self.event_names.keys():
                            self.current_event_flags_set[key] = self.event_names[key]
                        else:
                            print(f"could not find key: {key}")

        self.step_count += 1

        return obs, new_reward, False, step_limit_reached, {}

    # ------------------------------------------------------------------ #
    #  V3: Semantic Text Hooking                                          #
    # ------------------------------------------------------------------ #

    def read_text_buffer(self):
        """Read the active text buffer from WRAM (0xCF4B, up to 20 bytes)."""
        return [self.read_m(TEXT_BUFFER_ADDR + i) for i in range(TEXT_BUFFER_LEN)]

    def process_text_buffer(self):
        """Decode the text buffer and grant a one-time reward for new dialogue."""
        raw = self.read_text_buffer()
        text = decode_text(raw)

        if len(text) > 3 and text not in self.seen_dialogue:
            self.seen_dialogue.add(text)
            self.semantic_reward += 0.5  # one-time intrinsic reward
            self.current_text_hash = text_to_obs_hash(text, self.text_hash_size)

    # ------------------------------------------------------------------ #
    #  V3: Topological Graph Navigation                                   #
    # ------------------------------------------------------------------ #

    def update_world_graph(self):
        """Track movement as edges in a directed graph. Detect warp edges."""
        x, y, m = self.get_game_coords()
        current_node = (m, y, x)

        if current_node != self.previous_node:
            # Add edge (creates nodes automatically)
            self.world_graph.add_edge(self.previous_node, current_node)

            # Warp detection: map ID changed
            if current_node[0] != self.previous_node[0]:
                self.world_graph.edges[self.previous_node, current_node]["warp"] = True
                # High reward for discovering a new map via a warp
                if current_node[0] not in self.discovered_maps:
                    self.discovered_maps.add(current_node[0])
                    self.semantic_reward += 2.0  # warp discovery bonus

            self.previous_node = current_node

        # Update graph distance from root
        try:
            self.graph_distance = nx.shortest_path_length(
                self.world_graph, self.root_node, current_node
            )
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            self.graph_distance = 0

    # ------------------------------------------------------------------ #
    #  Emulator & Input                                                   #
    # ------------------------------------------------------------------ #

    def run_action_on_emulator(self, action):
        """Press button for 8 ticks, release, then tick remaining frames."""
        self.pyboy.send_input(self.valid_actions[action])
        render_screen = self.save_video or not self.headless
        press_step = 8
        self.pyboy.tick(press_step, render_screen)
        self.pyboy.send_input(self.release_actions[action])
        self.pyboy.tick(self.act_freq - press_step - 1, render_screen)
        self.pyboy.tick(1, True)  # final tick with rendering for observation
        if self.save_video and self.fast_video:
            self.add_video_frame()

    def append_agent_stats(self, action):
        """Log per-step stats for TensorBoard and CSV export."""
        x_pos, y_pos, map_n = self.get_game_coords()
        levels = [
            self.read_m(a) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
        ]
        self.agent_stats.append(
            {
                "step": self.step_count,
                "x": x_pos,
                "y": y_pos,
                "map": map_n,
                "max_map_progress": self.max_map_progress,
                "last_action": action,
                "pcount": self.read_m(0xD163),
                "levels": levels,
                "levels_sum": sum(levels),
                "ptypes": self.read_party(),
                "hp": self.read_hp_fraction(),
                "coord_count": len(self.seen_coords),
                "deaths": self.died_count,
                "badge": self.get_badges(),
                "event": self.progress_reward["event"],
                "healr": self.total_healing_rew,
                "dialogue_count": len(self.seen_dialogue),
                "graph_nodes": self.world_graph.number_of_nodes(),
                "graph_edges": self.world_graph.number_of_edges(),
                "maps_discovered": len(self.discovered_maps),
                "graph_distance": self.graph_distance,
            }
        )

    # ------------------------------------------------------------------ #
    #  Video Recording                                                    #
    # ------------------------------------------------------------------ #

    def start_video(self):
        """Initialize video writers for full, model, and map views."""
        if self.full_frame_writer is not None:
            self.full_frame_writer.close()
        if self.model_frame_writer is not None:
            self.model_frame_writer.close()
        if self.map_frame_writer is not None:
            self.map_frame_writer.close()

        base_dir = self.s_path / Path("rollouts")
        base_dir.mkdir(exist_ok=True)
        full_name = Path(
            f"full_reset_{self.reset_count}_id{self.instance_id}"
        ).with_suffix(".mp4")
        model_name = Path(
            f"model_reset_{self.reset_count}_id{self.instance_id}"
        ).with_suffix(".mp4")
        self.full_frame_writer = media.VideoWriter(
            base_dir / full_name, (144, 160), fps=60, input_format="gray"
        )
        self.full_frame_writer.__enter__()
        self.model_frame_writer = media.VideoWriter(
            base_dir / model_name, self.output_shape[:2], fps=60, input_format="gray"
        )
        self.model_frame_writer.__enter__()
        map_name = Path(
            f"map_reset_{self.reset_count}_id{self.instance_id}"
        ).with_suffix(".mp4")
        self.map_frame_writer = media.VideoWriter(
            base_dir / map_name,
            (self.coords_pad * 4, self.coords_pad * 4),
            fps=60, input_format="gray"
        )
        self.map_frame_writer.__enter__()

    def add_video_frame(self):
        """Record current frame to all video writers."""
        self.full_frame_writer.add_image(
            self.render(reduce_res=False)[:, :, 0]
        )
        self.model_frame_writer.add_image(
            self.render(reduce_res=True)[:, :, 0]
        )
        self.map_frame_writer.add_image(
            self.get_explore_map()
        )

    # ------------------------------------------------------------------ #
    #  Game State Reading                                                 #
    # ------------------------------------------------------------------ #

    def get_game_coords(self):
        """Read player (x, y, map_id) from memory."""
        return (self.read_m(0xD362), self.read_m(0xD361), self.read_m(0xD35E))

    def update_seen_coords(self):
        """Increment visit count for current tile (skipped during battles)."""
        if self.read_m(0xD057) == 0:  # not in battle
            x_pos, y_pos, map_n = self.get_game_coords()
            coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"
            if coord_string in self.seen_coords.keys():
                self.seen_coords[coord_string] += 1
            else:
                self.seen_coords[coord_string] = 1

    def get_current_coord_count_reward(self):
        """Penalty signal: returns 1 if current tile visited >= 600 times."""
        x_pos, y_pos, map_n = self.get_game_coords()
        coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"
        if coord_string in self.seen_coords.keys():
            count = self.seen_coords[coord_string]
        else:
            count = 0
        return 0 if count < 600 else 1

    def get_global_coords(self):
        """Convert current game position to global map coordinates."""
        x_pos, y_pos, map_n = self.get_game_coords()
        return local_to_global(y_pos, x_pos, map_n)

    def update_explore_map(self):
        """Mark current global position as visited on the exploration map."""
        c = self.get_global_coords()
        if c[0] >= self.explore_map.shape[0] or c[1] >= self.explore_map.shape[1]:
            print(f"coord out of bounds! global: {c} game: {self.get_game_coords()}")
        else:
            self.explore_map[c[0], c[1]] = 255

    def get_explore_map(self):
        """Extract and upscale a local crop of the exploration map around the player."""
        c = self.get_global_coords()
        if c[0] >= self.explore_map.shape[0] or c[1] >= self.explore_map.shape[1]:
            out = np.zeros((self.coords_pad * 2, self.coords_pad * 2), dtype=np.uint8)
        else:
            out = self.explore_map[
                c[0] - self.coords_pad:c[0] + self.coords_pad,
                c[1] - self.coords_pad:c[1] + self.coords_pad
            ]
        # 2x upscale for better visibility in the observation
        return repeat(out, 'h w -> (h h2) (w w2)', h2=2, w2=2)

    def update_recent_screens(self, cur_screen):
        """Roll screen stack and insert new frame at position 0."""
        self.recent_screens = np.roll(self.recent_screens, 1, axis=2)
        self.recent_screens[:, :, 0] = cur_screen[:, :, 0]

    # ------------------------------------------------------------------ #
    #  Reward                                                             #
    # ------------------------------------------------------------------ #

    def update_reward(self):
        """Compute step reward as delta from previous total."""
        self.progress_reward = self.get_game_state_reward()
        new_total = sum(
            [val for _, val in self.progress_reward.items()]
        )
        new_step = new_total - self.total_reward

        self.total_reward = new_total
        return new_step

    def group_rewards(self):
        """Group rewards into 3 channels (level, HP, explore)."""
        prog = self.progress_reward
        return (
            prog["level"] * 100 / self.reward_scale,
            self.read_hp_fraction() * 2000,
            prog["explore"] * 150 / (self.explore_weight * self.reward_scale),
        )

    def check_if_done(self):
        """Episode ends when step count reaches max_steps."""
        done = self.step_count >= self.max_steps - 1
        return done

    def save_and_print_info(self, done, obs):
        """Print progress and save screenshots/videos on episode end."""
        if self.print_rewards:
            prog_string = f"step: {self.step_count:6d}"
            for key, val in self.progress_reward.items():
                prog_string += f" {key}: {val:5.2f}"
            prog_string += f" sum: {self.total_reward:5.2f}"
            print(f"\r{prog_string}", end="", flush=True)

        if self.step_count % 50 == 0:
            plt.imsave(
                self.s_path / Path(f"curframe_{self.instance_id}.jpeg"),
                self.render(reduce_res=False)[:, :, 0],
            )

        if self.print_rewards and done:
            print("", flush=True)
            if self.save_final_state:
                fs_path = self.s_path / Path("final_states")
                fs_path.mkdir(exist_ok=True)
                plt.imsave(
                    fs_path
                    / Path(
                        f"frame_r{self.total_reward:.4f}_{self.reset_count}_explore_map.jpeg"
                    ),
                    obs["map"][:, :, 0],
                )
                plt.imsave(
                    fs_path
                    / Path(
                        f"frame_r{self.total_reward:.4f}_{self.reset_count}_full_explore_map.jpeg"
                    ),
                    self.explore_map,
                )
                plt.imsave(
                    fs_path
                    / Path(
                        f"frame_r{self.total_reward:.4f}_{self.reset_count}_full.jpeg"
                    ),
                    self.render(reduce_res=False)[:, :, 0],
                )

        if self.save_video and done:
            self.full_frame_writer.close()
            self.model_frame_writer.close()
            self.map_frame_writer.close()

    # ------------------------------------------------------------------ #
    #  Memory Access                                                      #
    # ------------------------------------------------------------------ #

    def read_m(self, addr):
        """Read a single byte from Game Boy memory (PyBoy v2 API)."""
        return self.pyboy.memory[addr]

    def read_bit(self, addr, bit: int) -> bool:
        """Read a specific bit from a memory address."""
        return bin(256 + self.read_m(addr))[-bit - 1] == "1"

    def read_event_bits(self):
        """Read all event flag bits as a flat list of 0s and 1s."""
        return [
            int(bit) for i in range(event_flags_start, event_flags_end)
            for bit in f"{self.read_m(i):08b}"
        ]

    def get_levels_sum(self):
        """Sum party levels, offset by starter pokemon level."""
        min_poke_level = 2
        starter_additional_levels = 4
        poke_levels = [
            max(self.read_m(a) - min_poke_level, 0)
            for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
        ]
        return max(sum(poke_levels) - starter_additional_levels, 0)

    def get_levels_reward(self):
        """Reward for party levels; scales down above threshold 22."""
        explore_thresh = 22
        scale_factor = 4
        level_sum = self.get_levels_sum()
        if level_sum < explore_thresh:
            scaled = level_sum
        else:
            scaled = (level_sum - explore_thresh) / scale_factor + explore_thresh
        self.max_level_rew = max(self.max_level_rew, scaled)
        return self.max_level_rew

    def get_badges(self):
        """Count gym badges earned."""
        return self.bit_count(self.read_m(0xD356))

    def read_party(self):
        """Read species IDs of all party pokemon."""
        return [
            self.read_m(addr)
            for addr in [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]
        ]

    def get_all_events_reward(self):
        """Count newly set event flags (minus baseline and museum ticket)."""
        return max(
            sum([
                self.bit_count(self.read_m(i))
                for i in range(event_flags_start, event_flags_end)
            ])
            - self.base_event_flags
            - int(self.read_bit(museum_ticket[0], museum_ticket[1])),
            0,
        )

    def get_game_state_reward(self, print_stats=False):
        """
        Compute composite reward dict.
        V3 adds: semantic (text) and warp discovery rewards.
        """
        state_scores = {
            "event": self.reward_scale * self.update_max_event_rew() * 4,
            "heal": self.reward_scale * self.total_healing_rew * 10,
            "badge": self.reward_scale * self.get_badges() * 10,
            "explore": self.reward_scale * self.explore_weight * len(self.seen_coords) * 0.1,
            "stuck": self.reward_scale * self.get_current_coord_count_reward() * -0.05,
            "semantic": self.reward_scale * self.semantic_reward,
        }

        return state_scores

    def update_max_op_level(self):
        """Track highest opponent pokemon level seen."""
        opp_base_level = 5
        opponent_level = (
            max([
                self.read_m(a)
                for a in [0xD8C5, 0xD8F1, 0xD91D, 0xD949, 0xD975, 0xD9A1]
            ])
            - opp_base_level
        )
        self.max_opponent_level = max(self.max_opponent_level, opponent_level)
        return self.max_opponent_level

    def update_max_event_rew(self):
        """Track max event reward (monotonically increasing)."""
        cur_rew = self.get_all_events_reward()
        self.max_event_rew = max(cur_rew, self.max_event_rew)
        return self.max_event_rew

    def update_heal_reward(self):
        """Track healing events and deaths from HP changes."""
        cur_health = self.read_hp_fraction()
        if cur_health > self.last_health and self.read_m(0xD163) == self.party_size:
            if self.last_health > 0:
                heal_amount = cur_health - self.last_health
                # Quadratic healing reward (encourages larger heals)
                self.total_healing_rew += heal_amount * heal_amount
            else:
                self.died_count += 1

    def read_hp_fraction(self):
        """Read total party HP as fraction of max HP."""
        hp_sum = sum([
            self.read_hp(add)
            for add in [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]
        ])
        max_hp_sum = sum([
            self.read_hp(add)
            for add in [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]
        ])
        max_hp_sum = max(max_hp_sum, 1)
        return hp_sum / max_hp_sum

    def read_hp(self, start):
        """Read a 16-bit HP value (big-endian) from memory."""
        return 256 * self.read_m(start) + self.read_m(start + 1)

    def bit_count(self, bits):
        """Count set bits (built-in since Python 3.10)."""
        return bin(bits).count("1")

    def fourier_encode(self, val):
        """Fourier positional encoding: sin(val * 2^i) for i in [0, enc_freqs)."""
        return np.sin(val * 2 ** np.arange(self.enc_freqs))

    def update_map_progress(self):
        """Track furthest map location reached (ordered by game progression)."""
        map_idx = self.read_m(0xD35E)
        self.max_map_progress = max(self.max_map_progress, self.get_map_progress(map_idx))

    def get_map_progress(self, map_idx):
        """Return progress index for known map locations, -1 for unknown."""
        if map_idx in self.essential_map_locations.keys():
            return self.essential_map_locations[map_idx]
        else:
            return -1
