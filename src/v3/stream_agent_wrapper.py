"""
WebSocket streaming wrapper for broadcasting agent coordinates to a live map.

Wraps a Gym environment and periodically sends player position data
to a WebSocket server for real-time visualization at:
https://pwhiddy.github.io/pokerl-map-viz/

Uses PyBoy v2 memory API (self.emulator.memory[addr]).
"""

import asyncio
import websockets
import json

import gymnasium as gym

# Game Boy memory addresses for player position
X_POS_ADDRESS, Y_POS_ADDRESS = 0xD362, 0xD361
MAP_N_ADDRESS = 0xD35E


class StreamWrapper(gym.Wrapper):
    """Gym wrapper that broadcasts (x, y, map) coords via WebSocket."""

    def __init__(self, env, stream_metadata={}):
        super().__init__(env)
        self.ws_address = "wss://transdimensional.xyz/broadcast"
        self.stream_metadata = stream_metadata
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.websocket = None
        self.loop.run_until_complete(
            self.establish_wc_connection()
        )
        self.upload_interval = 300  # send coords every 300 steps
        self.steam_step_counter = 0
        self.env = env
        self.coord_list = []
        # Find the emulator instance on the wrapped env
        if hasattr(env, "pyboy"):
            self.emulator = env.pyboy
        elif hasattr(env, "game"):
            self.emulator = env.game
        else:
            raise Exception("Could not find emulator!")

    def step(self, action):
        """Step the env and periodically broadcast position data."""
        x_pos = self.emulator.memory[X_POS_ADDRESS]
        y_pos = self.emulator.memory[Y_POS_ADDRESS]
        map_n = self.emulator.memory[MAP_N_ADDRESS]
        self.coord_list.append([x_pos, y_pos, map_n])

        if self.steam_step_counter >= self.upload_interval:
            self.stream_metadata["extra"] = f"coords: {len(self.env.seen_coords)}"
            self.loop.run_until_complete(
                self.broadcast_ws_message(
                    json.dumps(
                        {
                            "metadata": self.stream_metadata,
                            "coords": self.coord_list
                        }
                    )
                )
            )
            self.steam_step_counter = 0
            self.coord_list = []

        self.steam_step_counter += 1

        return self.env.step(action)

    async def broadcast_ws_message(self, message):
        """Send a message over WebSocket, reconnecting if needed."""
        if self.websocket is None:
            await self.establish_wc_connection()
        if self.websocket is not None:
            try:
                await self.websocket.send(message)
            except websockets.exceptions.WebSocketException as e:
                self.websocket = None

    async def establish_wc_connection(self):
        """Establish WebSocket connection (silently fails if server unavailable)."""
        try:
            self.websocket = await websockets.connect(self.ws_address)
        except Exception:
            self.websocket = None
