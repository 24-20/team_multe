import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod

import websockets
from dotenv import load_dotenv

from shared.game_state import GameState

load_dotenv()

logger = logging.getLogger(__name__)


class BaseBot(ABC):
    """Base class for all WebSocket-based bots in the competition."""

    WS_BASE = "wss://game.ainm.no/ws"

    def __init__(self, token: str):
        # Token is single-use per game session — get it by clicking Play on app.ainm.no
        self.token = token

    @property
    def ws_url(self) -> str:
        return f"{self.WS_BASE}?token={self.token}"

    @abstractmethod
    def decide_actions(self, state: GameState) -> list[dict]:
        """Return list of action dicts for this round, e.g. [{"bot": 0, "action": "move_up"}]"""

    async def run(self):
        logger.info("Connecting to %s", self.ws_url)
        async with websockets.connect(self.ws_url) as ws:
            async for raw in ws:
                data = json.loads(raw)
                msg_type = data.get("type")

                if msg_type == "game_over":
                    logger.info(
                        "Game over! Score=%s  Rounds=%s  Items=%s  Orders=%s",
                        data["score"],
                        data["rounds_used"],
                        data["items_delivered"],
                        data["orders_completed"],
                    )
                    return data

                if msg_type == "game_state":
                    state = GameState.from_json(data)
                    actions = self.decide_actions(state)
                    payload = {"round": state.round, "actions": actions}
                    await ws.send(json.dumps(payload))

    def play(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        return asyncio.run(self.run())
