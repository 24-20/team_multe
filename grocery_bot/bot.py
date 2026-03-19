"""Grocery Bot — Challenge 1 (warm-up + main competition challenge).

WebSocket game: control bots to navigate a grocery store, pick up items, and deliver orders.

Score = items_delivered × 1 + orders_completed × 5
Leaderboard = sum of best scores across all 21 maps.

To run:
    python -m grocery_bot.bot

Set AINM_TOKEN in .env first.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from shared.game_state import GameState, Bot
from shared.ws_client import BaseBot
from shared.pathfinding import next_step_towards, is_adjacent, adjacent_cells


class GroceryBot(BaseBot):
    def decide_actions(self, state: GameState) -> list[dict]:
        actions = []
        walls = set(state.grid.walls)
        bot_positions = {b.position for b in state.bots}
        active_order = state.active_order

        for bot in state.bots:
            action = self._decide_for_bot(bot, state, walls, bot_positions, active_order)
            actions.append(action)

        return actions

    def _decide_for_bot(self, bot: Bot, state: GameState, walls, bot_positions, active_order) -> dict:
        needed = active_order.remaining if active_order else []
        drop_offs = state.all_drop_offs

        # If at a drop-off with inventory → drop off
        if bot.inventory and drop_offs:
            for dz in drop_offs:
                if bot.position == dz:
                    return {"bot": bot.id, "action": "drop_off"}

        # If inventory full → head to closest drop-off
        if len(bot.inventory) >= 3 and drop_offs:
            target = min(drop_offs, key=lambda d: abs(d[0] - bot.position[0]) + abs(d[1] - bot.position[1]))
            return self._move_to(bot, target, walls, bot_positions, state)

        # If carrying items for active order → deliver
        carrying_needed = [i for i in bot.inventory if i in needed]
        if carrying_needed and drop_offs:
            target = min(drop_offs, key=lambda d: abs(d[0] - bot.position[0]) + abs(d[1] - bot.position[1]))
            return self._move_to(bot, target, walls, bot_positions, state)

        # Find an item we need
        if needed and len(bot.inventory) < 3:
            for item_type in needed:
                if any(item_type in b.inventory for b in state.bots):
                    continue
                item = state.item_by_type(item_type)
                if item:
                    if is_adjacent(bot.position, item.position):
                        return {"bot": bot.id, "action": "pick_up", "item_id": item.id}
                    return self._move_to(bot, item.position, walls, bot_positions, state, shelf=item.position)

        return {"bot": bot.id, "action": "wait"}

    def _move_to(self, bot: Bot, target: tuple, walls, bot_positions, state: GameState, shelf: tuple | None = None) -> dict:
        others = bot_positions - {bot.position}
        effective_target = target
        if shelf and shelf in walls:
            candidates = [c for c in adjacent_cells(shelf) if c not in walls]
            if candidates:
                effective_target = min(candidates, key=lambda c: abs(c[0] - bot.position[0]) + abs(c[1] - bot.position[1]))

        action = next_step_towards(bot.position, effective_target, walls, state.grid.width, state.grid.height, others)
        if action:
            return {"bot": bot.id, "action": action}
        return {"bot": bot.id, "action": "wait"}


if __name__ == "__main__":
    # Token is single-use per game session — get it from the app by clicking Play on a map.
    # Usage: python -m grocery_bot.bot <token>
    # or:    python -m grocery_bot.bot wss://game.ainm.no/ws?token=<token>  (full URL also works)
    if len(sys.argv) < 2:
        print("Usage: python -m grocery_bot.bot <token_or_ws_url>")
        sys.exit(1)
    arg = sys.argv[1]
    token = arg.split("token=")[-1] if "token=" in arg else arg
    GroceryBot(token=token).play()
