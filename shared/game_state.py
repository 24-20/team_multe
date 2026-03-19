from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Bot:
    id: int
    position: tuple[int, int]
    inventory: list[str]


@dataclass
class Item:
    id: str
    type: str
    position: tuple[int, int]


@dataclass
class Order:
    id: str
    items_required: list[str]
    items_delivered: list[str]
    complete: bool
    status: str  # "active" | "preview"

    @property
    def remaining(self) -> list[str]:
        delivered = list(self.items_delivered)
        remaining = list(self.items_required)
        for item in delivered:
            if item in remaining:
                remaining.remove(item)
        return remaining


@dataclass
class Grid:
    width: int
    height: int
    walls: list[tuple[int, int]]


@dataclass
class GameState:
    round: int
    max_rounds: int
    action_status: str
    grid: Grid
    bots: list[Bot]
    items: list[Item]
    orders: list[Order]
    drop_off: tuple[int, int] | None
    drop_off_zones: list[tuple[int, int]]
    score: int
    active_order_index: int
    total_orders: int

    @classmethod
    def from_json(cls, data: dict) -> "GameState":
        grid = Grid(
            width=data["grid"]["width"],
            height=data["grid"]["height"],
            walls=[tuple(w) for w in data["grid"]["walls"]],
        )
        bots = [
            Bot(id=b["id"], position=tuple(b["position"]), inventory=b["inventory"])
            for b in data["bots"]
        ]
        items = [
            Item(id=i["id"], type=i["type"], position=tuple(i["position"]))
            for i in data["items"]
        ]
        orders = [
            Order(
                id=o["id"],
                items_required=o["items_required"],
                items_delivered=o["items_delivered"],
                complete=o["complete"],
                status=o["status"],
            )
            for o in data["orders"]
        ]
        drop_off = tuple(data["drop_off"]) if "drop_off" in data else None
        drop_off_zones = [tuple(z) for z in data.get("drop_off_zones", [])]

        return cls(
            round=data["round"],
            max_rounds=data["max_rounds"],
            action_status=data["action_status"],
            grid=grid,
            bots=bots,
            items=items,
            orders=orders,
            drop_off=drop_off,
            drop_off_zones=drop_off_zones,
            score=data["score"],
            active_order_index=data["active_order_index"],
            total_orders=data["total_orders"],
        )

    @property
    def active_order(self) -> Optional[Order]:
        for o in self.orders:
            if o.status == "active":
                return o
        return None

    @property
    def preview_order(self) -> Optional[Order]:
        for o in self.orders:
            if o.status == "preview":
                return o
        return None

    @property
    def all_drop_offs(self) -> list[tuple[int, int]]:
        if self.drop_off_zones:
            return self.drop_off_zones
        if self.drop_off:
            return [self.drop_off]
        return []

    def item_by_type(self, item_type: str) -> Optional[Item]:
        for item in self.items:
            if item.type == item_type:
                return item
        return None
