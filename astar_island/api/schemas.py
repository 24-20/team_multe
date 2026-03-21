from typing import Any, TypedDict


class SettlementInfo(TypedDict):
    x: int
    y: int
    has_port: bool
    alive: bool


class SeedState(TypedDict):
    grid: list[list[int]]       # height x width raw terrain codes
    settlements: list[SettlementInfo]


class RoundMeta(TypedDict):
    id: str
    round_number: int
    status: str                 # "active" | "scoring" | "completed"
    opened_at: str
    closed_at: str | None
    queries_used: int
    seeds_count: int
    map_width: int
    map_height: int


class RoundDetail(TypedDict):
    id: str
    round_number: int
    status: str
    map_width: int
    map_height: int
    seeds_count: int
    initial_states: list[SeedState]
    queries_used: int


class SimulateResult(TypedDict):
    grid: list[list[int]]       # viewport_h x viewport_w terrain after simulation
    settlements: list[dict]     # settlements in viewport with full stats
    viewport: dict              # {x, y, w, h}


class SubmitResult(TypedDict):
    status: str
    message: str | None
    seed_index: int
    round_id: str
