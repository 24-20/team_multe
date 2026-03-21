"""Astar Island — Viking Civilisation Prediction.

Main components:

    from astar_island.api import AstarClient
    from astar_island.online import RoundState, QueryPolicy, infer_full_map, submit_all_seeds
    from astar_island.storage import RoundStore
    from astar_island.utils.math import floor_renorm, weighted_kl_score
    from astar_island.validators import validate_prediction_tensor

Run a live round:
    python -m astar_island.scripts.run_online_round --mode live
    python -m astar_island.scripts.run_online_round --mode dryrun

Backfill history:
    python -m astar_island.scripts.backfill_rounds --build-dataset

Post-round analysis and retrain:
    python -m astar_island.scripts.post_round_analysis --round_id ID --retrain

Replay a stored round offline:
    python -m astar_island.scripts.replay_round --round_id ID
"""
