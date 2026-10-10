"""
Utility functions shared across the ASAR source modules.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

class CoordinateConverter:
    """Convert between Unreal Engine (UE) and AirSim (NED) coordinate systems.

    UE positions are in centimetres; AirSim uses metres in a North-East-Down
    (NED) frame where **negative Z is upward**.

    Conversion formula for map origin ``(a, b, c)`` (UE cm):

    .. math::
        x' = (x - a) / 100, \\quad
        y' = (y - b) / 100, \\quad
        z' = (c - z) / 100
    """

    PLAYER_STARTS: Dict[str, Tuple[float, float, float]] = {
        "aotai2800": (0, 0, -16000),
        "lopnur":    (0, 0, 0),
        "k2":        (0, 0, 34000),
        "dapeng":    (0, 0, -4000),
    }

    @staticmethod
    def to_airsim(
        map_name: str,
        ue_pos_dict: Dict[str, float],
    ) -> Dict[str, float]:
        """Convert a UE position dict to AirSim NED metres.

        Args:
            map_name:    Name of the UE5 map (e.g. ``"aotai2800"``).
            ue_pos_dict: UE position as ``{"x": …, "y": …, "z": …}`` (cm).

        Returns:
            AirSim position as ``{"x": …, "y": …, "z": …}`` (metres, NED).
        """
        if map_name not in CoordinateConverter.PLAYER_STARTS:
            print(f"Warning: Map {map_name} not in config, utilizing (0,0,0) as origin.")
            a, b, c = 0.0, 0.0, 0.0
        else:
            a, b, c = CoordinateConverter.PLAYER_STARTS[map_name]

        x = ue_pos_dict["x"]
        y = ue_pos_dict["y"]
        z = ue_pos_dict["z"]

        return {
            "x": (x - a) / 100.0,
            "y": (y - b) / 100.0,
            "z": (c - z) / 100.0,
        }


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def calculate_distance(
    pos1: Optional[Union[Dict[str, float], List[float], Tuple[float, ...]]],
    pos2: Optional[Union[Dict[str, float], List[float], Tuple[float, ...]]],
) -> float:
    """Return the Euclidean distance between two 3-D positions.

    Positions may be dicts (``{"x":…,"y":…,"z":…}``) or array-likes.
    Returns ``float('inf')`` if either argument is ``None``.

    Args:
        pos1: First position.
        pos2: Second position.

    Returns:
        Euclidean distance in the same units as the inputs.
    """
    if pos1 is None or pos2 is None:
        return float("inf")

    def _to_np(p: Union[Dict[str, float], List[float], Tuple[float, ...]]) -> np.ndarray:
        if isinstance(p, dict):
            return np.array([p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)],
                            dtype=float)
        return np.asarray(p, dtype=float)

    return float(np.linalg.norm(_to_np(pos1) - _to_np(pos2)))
