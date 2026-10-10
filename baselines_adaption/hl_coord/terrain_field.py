"""Shared continuous terrain-relevance field A_high(v) (idea doc §4.2-4.3).

The high-altitude leader writes VLM-judged terrain relevance scores into
this field as it sweeps the area; the low-altitude follower reads it back,
sampled at its own (finer) grid resolution, to bias its APEX maps.
Both agents run in the same Python process/loop, so "broadcasting" the
field is just sharing one object by reference — no serialization needed.

Sparse dict representation (not a dense array): the drone can wander
anywhere without needing to know the map bounds ahead of time, and the
region grid is coarse (default 50m/cell) so the dict stays tiny.

§4.3's Q-value-style feedback decay ("A_high(region) -= eta * [visited but
no new evidence]") is implemented as ``decay_at`` — call it from the
follower each step against the region it currently occupies.
"""
import math
from typing import Dict, Optional, Tuple


class TerrainField:
    """Sparse coarse-grid field of terrain-relevance scores in [0, 1]."""

    def __init__(self, cell_size: float = 50.0):
        self.cell_size = cell_size
        self._scores: Dict[Tuple[int, int], float] = {}
        # step index of the most recent write per region (for recency weighting)
        self._last_write: Dict[Tuple[int, int], int] = {}
        self._write_count = 0
        # how many times the follower has occupied each region — used to make
        # the "best region" self-extinguishing: once the follower has spent
        # time somewhere, it stops being the target and the next unexplored
        # high-terrain region takes over.
        self._visits: Dict[Tuple[int, int], int] = {}

    def region_key(self, world_x: float, world_y: float) -> Tuple[int, int]:
        return (
            int(math.floor(world_x / self.cell_size)),
            int(math.floor(world_y / self.cell_size)),
        )

    def region_center(self, key: Tuple[int, int]) -> Tuple[float, float]:
        return ((key[0] + 0.5) * self.cell_size, (key[1] + 0.5) * self.cell_size)

    def update(self, world_x: float, world_y: float, score: float) -> None:
        """Record a VLM terrain-relevance observation. Keeps the max score
        seen so far for a region (evidence accumulates, weak later
        observations don't overwrite a strong earlier one)."""
        key = self.region_key(world_x, world_y)
        score = max(0.0, min(1.0, float(score)))
        self._write_count += 1
        self._last_write[key] = self._write_count
        if score > self._scores.get(key, 0.0):
            self._scores[key] = score

    def get(self, key: Tuple[int, int], default: float = 0.0) -> float:
        return self._scores.get(key, default)

    def decay_at(self, world_x: float, world_y: float, eta: float,
                 radius_cells: int = 0) -> None:
        """Reduce the score of the region containing (world_x, world_y), and
        optionally its neighbours out to ``radius_cells``, at a fraction of
        ``eta`` that falls off with distance — the follower has covered this
        area, so trust the leader's prior here less going forward."""
        if eta <= 0.0:
            return
        cx, cy = self.region_key(world_x, world_y)
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                key = (cx + dx, cy + dy)
                if key not in self._scores:
                    continue
                falloff = 1.0 / (1.0 + abs(dx) + abs(dy))
                self._scores[key] = max(0.0, self._scores[key] - eta * falloff)

    def note_visit(self, world_x: float, world_y: float) -> None:
        """Record that the follower currently occupies this region."""
        key = self.region_key(world_x, world_y)
        self._visits[key] = self._visits.get(key, 0) + 1

    def mark_exhausted(self, key: Tuple[int, int], radius_cells: int = 1) -> None:
        """The follower has searched the ROI centred on ``key`` and found
        nothing new — zero it and its neighbours so ``best_region`` moves on."""
        cx, cy = key
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                k = (cx + dx, cy + dy)
                if k in self._scores:
                    self._scores[k] = 0.0

    def visits(self, key: Tuple[int, int]) -> int:
        return self._visits.get(key, 0)

    def frac_above(self, threshold: float) -> float:
        """Fraction of populated cells whose score is >= threshold. High on
        maps where the leader's terrain match lights up everywhere (k2/aotai
        'summit') and the prior is therefore uninformative; low where the
        match is localised (lopnur). Used to damp fusion strength when the
        field is 'flooded'."""
        n = len(self._scores)
        if n == 0:
            return 0.0
        hi = sum(1 for v in list(self._scores.values()) if v >= threshold)
        return hi / n

    def best_region(self, min_score: float = 0.0,
                    recency_halflife: float = 0.0,
                    visit_penalty: float = 0.0
                    ) -> Optional[Tuple[Tuple[int, int], float, Tuple[float, float]]]:
        """Highest-scoring region above ``min_score`` (compared on the *raw*
        score, so ``min_score`` gates real terrain evidence, not the
        down-weighted value).

        Returns (key, effective_score, (center_x, center_y)) or None.
        - ``recency_halflife`` > 0: older writes are down-weighted so the
          follower follows the leader's *current* best guess.
        - ``visit_penalty`` > 0: regions the follower has already occupied are
          down-weighted by ``exp(-visit_penalty * visit_count)`` so the
          target advances to the next unexplored high-terrain region instead
          of the follower camping on the first one (the fixation bug).
        """
        best = None
        # snapshot: the leader may write from a background VLM thread while
        # the follower iterates here on the main thread.
        for key, score in list(self._scores.items()):
            if score < min_score:
                continue
            eff = score
            if recency_halflife > 0.0 and key in self._last_write:
                age = self._write_count - self._last_write[key]
                eff *= 0.5 ** (age / recency_halflife)
            if visit_penalty > 0.0:
                eff *= math.exp(-visit_penalty * self._visits.get(key, 0))
            if best is None or eff > best[1]:
                best = (key, eff, self.region_center(key))
        return best

    def __len__(self) -> int:
        return len(self._scores)
