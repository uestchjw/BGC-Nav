import time
import numpy as np
import airsim
from typing import List, Dict, Any, Optional


# Spatial dedup: clues with the same name within this distance are merged.
_CLUE_DEDUP_DIST = 5.0  # metres


class BaseAgent:
    """Abstract base class for all ASAR drone agents.

    Subclasses must implement :meth:`run_step`.  The platform calls
    ``run_step`` once per control loop iteration and uses the reporting
    methods below to collect results.

    Args:
        client: Connected :class:`airsim.MultirotorClient` instance.
        task: Task dictionary loaded from ``task_data*.json``.
    """

    def __init__(self, client: "airsim.MultirotorClient", task: Dict[str, Any]) -> None:
        self.client = client
        self.task = task
        self._found_clues: List[Dict[str, Any]] = []
        self._reported_positions: List[Dict[str, float]] = []
        self._is_finished: bool = False

        # 飞行轨迹记录
        self._trajectory: List[Dict[str, Any]] = []
        self._trajectory_interval: int = 10  # 默认每10步记录一次，子类可覆盖
        self._internal_step_count: int = 0

        # 辅助信息
        self.start_time: float = time.time()

    def run_step(self, current_time: float) -> None:
        """Execute one control loop step.

        Override this method in subclasses to implement the
        perception → reasoning → action pipeline.

        Args:
            current_time: Wall-clock time at the start of this step (seconds).

        Raises:
            NotImplementedError: Always, if not overridden.
        """
        raise NotImplementedError("Please implement run_step in your subclass")

    # --- 汇报接口 (API) ---

    def report_clue(self, clue_name: str,
                    position: Optional[Dict[str, float]] = None) -> None:
        """Report a detected clue object with its estimated position.

        Deduplication: a clue with the same name within
        :data:`_CLUE_DEDUP_DIST` metres of an existing report is ignored.

        Args:
            clue_name: Human-readable name of the discovered clue
                       (e.g. ``"Backpack"``).
            position:  AirSim NED coordinates ``{"x": …, "y": …, "z": …}``
                       (metres).  Required for CDS spatial evaluation.
        """
        if position is not None:
            for existing in self._found_clues:
                if existing["name"] == clue_name and existing["position"] is not None:
                    d = np.linalg.norm([
                        position["x"] - existing["position"]["x"],
                        position["y"] - existing["position"]["y"],
                        position["z"] - existing["position"]["z"],
                    ])
                    if d < _CLUE_DEDUP_DIST:
                        return
        else:
            # No position: fall back to name-only dedup
            if any(c["name"] == clue_name for c in self._found_clues):
                return

        entry = {"name": clue_name, "position": position}
        self._found_clues.append(entry)
        print(f"[Agent] Found Clue: {clue_name}  pos={position}")

    def report_target(self, position: Dict[str, float]) -> None:
        """Report the estimated world position of a human victim.

        Args:
            position: AirSim NED coordinates as ``{"x": …, "y": …, "z": …}`` (metres).
        """
        print(f"[Agent] Reported Target at: {position}")
        self._reported_positions.append(position)

    def finish_mission(self) -> None:
        """Signal the platform that the mission is complete.

        After this call :meth:`is_finished` returns ``True`` and the
        platform will exit the control loop at the next iteration.
        """
        print("[Agent] Mission Finished by Agent.")
        self._is_finished = True

    # --- 轨迹记录接口 (Trajectory API) ---

    def record_position(self, step: int,
                        position: Dict[str, float],
                        extra: Optional[Dict[str, Any]] = None) -> None:
        """Record the drone's current position as a trajectory waypoint.

        Agents should call this method every N steps (controlled by
        ``_trajectory_interval``) to log the flight path.

        Args:
            step:     Current step number.
            position: AirSim NED coordinates ``{"x": …, "y": …, "z": …}``
                      (metres).
            extra:    Optional dict of extra info to attach (e.g.
                      explored percentage, current goal).
        """
        entry: Dict[str, Any] = {
            "step": step,
            "timestamp": time.time() - self.start_time,
            "position": {
                "x": round(position["x"], 2),
                "y": round(position["y"], 2),
                "z": round(position["z"], 2),
            },
        }
        if extra:
            entry["extra"] = extra
        self._trajectory.append(entry)

    def auto_record_position(self, step: int) -> None:
        """Convenience wrapper: query AirSim and record if on schedule.

        Call this once per ``run_step``; the method automatically checks
        whether the current *step* is a multiple of
        ``_trajectory_interval`` and, if so, reads the drone state and
        records it.

        Args:
            step: Current step number (1-based).
        """
        if step % self._trajectory_interval != 0:
            return
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        self.record_position(step, {
            "x": pos.x_val,
            "y": pos.y_val,
            "z": pos.z_val,
        })

    def get_trajectory(self) -> List[Dict[str, Any]]:
        """Return the full recorded trajectory list."""
        return self._trajectory

    def set_trajectory_interval(self, n: int) -> None:
        """Set the recording interval (record once every *n* steps).

        Args:
            n: Number of steps between trajectory recordings (must be ≥ 1).
        """
        self._trajectory_interval = max(1, n)

    # --- 查询接口 (Query API) ---

    def get_task(self) -> Dict[str, Any]:
        """Return a shallow copy of the current task metadata dict."""
        return dict(self.task)

    def get_task_field(self, key: str, default: Any = None) -> Any:
        """Return one field from the current task metadata."""
        return self.task.get(key, default)

    def get_task_prompt(self, default: str = "") -> str:
        """Return the task prompt string for this episode.

        The canonical field in ``task_data*.json`` is ``prompt``.
        Fallback keys are accepted for compatibility with future
        baseline adapters.
        """
        for key in ("prompt", "task_prompt", "instruction", "description"):
            value = self.task.get(key)
            if isinstance(value, str):
                value = value.strip()
                if value:
                    return value
        return default

    def get_clues(self) -> List[str]:
        """Return the list of unique clue names reported so far (backward compat)."""
        return [c["name"] for c in self._found_clues]

    def get_clues_with_positions(self) -> List[Dict[str, Any]]:
        """Return the full clue list with names and positions."""
        return self._found_clues

    def get_reported_positions(self) -> List[Dict[str, float]]:
        """Return the list of target positions reported so far."""
        return self._reported_positions

    def is_finished(self) -> bool:
        """Return ``True`` if the agent has called :meth:`finish_mission`."""
        return self._is_finished
