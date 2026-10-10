"""Per-vehicle proxy around a shared AirSim client.

Every existing agent (``BaseAgent`` subclasses, ``DetectionModule``, ...) was
written assuming a single drone and never passes ``vehicle_name`` to the
AirSim client. To run several of them at once against a multi-vehicle
AirSim session without touching any of that code, wrap the shared client in
a :class:`VehicleClientProxy` bound to one vehicle name and hand the proxy
to the agent instead of the real client — every call that supports
``vehicle_name`` gets it injected automatically.
"""
import inspect


class VehicleClientProxy:
    def __init__(self, client, vehicle_name):
        self._client = client
        self._vehicle_name = vehicle_name
        self._accepts_vehicle_name_cache = {}

    def __getattr__(self, name):
        attr = getattr(self._client, name)
        if not callable(attr):
            return attr

        accepts = self._accepts_vehicle_name_cache.get(name)
        if accepts is None:
            try:
                accepts = "vehicle_name" in inspect.signature(attr).parameters
            except (TypeError, ValueError):
                accepts = False
            self._accepts_vehicle_name_cache[name] = accepts

        if not accepts:
            return attr

        def _bound(*args, **kwargs):
            kwargs.setdefault("vehicle_name", self._vehicle_name)
            return attr(*args, **kwargs)
        return _bound
