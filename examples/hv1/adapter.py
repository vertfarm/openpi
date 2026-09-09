"""Shadow-only policy boundary. There is deliberately NO ROS publisher or live mode.

Real actuation is unavailable until the recorder/controller contract is accepted.
This module can be imported in the lightweight operations environment.
"""

import time
from urllib.parse import urlparse

import numpy as np

from .workflow import ContractError
from .workflow import digest
from .workflow import validate_profile


class ShadowAdapter:
    def __init__(self, profile, metadata, *, max_observation_age_s, response_timeout_s):
        self.profile = validate_profile(profile)
        if max_observation_age_s <= 0 or response_timeout_s <= 0:
            raise ContractError("explicit positive time limits required")
        expected = {
            "profile_sha256": digest(profile),
            "action_names": profile["action"]["names"],
            "action_units": profile["action"]["units"],
            "output_action_space": "commanded_target",
        }
        if any(metadata.get(k) != value for k, value in expected.items()):
            raise ContractError("policy metadata/profile mismatch")
        self.max_age, self.timeout = max_observation_age_s, response_timeout_s
        self.faulted = False

    def validate(self, actions, *, observation_monotonic, request_monotonic, now=None):
        if self.faulted:
            raise ContractError("adapter fault latched; recreate with verified metadata/new observations")
        now = time.monotonic() if now is None else now
        try:
            if not all(np.isfinite([now, observation_monotonic, request_monotonic])):
                raise ContractError("invalid timestamps")
            if not 0 <= now - observation_monotonic <= self.max_age:
                raise ContractError("observation stale/future or clock-domain mismatch")
            if not 0 <= now - request_monotonic <= self.timeout:
                raise ContractError("inference response late/future")
            values = np.asarray(actions, dtype=np.float32)
            if values.shape != (self.profile["action_horizon"], len(self.profile["action"]["names"])):
                raise ContractError("action chunk shape mismatch")
            if not np.isfinite(values).all():
                raise ContractError("nonfinite action")
            return {
                "mode": "shadow",
                "sent": False,
                "candidate_first_action": values[0].tolist(),
                "note": "shape/time validation only; physical limits and ownership NOT qualified",
            }
        except (ContractError, ValueError, TypeError):
            self.faulted = True
            raise


class BoundedPolicyClient:
    """OpenPI wire-compatible client with bounded handshake/response; no retries/rearm."""

    def __init__(self, uri, *, timeout_s):
        from openpi_client import msgpack_numpy
        from websockets.sync.client import connect

        parsed = urlparse(uri)
        if parsed.scheme != "ws" or parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ContractError("v1 policy client only connects to loopback")
        if not 0 < timeout_s <= 60:
            raise ContractError("timeout must be 0..60 seconds")
        self.timeout, self.codec = timeout_s, msgpack_numpy
        self.connection = connect(
            uri, open_timeout=timeout_s, close_timeout=1, compression=None, max_size=64 * 1024 * 1024
        )
        try:
            self.metadata = msgpack_numpy.unpackb(self.connection.recv(timeout=timeout_s))
        except Exception:
            self.connection.close()
            raise

    def infer(self, observation):
        try:
            self.connection.send(self.codec.packb(observation))
            response = self.connection.recv(timeout=self.timeout)
            if isinstance(response, str):
                raise ContractError("policy server returned an error")
            return self.codec.unpackb(response)
        except Exception:
            self.connection.close()
            raise

    def close(self):
        self.connection.close()
