from pathlib import Path
import threading

import numpy as np
import pytest
from websockets.sync.server import serve

from examples.hv1.adapter import BoundedPolicyClient


@pytest.mark.parametrize("respond", [True, False])
def test_policy_wire_response_and_timeout(monkeypatch, respond):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "packages/openpi-client/src"))
    from openpi_client import msgpack_numpy

    release = threading.Event()
    seen = []

    def handle(ws):
        ws.send(msgpack_numpy.packb({"model": "SYNTHETIC"}))
        seen.append(msgpack_numpy.unpackb(ws.recv()))
        if respond:
            ws.send(msgpack_numpy.packb({"actions": np.zeros((4, 3), np.float32)}))
        release.wait(timeout=3)

    server = serve(handle, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = None
    try:
        client = BoundedPolicyClient(f"ws://127.0.0.1:{server.socket.getsockname()[1]}", timeout_s=2)
        assert client.metadata == {"model": "SYNTHETIC"}
        client.timeout = 1 if respond else 0.1
        observation = {"state": np.zeros(3, np.float32), "prompt": "synthetic only"}
        if respond:
            assert client.infer(observation)["actions"].shape == (4, 3)
        else:
            with pytest.raises(TimeoutError):
                client.infer(observation)
        assert len(seen) == 1
    finally:
        release.set()
        if client is not None:
            client.close()
        server.shutdown()
        thread.join(timeout=3)
