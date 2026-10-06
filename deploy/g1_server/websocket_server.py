"""WebSocket endpoint of the G1 policy protocol (`protocol.py`), compatible with `tmp/g1-client/g1_client/policy_client.py`.

Per connection: send `ServerMetadata`, then answer each msgpack request in order. Observations go through
`PolicyManager.serve`, which applies the output mode and the action safety check. A failed request, and an
observation in the stop mode, is answered with a text frame holding the error; the client raises it as `RuntimeError`.

The event loop runs on its own thread (`serve_forever`). `change_port` and `disconnect_all` are called from other
threads (the web page): a port change binds the new port first, so a port that cannot be bound leaves the old
listener running; then the old listener and its connections are closed.
"""

import asyncio
import logging
import socket
import threading
import time
import traceback
from dataclasses import dataclass

from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from g1_server.msgpack_numpy import packb, unpackb
from g1_server.policy_manager import PolicyManager, ServingStopped
from g1_server.protocol import ActionResponse, PreviewRequest, ResetRequest, parse_request

log = logging.getLogger("g1_server.websocket")
PORT_RANGE = (1024, 65535)
CALL_TIMEOUT_S = 15.0


@dataclass(frozen=True)
class ClientConnection:
    address: str  # host:port of the client
    connected_at: float  # time.time()
    requests: int


class PolicyWebSocketServer:
    def __init__(self, manager: PolicyManager, host: str, port: int) -> None:
        self.manager = manager
        self.host = host
        self.port = port
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: Server | None = None
        self._started = threading.Event()
        self._clients_lock = threading.Lock()
        self._clients: dict[ServerConnection, ClientConnection] = {}

    def url(self) -> str:
        return f"ws://{socket.gethostname()}:{self.port}"

    def serve_forever(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._server = self._loop.run_until_complete(self._listen(self.port))
        self._started.set()
        self._loop.run_forever()

    async def _listen(self, port: int) -> Server:
        server = await serve(self._handle, self.host, port, compression=None, max_size=None, ping_interval=None)
        log.info("policy WebSocket listening on ws://%s:%d", self.host, port)
        return server

    def _call(self, coroutine) -> object:
        if not self._started.wait(timeout=CALL_TIMEOUT_S):
            raise RuntimeError("the WebSocket server is not running")
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result(timeout=CALL_TIMEOUT_S)

    def change_port(self, port: int) -> None:
        """Moves the listener to `port`; raises (and keeps the old listener) if it cannot be bound."""
        if not PORT_RANGE[0] <= port <= PORT_RANGE[1]:
            raise ValueError(f"port must be in {PORT_RANGE}, got {port}")
        if port == self.port:
            return
        self._call(self._move(port))

    async def _move(self, port: int) -> None:
        new_server = await self._listen(port)  # OSError here leaves the old listener untouched
        old_server, old_port = self._server, self.port
        self._server, self.port = new_server, port
        old_server.close(close_connections=True, reason="the policy server moved to another port")
        await old_server.wait_closed()
        log.info("listener moved from port %d to %d", old_port, port)

    def connections(self) -> list[ClientConnection]:
        with self._clients_lock:
            return sorted(self._clients.values(), key=lambda c: c.connected_at)

    def disconnect_all(self) -> int:
        """Closes every client connection; returns how many were open. A closed client raises and runs its cleanup."""
        return self._call(self._close_all())

    async def _close_all(self) -> int:
        with self._clients_lock:
            connections = list(self._clients)
        for connection in connections:
            await connection.close(reason="closed from the policy server web page")
        return len(connections)

    async def _handle(self, connection: ServerConnection) -> None:
        client = f"{connection.remote_address[0]}:{connection.remote_address[1]}"
        log.info("client connected: %s", client)
        with self._clients_lock:
            self._clients[connection] = ClientConnection(address=client, connected_at=time.time(), requests=0)
        try:
            await connection.send(packb(self.manager.metadata().to_message()))
            async for message in connection:
                with self._clients_lock:
                    record = self._clients[connection]
                    self._clients[connection] = ClientConnection(record.address, record.connected_at,
                                                                 record.requests + 1)
                try:
                    request = parse_request(unpackb(message))
                    if isinstance(request, ResetRequest):
                        await connection.send(packb({}))
                        continue
                    if isinstance(request, PreviewRequest):
                        await asyncio.to_thread(self.manager.preview, request, client)
                        await connection.send(packb({}))
                        continue
                    served = await asyncio.to_thread(self.manager.serve, request, client)
                    output = served.output
                    response = ActionResponse(actions=served.actions, timing=output.timing, model=output.model,
                                              checkpoint_step=output.step,
                                              num_inference_steps=output.num_inference_steps, seed=output.seed,
                                              output_mode=served.mode.value, actions_source=served.source,
                                              safety_ok=served.verdict.ok, safety_reasons=served.verdict.reasons)
                    await connection.send(packb(response.to_message()))
                except ConnectionClosed:
                    raise
                except ServingStopped as stopped:
                    log.warning("request from %s rejected: %s", client, stopped)
                    await connection.send(f"ServingStopped: {stopped}")
                except Exception:
                    log.exception("request from %s failed", client)
                    await connection.send(traceback.format_exc())
        except ConnectionClosed:
            pass
        except Exception:
            log.exception("connection %s failed", client)
        finally:
            with self._clients_lock:
                self._clients.pop(connection, None)
            log.info("client disconnected: %s", client)
