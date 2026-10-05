"""WebSocket endpoint of the G1 policy protocol (`protocol.py`), compatible with `tmp/g1-client/g1_client/policy_client.py`.

Per connection: send `ServerMetadata`, then answer each msgpack request in order. Inference runs in a worker thread
through `PolicyManager.infer`. A failed request is answered with a text frame holding the error, which the client
raises as `RuntimeError`.
"""

import asyncio
import logging
import traceback

from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from g1_server.msgpack_numpy import packb, unpackb
from g1_server.policy_manager import PolicyManager
from g1_server.protocol import ActionResponse, ResetRequest, parse_request

log = logging.getLogger("g1_server.websocket")


class PolicyWebSocketServer:
    def __init__(self, manager: PolicyManager, host: str, port: int) -> None:
        self.manager = manager
        self.host = host
        self.port = port

    def serve_forever(self) -> None:
        asyncio.run(self._serve())

    async def _serve(self) -> None:
        async with serve(self._handle, self.host, self.port, compression=None, max_size=None,
                         ping_interval=None) as server:
            log.info("policy WebSocket listening on ws://%s:%d", self.host, self.port)
            await server.serve_forever()

    async def _handle(self, connection: ServerConnection) -> None:
        client = f"{connection.remote_address[0]}:{connection.remote_address[1]}"
        log.info("client connected: %s", client)
        try:
            await connection.send(packb(self.manager.metadata().to_message()))
            async for message in connection:
                try:
                    request = parse_request(unpackb(message))
                    if isinstance(request, ResetRequest):
                        await connection.send(packb({}))
                        continue
                    output = await asyncio.to_thread(self.manager.infer, request, client)
                    response = ActionResponse(actions=output.actions, timing=output.timing, model=output.model,
                                              checkpoint_step=output.step,
                                              num_inference_steps=output.num_inference_steps, seed=output.seed)
                    await connection.send(packb(response.to_message()))
                except ConnectionClosed:
                    raise
                except Exception:
                    log.exception("request from %s failed", client)
                    await connection.send(traceback.format_exc())
        except ConnectionClosed:
            pass
        except Exception:
            log.exception("connection %s failed", client)
        finally:
            log.info("client disconnected: %s", client)
