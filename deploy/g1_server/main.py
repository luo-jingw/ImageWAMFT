"""G1 policy server: the WebSocket endpoint for the robot client (`tmp/g1-client/fastwam/main.py`) and the web page.

The GPU is chosen with CUDA_VISIBLE_DEVICES; see `launch/g1_server.sh` and `docs/g1_policy_server.md`.

  PYTHONPATH=deploy CUDA_VISIBLE_DEVICES=4 .venv/bin/python deploy/g1_server/main.py --models flux2_4b,qwenimage21,flux2_9b
"""

import argparse
import logging
import socket
import threading
from pathlib import Path

import torch

from g1_server.model_registry import MODEL_SPECS, available_steps, spec_by_name
from g1_server.policy_manager import PolicyManager
from g1_server.web_ui import build_web_ui
from g1_server.websocket_server import PolicyWebSocketServer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default="flux2_4b",
                        help=f"comma-separated models loaded at start (latest step), the first one active; "
                             f"choices: {[spec.name for spec in MODEL_SPECS]}")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--ws-port", type=int, default=8000)
    parser.add_argument("--ui-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="also publish the web page through a Gradio share link")
    parser.add_argument("--auth-file", type=Path, default=None, help="file holding user:password for the web page")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    manager = PolicyManager(torch.device("cuda:0"))
    names = [name for name in args.models.split(",") if name]
    for name in names:
        report = manager.load(name, available_steps(spec_by_name(name))[-1])
        logging.info("loaded %s step %d in %.1f s (%.1f GiB allocated)", report.name, report.step, report.seconds,
                     report.gpu_memory_allocated_gib)
    manager.activate(names[0])
    auth = None
    if args.auth_file is not None:
        user, password = args.auth_file.read_text().strip().split(":", 1)
        auth = (user, password)
    server = PolicyWebSocketServer(manager, args.host, args.ws_port)
    thread = threading.Thread(target=server.serve_forever, name="policy-websocket", daemon=True)
    thread.start()
    page = build_web_ui(manager, f"ws://{socket.gethostname()}:{args.ws_port}")
    page.queue(default_concurrency_limit=4).launch(server_name=args.host, server_port=args.ui_port, share=args.share,
                                                   auth=auth)


if __name__ == "__main__":
    main()
