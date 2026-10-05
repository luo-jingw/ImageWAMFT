"""G1 policy server: the WebSocket endpoint for the robot client (`tmp/g1-client/fastwam/main.py`) and the web page.

The GPU is chosen with CUDA_VISIBLE_DEVICES; see `launch/g1_server.sh` and `docs/g1_policy_server.md`.

  PYTHONPATH=deploy CUDA_VISIBLE_DEVICES=4 .venv/bin/python deploy/g1_server/main.py --model flux2_4b
"""

import argparse
import logging
import socket
import threading

import torch

from g1_server.model_registry import MODEL_SPECS, available_steps, spec_by_name
from g1_server.policy_manager import PolicyManager
from g1_server.web_ui import build_web_ui
from g1_server.websocket_server import PolicyWebSocketServer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=[spec.name for spec in MODEL_SPECS], default="flux2_4b")
    parser.add_argument("--step", type=int, default=None, help="weights step; default: the latest saved")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--ws-port", type=int, default=8000)
    parser.add_argument("--ui-port", type=int, default=7860)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    manager = PolicyManager(torch.device("cuda:0"))
    step = args.step if args.step is not None else available_steps(spec_by_name(args.model))[-1]
    report = manager.load(args.model, step)
    logging.info("loaded %s step %d in %.1f s (%.1f GiB allocated)", report.name, report.step, report.seconds,
                 report.gpu_memory_allocated_gib)
    server = PolicyWebSocketServer(manager, args.host, args.ws_port)
    thread = threading.Thread(target=server.serve_forever, name="policy-websocket", daemon=True)
    thread.start()
    page = build_web_ui(manager, f"ws://{socket.gethostname()}:{args.ws_port}")
    page.queue(default_concurrency_limit=4).launch(server_name=args.host, server_port=args.ui_port, share=False)


if __name__ == "__main__":
    main()
