"""Replays held-out LGG100 samples through a running G1 policy server with the robot client's own code (run in `.venv`).

The client is `FastWAMPolicy` from `tmp/g1-client/fastwam/fastwam_policy.py`, unmodified. Each request carries the
three cameras through the robot camera path (`g1_client/camera_client.py` resize of the 480×640 BGR frame to 256×320
with INTER_AREA, JPEG quality 90; `fastwam/main.py` decode and BGR→RGB), the raw state and the task string. Reported
per request and summarized:
  - action L1 (rad) against the recorded actions t..t+15, arm joints and grippers separately, for the camera path and
    for the raw 480×640 frames (the training preprocessing input);
  - action L1 between the camera path and the raw 480×640 frames with the same seed (camera-path effect);
  - response shape and dtype, client round trip and server timing.

  PYTHONPATH=deploy .venv/bin/python deploy/g1_server/replay_check.py --server-host 127.0.0.1 --server-port 8000
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from g1_server.heldout_samples import raw_request, validation_dataset
from g1_server.model_registry import MODEL_SPECS, PROJECT_ROOT

G1_CLIENT_ROOT = PROJECT_ROOT / "tmp" / "g1-client"
sys.path.insert(0, str(G1_CLIENT_ROOT))
sys.path.insert(0, str(G1_CLIENT_ROOT / "fastwam"))
from fastwam_policy import FastWAMPolicy  # noqa: E402

CAMERA_HEIGHT, CAMERA_WIDTH, JPEG_QUALITY = 256, 320, 90  # g1_client/camera_client.py


def robot_camera_view(rgb: np.ndarray) -> np.ndarray:
    """480×640 RGB → what `fastwam/main.py` sends: BGR resize to 256×320 (INTER_AREA), JPEG q90, decode, RGB."""
    bgr = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (CAMERA_WIDTH, CAMERA_HEIGHT), interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return cv2.cvtColor(cv2.imdecode(buffer, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--stride", type=int, default=64, help="frames between replayed samples")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    policy = FastWAMPolicy(args.server_host, args.server_port)
    metadata = policy.get_server_metadata()
    reset = policy._client.infer({"reset": True, "prompt": ""})
    _, dataset = validation_dataset(MODEL_SPECS[0].run_dir)
    rows = []
    for index in range(0, len(dataset) - 17, args.stride):
        request, recorded = raw_request(dataset, index)
        camera = [robot_camera_view(view) for view in request.views]
        sent = {"image": camera, "state": request.state, "prompt": request.prompt}
        response = policy._client.infer(sent)
        timing = dict(policy.last_timing)
        actions = np.asarray(response["actions"])
        full = np.asarray(policy.infer({"image": list(request.views), "state": request.state,
                                        "prompt": request.prompt})["actions"])
        rows.append({"index": index, "task": request.prompt[:40], "shape": list(actions.shape),
                     "dtype": str(actions.dtype), "model": response["model"], "step": response["checkpoint_step"],
                     "arm_l1": float(np.abs(actions[:, :14] - recorded[:, :14]).mean()),
                     "gripper_l1": float(np.abs(actions[:, 14:] - recorded[:, 14:]).mean()),
                     "camera_path_l1": float(np.abs(actions - full).mean()),
                     "raw_arm_l1": float(np.abs(full[:, :14] - recorded[:, :14]).mean()),
                     "raw_gripper_l1": float(np.abs(full[:, 14:] - recorded[:, 14:]).mean()),
                     "round_trip_ms": timing["total_s"] * 1e3, "server_total_ms": response["server_timing"]["total_ms"],
                     "server_model_ms": response["server_timing"]["model_ms"], "bytes_sent": timing["bytes_sent"]})
        print(json.dumps(rows[-1]), flush=True)
    by_task = {}
    for row in rows:
        by_task.setdefault(row["task"], []).append(row)
    summary = {"metadata": metadata, "reset_response": reset, "requests": len(rows),
               "shapes": sorted({str(r["shape"]) for r in rows}), "dtypes": sorted({r["dtype"] for r in rows}),
               "arm_l1_rad": float(np.mean([r["arm_l1"] for r in rows])),
               "gripper_l1": float(np.mean([r["gripper_l1"] for r in rows])),
               "camera_path_l1": float(np.mean([r["camera_path_l1"] for r in rows])),
               "raw_arm_l1_rad": float(np.mean([r["raw_arm_l1"] for r in rows])),
               "raw_arm_l1_se": float(np.std([r["raw_arm_l1"] for r in rows], ddof=1) / np.sqrt(len(rows))),
               "raw_gripper_l1": float(np.mean([r["raw_gripper_l1"] for r in rows])),
               "round_trip_ms_median": float(np.median([r["round_trip_ms"] for r in rows[1:]])),
               "server_model_ms_median": float(np.median([r["server_model_ms"] for r in rows[1:]])),
               "per_task_arm_l1": {task: float(np.mean([r["arm_l1"] for r in group])) for task, group in by_task.items()},
               "per_task_raw_arm_l1": {task: float(np.mean([r["raw_arm_l1"] for r in group]))
                                       for task, group in by_task.items()}}
    print(json.dumps(summary, indent=1))
    if args.out:
        args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    policy.close()


if __name__ == "__main__":
    main()
