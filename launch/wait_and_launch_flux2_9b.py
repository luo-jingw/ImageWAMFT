#!/usr/bin/env python3
"""Start the FLUX.2-9B G1 run (launch/g1_flux2_9b_train.sh) once enough GPUs stay idle, then verify its first step.

Runs on nnmc71 in tmux and polls the nodes over ssh every POLL_SECONDS. A GPU counts as idle when it has no compute
process, uses < 1000 MiB and reports < 5 % utilization; a placement starts only after the same GPUs were idle in
STREAK consecutive polls. At most 4 GPUs per node are used by this project, counting running ImageWAMFT processes.
Placements in order of preference:
  2 x B300 on nnmc73  ->  4 x H100 on nnmc71 or nnmc72  ->  1 x B300 on nnmc73
After launching it waits for "step=1/" in the run log. On an error it stops the tmux session it started and exits
non-zero instead of retrying.
"""

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

POLL_SECONDS = 300
STREAK = 3
FIRST_STEP_TIMEOUT_SECONDS = 3600
SESSION = "imagewamft_flux9b"
RUN_LOG = Path("/home/user1/workspace/jingwu/ImageWAMFT/checkpoints/g1_lgg100_flux2_klein_9b_base_imagewam/train.log")
LAUNCHERS = {
    "nnmc73": "/shared/user71/workspace/jingwu/ImageWAMFT/launch/g1_flux2_9b_train.sh",
    "nnmc71": "/home/user1/workspace/jingwu/ImageWAMFT/launch/g1_flux2_9b_train.sh",
    "nnmc72": "/home/user1/workspace/jingwu/ImageWAMFT/launch/g1_flux2_9b_train.sh",
}
NVLINK_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]


@dataclass
class NodeState:
    idle: list[int]
    ours: int


def log(message: str) -> None:
    print(f"{time.strftime('%F %T')} {message}", flush=True)


def ssh(node: str, command: str, timeout: int = 60) -> str:
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", node, command],
                            capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"{node}: {result.stderr.strip()[:200]}")
    return result.stdout


def node_state(node: str) -> NodeState:
    script = (
        "nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits; echo ---; "
        "for p in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | tr -d ' '); do "
        "u=${p%,*}; pid=${p#*,}; echo \"$u $(ps -o args= -p $pid 2>/dev/null | grep -c ImageWAMFT)\"; done"
    )
    gpus_part, apps_part = ssh(node, script).split("---")
    busy_uuid, ours_uuid = set(), set()
    for line in apps_part.strip().splitlines():
        uuid, is_ours = line.split()
        busy_uuid.add(uuid)
        if is_ours != "0":
            ours_uuid.add(uuid)
    idle = []
    for line in gpus_part.strip().splitlines():
        index, uuid, memory, util = [field.strip() for field in line.split(",")]
        util_value = float(util) if util.replace(".", "").isdigit() else 100.0
        if uuid not in busy_uuid and float(memory) < 1000 and util_value < 5:
            idle.append(int(index))
    return NodeState(idle=idle, ours=len(ours_uuid))


def pick_h100(idle: list[int]) -> list[int] | None:
    pairs = [pair for pair in NVLINK_PAIRS if pair[0] in idle and pair[1] in idle]
    if len(pairs) >= 2:
        return [*pairs[0], *pairs[1]]
    return sorted(idle)[:4] if len(idle) >= 4 else None


def candidates(states: dict[str, NodeState]) -> list[tuple[str, list[int]]]:
    out = []
    b300 = states.get("nnmc73")
    if b300 is not None and 4 - b300.ours >= 2 and len(b300.idle) >= 2:
        out.append(("nnmc73", sorted(b300.idle)[:2]))
    for node in ("nnmc71", "nnmc72"):
        state = states.get(node)
        if state is not None and 4 - state.ours >= 4:
            choice = pick_h100(state.idle)
            if choice:
                out.append((node, choice))
    if b300 is not None and 4 - b300.ours >= 1 and len(b300.idle) >= 1:
        out.append(("nnmc73", sorted(b300.idle)[:1]))
    return out


def launch(node: str, gpus: list[int]) -> None:
    gpu_list = ",".join(str(g) for g in gpus)
    command = (f"tmux new-session -d -s {SESSION} "
               f"\"GPUS={gpu_list} bash {LAUNCHERS[node]}; echo '[launch] exited with status '\\$?; exec bash\"")
    ssh(node, command)
    log(f"launched {SESSION} on {node} GPUs {gpu_list}")


def wait_first_step(node: str, start_offset: int) -> bool:
    deadline = time.time() + FIRST_STEP_TIMEOUT_SECONDS
    while time.time() < deadline:
        time.sleep(60)
        text = RUN_LOG.read_text(errors="replace")[start_offset:] if RUN_LOG.exists() else ""
        flat = " ".join(text.split())
        if "step=1/" in flat or "step=10/" in flat:
            log("first optimizer step logged; run is training")
            return True
        for marker in ("Traceback", "OutOfMemoryError", "exited with status"):
            if marker in flat:
                index = flat.find(marker)
                log(f"launch failed ({marker}): {flat[max(0, index - 300): index + 300]}")
                return False
    log("no first step within the timeout")
    return False


def main() -> None:
    streaks: dict[tuple[str, tuple[int, ...]], int] = {}
    while True:
        states = {}
        for node in ("nnmc73", "nnmc71", "nnmc72"):
            try:
                states[node] = node_state(node)
            except Exception as error:  # node unreachable this round
                log(f"poll {node} failed: {error}")
        current = [(node, tuple(gpus)) for node, gpus in candidates(states)]
        streaks = {key: streaks.get(key, 0) + 1 for key in current}
        summary = {node: (state.idle, state.ours) for node, state in states.items()}
        log(f"idle GPUs (idle, ours) {summary}; candidate streaks {streaks}")
        ready = [key for key in current if streaks[key] >= STREAK]
        if ready:
            node, gpus = ready[0]
            offset = RUN_LOG.stat().st_size if RUN_LOG.exists() else 0
            launch(node, list(gpus))
            if wait_first_step(node, offset):
                return
            ssh(node, f"tmux kill-session -t {SESSION}")
            log(f"stopped {SESSION} on {node}; not retrying")
            sys.exit(1)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
