"""Splits the 50 RoboTwin evaluation tasks into one TASKS list per node (plan.md E9).

Each task is weighted by its step limit (`third_party/RoboTwin/task_config/_eval_step_limit.yml`), the length of a
failed episode. Tasks are taken in decreasing weight, ties in file order, and each goes to the node whose load per
GPU after adding it is lowest, so the split is deterministic for the same node list.

  .venv/bin/python launch/robotwin_eval_shards.py nnmc62=4 nnmc63=2 nnmc64=2 nnmc65=2 nnmc70=2

Prints one line per node: name, GPUs, step-limit sum per GPU, and the comma-separated TASKS list for
launch/robotwin_eval.sh.
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import yaml

STEP_LIMIT_FILE = (
    Path(__file__).resolve().parents[1] / "ImageWAM" / "third_party" / "RoboTwin" / "task_config" / "_eval_step_limit.yml"
)


@dataclass
class NodeShard:
    name: str
    gpus: int
    tasks: list[str] = field(default_factory=list)
    load: int = 0

    def load_per_gpu(self, extra: int = 0) -> float:
        return (self.load + extra) / self.gpus


def parse_node(text: str) -> NodeShard:
    name, gpus = text.split("=", 1)
    if int(gpus) <= 0:
        raise ValueError(f"{name}: GPU count must be positive")
    return NodeShard(name=name, gpus=int(gpus))


def split_tasks(step_limits: dict[str, int], nodes: list[NodeShard]) -> list[NodeShard]:
    order = {task: i for i, task in enumerate(step_limits)}
    for task in sorted(step_limits, key=lambda t: (-step_limits[t], order[t])):
        node = min(nodes, key=lambda n: (n.load_per_gpu(step_limits[task]), nodes.index(n)))
        node.tasks.append(task)
        node.load += step_limits[task]
    for node in nodes:
        node.tasks.sort(key=lambda t: order[t])
    return nodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("nodes", nargs="+", help="<node>=<number of GPUs>")
    args = parser.parse_args()
    step_limits = {str(k): int(v) for k, v in yaml.safe_load(STEP_LIMIT_FILE.read_text()).items()}
    nodes = split_tasks(step_limits, [parse_node(text) for text in args.nodes])
    assigned = [task for node in nodes for task in node.tasks]
    if sorted(assigned) != sorted(step_limits):
        raise RuntimeError("tasks are not assigned exactly once")
    for node in nodes:
        print(f"{node.name} gpus={node.gpus} tasks={len(node.tasks)} step_limit_per_gpu={node.load_per_gpu():.0f}")
        print(f"  TASKS={','.join(node.tasks)}")
    print(f"{len(assigned)} tasks, step-limit sum {sum(step_limits.values())}")


if __name__ == "__main__":
    main()
