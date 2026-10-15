"""Merge the four LGG100 datasets into one LeRobot v2.1 dataset for the Hugging Face Hub (run in `.venv`).

- Episodes listed in `ImageWAM/configs/data/g1_lgg100.yaml` (`train.episode_index_filter.exclude`;
  docs/lgg100_dataset.md -> Invalid Episodes) are dropped.
- Task strings are the project ones in `dataset/LGG100/<name>/meta/tasks.jsonl`; `task_index` follows SOURCES.
- Episodes are renumbered 0..N-1 in SOURCES order, keeping the source order. Parquet rows keep their values except
  `episode_index`, `index` and `task_index`; videos are copied byte for byte.
- `meta/episodes_stats.jsonl` keeps the source statistics and recomputes those of `episode_index`, `index` and
  `task_index`.
- `meta/episode_sources.jsonl` maps every episode to its source repository and episode index.

  .venv/bin/python launch/lgg100_release_dataset.py --output-dir dataset/hf_release/G1-Dex1-4tasks
"""

import argparse
import json
import os
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCES = ["drawer", "biomanual", "simple-pick", "Stack-the-cubes"]
REINDEXED = ["episode_index", "index", "task_index"]


def read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def int_stats(values: np.ndarray) -> dict:
    return {
        "min": [int(values.min())],
        "max": [int(values.max())],
        "mean": [float(values.mean())],
        "std": [float(values.std())],
        "count": [int(values.size)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", default=os.path.join(PROJECT_ROOT, "dataset", "LGG100"))
    parser.add_argument("--config", default=os.path.join(PROJECT_ROOT, "ImageWAM", "configs", "data", "g1_lgg100.yaml"))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        exclude = yaml.safe_load(f)["train"]["episode_index_filter"]["exclude"]
    if os.path.exists(args.output_dir):
        raise FileExistsError(args.output_dir)

    infos = {}
    for name in SOURCES:
        with open(os.path.join(args.source_root, name, "meta", "info.json")) as f:
            infos[name] = json.load(f)
    info = json.loads(json.dumps(infos[SOURCES[0]]))
    for name in SOURCES[1:]:
        for key in ["codebase_version", "robot_type", "fps", "chunks_size", "data_path", "video_path", "features"]:
            if infos[name][key] != info[key]:
                raise ValueError(f"{name}: info.json {key} differs from {SOURCES[0]}")
    video_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]

    tasks, episodes, stats, sources, summary = [], [], [], [], []
    new_episode, global_index = 0, 0
    for task_index, name in enumerate(SOURCES):
        src = os.path.join(args.source_root, name)
        src_tasks = read_jsonl(os.path.join(src, "meta", "tasks.jsonl"))
        if len(src_tasks) != 1:
            raise ValueError(f"{name}: expected one task, found {len(src_tasks)}")
        task = src_tasks[0]["task"]
        tasks.append({"task_index": task_index, "task": task})
        src_episodes = {e["episode_index"]: e for e in read_jsonl(os.path.join(src, "meta", "episodes.jsonl"))}
        src_stats = {e["episode_index"]: e["stats"] for e in read_jsonl(os.path.join(src, "meta", "episodes_stats.jsonl"))}
        dropped = sorted(exclude.get(name, []))
        kept_episodes, kept_frames = 0, 0
        for old in sorted(src_episodes):
            if old in dropped:
                continue
            src_chunk = old // info["chunks_size"]
            new_chunk = new_episode // info["chunks_size"]
            src_parquet = os.path.join(src, info["data_path"].format(episode_chunk=src_chunk, episode_index=old))
            table = pq.read_table(src_parquet)
            n = table.num_rows
            if n != src_episodes[old]["length"]:
                raise ValueError(f"{name} episode {old}: {n} rows, episodes.jsonl length {src_episodes[old]['length']}")
            columns = {
                "episode_index": np.full(n, new_episode, dtype=np.int64),
                "index": np.arange(global_index, global_index + n, dtype=np.int64),
                "task_index": np.full(n, task_index, dtype=np.int64),
            }
            for key, values in columns.items():
                table = table.set_column(table.schema.get_field_index(key), key, pa.array(values))
            table = table.replace_schema_metadata(pq.read_schema(src_parquet).metadata)
            out_parquet = os.path.join(args.output_dir, info["data_path"].format(episode_chunk=new_chunk, episode_index=new_episode))
            os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
            pq.write_table(table, out_parquet)
            for key in video_keys:
                out_video = os.path.join(
                    args.output_dir, info["video_path"].format(episode_chunk=new_chunk, video_key=key, episode_index=new_episode)
                )
                os.makedirs(os.path.dirname(out_video), exist_ok=True)
                shutil.copyfile(
                    os.path.join(src, info["video_path"].format(episode_chunk=src_chunk, video_key=key, episode_index=old)),
                    out_video,
                )
            episode_stats = src_stats[old]
            for key in REINDEXED:
                episode_stats[key] = int_stats(columns[key])
            episodes.append({"episode_index": new_episode, "tasks": [task], "length": n})
            stats.append({"episode_index": new_episode, "stats": episode_stats})
            sources.append({"episode_index": new_episode, "source_repo": f"LGG100/{name}", "source_episode_index": old})
            new_episode += 1
            global_index += n
            kept_episodes += 1
            kept_frames += n
        summary.append({"source": name, "task_index": task_index, "dropped": dropped, "episodes": kept_episodes, "frames": kept_frames})

    info.update(
        total_episodes=new_episode,
        total_frames=global_index,
        total_tasks=len(tasks),
        total_videos=new_episode * len(video_keys),
        total_chunks=(new_episode - 1) // info["chunks_size"] + 1,
        splits={"train": f"0:{new_episode}"},
    )
    meta = os.path.join(args.output_dir, "meta")
    os.makedirs(meta)
    with open(os.path.join(meta, "info.json"), "w") as f:
        json.dump(info, f, indent=4)
    write_jsonl(os.path.join(meta, "tasks.jsonl"), tasks)
    write_jsonl(os.path.join(meta, "episodes.jsonl"), episodes)
    write_jsonl(os.path.join(meta, "episodes_stats.jsonl"), stats)
    write_jsonl(os.path.join(meta, "episode_sources.jsonl"), sources)
    print(json.dumps({"episodes": new_episode, "frames": global_index, "sources": summary}, indent=1))


if __name__ == "__main__":
    main()
