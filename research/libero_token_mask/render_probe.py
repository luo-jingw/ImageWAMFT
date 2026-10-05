"""Renders a fixed LIBERO state with the active MUJOCO_GL backend; saves the first observation and times steps."""
import os
import sys
import time

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(3)
env = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
                         camera_heights=256, camera_widths=256)
env.seed(42)
env.reset()
obs = env.set_init_state(suite.get_task_init_states(3)[0])
for _ in range(30):
    obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
np.savez(sys.argv[1], agent=obs["agentview_image"], wrist=obs["robot0_eye_in_hand_image"])
start = time.time()
for _ in range(50):
    env.step([0, 0, 0, 0, 0, 0, -1])
print(f"{os.environ['MUJOCO_GL']}: {(time.time() - start) / 50 * 1000:.1f} ms/step")
