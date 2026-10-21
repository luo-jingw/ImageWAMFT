# RoboTwin 2.0 Evaluation Results

RoboTwin 2.0 simulation success of the models trained in `docs/robotwin_training_results.md`, against the released FLUX.2-4B and paper Table 12. The evaluation pipeline and its configuration check are in `docs/robotwin_training_setup.md` → Evaluation Protocol.

## Protocol

- 50 tasks, each in clean (`demo_clean`) and randomized (`demo_randomized`) scenes, 100 episodes per task and setting.
- Seed 42: the expert-checked seeds from 4,300,000. Unseen instructions.
- 10 denoising steps, action horizon 16, replan every 16 actions, `compact_288x256` layout, text CFG scale 1.0.
- A failed episode runs to its task's step limit (400–1,700 actions, `third_party/RoboTwin/task_config/_eval_step_limit.yml`).
- The evaluation process never seeds Python `random`, which draws the instructions and is used by cuRobo's planner. Two runs of the same protocol therefore keep different seeds and instructions per task. Models are compared by the difference of their means.
- Mean: the mean of the 50 per-task success rates. Standard error: sqrt(Σ p(1 − p) / 100) / 50 over the tasks (`launch/robotwin_eval_compare.py`).
- Reference: paper Table 12, FLUX.2-4B (`launch/robotwin_table12_flux2_4b.csv`). A per-task difference is counted when its pooled two-proportion z-score exceeds 3 in absolute value.

## Models

| | Released FLUX.2-4B | FLUX.2-9B | Qwen-Image-2.1 |
|---|---|---|---|
| Weights | `checkpoints/imagewam_release/robotwin/flux2_klein_4b/model.pt` | `checkpoints/robotwin_flux2_klein_9b_base_b384_imagewam/checkpoints/weights/step_065000.pt` | `checkpoints/robotwin_qwenimage21_imagewam/checkpoints/weights/step_065000.pt` |
| Training | 69,900 steps, most likely without gradient clipping (`docs/robotwin_training_setup.md` → Training Recipe of the Released Model) | 65,000 of 69,900 scheduled steps, `gradient_clipping` 1.0 | 65,000 of 69,900 scheduled steps, `gradient_clipping` 1.0 |
| Text features | Qwen3-4B, encoded in each worker, 512 slots | Qwen3-8B, encoded in each worker, 512 slots | Qwen3-VL, from `qwen_text_server.py` on the first GPU, 128 slots |
| Results | `ImageWAM/evaluate_results/robotwin/model/flux2_4b_release_full/` | `ImageWAM/evaluate_results/robotwin/step_065000/flux2_9b_step065000_full/` | `ImageWAM/evaluate_results/robotwin/step_065000/qwenimage21_step065000_full/` |

The released FLUX.2-4B is the model of paper Table 12, so its run checks this pipeline against the paper. FLUX.2-9B and Qwen-Image-2.1 share this project's training recipe (`docs/robotwin_training_results.md`).

## Runs

All runs: `nnmc71` GPUs 0–3 (4 × H100 NVL, 93.6 GiB each). GPU memory is sampled every minute.

| | Released FLUX.2-4B | FLUX.2-9B | Qwen-Image-2.1 |
|---|---|---|---|
| Workers per GPU | 3: 12 workers | 2: 8 workers | 2 on GPU 0 (with the text server), 3 on GPUs 1–3: 11 workers |
| Wall clock | 11.4 h, 2026-10-20 04:59 – 16:23 EDT | 13.2 h, 2026-10-20 16:38 – 10-21 05:47 EDT | 12.1 h, 2026-10-19 16:49 – 10-20 04:57 EDT |
| Worker time | 116.3 worker-hours, 41.9 worker-seconds per episode | 95.8 worker-hours, 34.5 worker-seconds per episode | 112.8 worker-hours, 40.6 worker-seconds per episode |
| Shortest and longest (task, setting) | `click_bell` clean 19 min; `put_bottles_dustbin` clean 171 min | `click_bell` clean 15 min; `put_object_cabinet` randomized 205 min | `press_stapler` clean 18 min; `open_microwave` clean 206 min |
| Peak GPU memory | 76.7–78.3 GiB per GPU | 89.0–90.0 GiB per GPU | 68.2 GiB on GPU 0, 72.1–74.1 GiB on GPUs 1–3 |
| Mean GPU utilization | 63–71% | 48–56% | 53–67% |
| Failed or resumed workers | none | none | none |

- FLUX.2-9B fits 2 workers per GPU: each worker holds the 9B model and the Qwen3-8B text encoder. Its 2-worker memory check on 8 randomized tasks peaked at 86.9 GiB.
- Summed over the 100 (task, setting) runs, a FLUX.2-9B worker takes 0.85 of the Qwen-Image-2.1 time and 0.82 of the FLUX.2-4B time, with fewer workers per GPU.

## Success Rates

| Model | Clean | Randomized |
|---|---|---|
| FLUX.2-4B, paper Table 12 | 93.20 | 93.56 |
| Released FLUX.2-4B, this pipeline | 93.46 ± 0.33 | 93.36 ± 0.33 |
| FLUX.2-9B, step 65,000 | 94.58 ± 0.30 | 94.34 ± 0.31 |
| Qwen-Image-2.1, step 65,000 | 92.82 ± 0.33 | 94.00 ± 0.32 |

**Pipeline check:** the released FLUX.2-4B reproduces Table 12. Its means are within one standard error of the paper, and no task differs from Table 12 by |z| > 3 in either setting (clean: 17 tasks below, 20 above; randomized: 19 below, 12 above).

Differences in points, as a difference of means and paired over the seeds both runs kept (4,687–4,721 of 5,000 per pair and setting):

| Pair | Clean, means | Clean, paired | Randomized, means | Randomized, paired |
|---|---|---|---|---|
| FLUX.2-9B − released FLUX.2-4B | +1.12 ± 0.45 | +1.44 ± 0.40 | +0.98 ± 0.45 | +1.22 ± 0.39 |
| Qwen-Image-2.1 − released FLUX.2-4B | −0.64 ± 0.47 | −0.50 ± 0.42 | +0.64 ± 0.46 | +0.80 ± 0.41 |
| Qwen-Image-2.1 − FLUX.2-9B | −1.76 ± 0.45 | −1.74 ± 0.41 | −0.34 ± 0.45 | −0.23 ± 0.39 |

- FLUX.2-9B is above the released FLUX.2-4B in both settings: by 2.5 and 2.2 standard errors of the difference of means, and by 3.6 and 3.1 standard errors paired. It trained 4,900 fewer steps and with gradient clipping 1.0.
- Qwen-Image-2.1 and the released FLUX.2-4B do not differ beyond 2 standard errors in either setting.
- Qwen-Image-2.1 is below FLUX.2-9B in clean scenes. `open_microwave` alone contributes 1.02 of the 1.76 points (98 against 47). In randomized scenes the two do not differ.

Tasks with |z| > 3 against Table 12 in any model:

| Task | Setting | Table 12 | Released FLUX.2-4B | FLUX.2-9B | Qwen-Image-2.1 |
|---|---|---|---|---|---|
| `open_microwave` | clean | 97 | 93 | 98 | **47** (z −7.9) |
| `stamp_seal` | clean | 79 | 81 | 87 | **98** (z +4.2) |
| `stamp_seal` | randomized | 84 | 80 | 92 | **97** (z +3.1) |
| `move_stapler_pad` | clean | 67 | 71 | **86** (z +3.2) | 82 |

`open_microwave` draws one of two microwave models (`envs/open_microwave.py`, `model_id`). The unseen instructions name the model (`description/objects_description/044_microwave/base{0,1}.json`). Success per model:

| | base0 (gray and yellow, top vents) | base1 (dark gray, slanted walls) |
|---|---|---|
| Released FLUX.2-4B, clean | 46 / 49 | 47 / 51 |
| Released FLUX.2-4B, randomized | 46 / 46 | 46 / 54 |
| FLUX.2-9B, clean | 51 / 53 | 47 / 47 |
| FLUX.2-9B, randomized | 49 / 50 | 40 / 50 |
| Qwen-Image-2.1, clean | 41 / 48 | 6 / 52 |
| Qwen-Image-2.1, randomized | 46 / 47 | 43 / 53 |

All 53 failed clean Qwen-Image-2.1 episodes end at the 1,500-action limit. The cause is tracked in ISSUE-014.

Per task, success in percent (4B: released FLUX.2-4B; 9B: FLUX.2-9B; Qwen: Qwen-Image-2.1):

| Task | Clean: Table 12 | 4B | 9B | Qwen | Randomized: Table 12 | 4B | 9B | Qwen |
|---|---|---|---|---|---|---|---|---|
| `adjust_bottle` | 100 | 100 | 100 | 100 | 99 | 100 | 99 | 100 |
| `beat_block_hammer` | 98 | 100 | 98 | 99 | 99 | 97 | 99 | 98 |
| `blocks_ranking_rgb` | 96 | 96 | 100 | 99 | 99 | 100 | 100 | 97 |
| `blocks_ranking_size` | 96 | 95 | 93 | 93 | 100 | 98 | 96 | 94 |
| `click_alarmclock` | 98 | 99 | 99 | 97 | 100 | 100 | 100 | 100 |
| `click_bell` | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| `dump_bin_bigbin` | 96 | 97 | 99 | 98 | 90 | 90 | 97 | 97 |
| `grab_roller` | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| `handover_block` | 96 | 92 | 96 | 95 | 95 | 97 | 92 | 94 |
| `handover_mic` | 100 | 98 | 100 | 97 | 100 | 100 | 100 | 100 |
| `hanging_mug` | 74 | 79 | 66 | 66 | 84 | 80 | 66 | 83 |
| `lift_pot` | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| `move_can_pot` | 96 | 95 | 96 | 98 | 98 | 97 | 98 | 95 |
| `move_pillbottle_pad` | 98 | 96 | 98 | 99 | 100 | 99 | 100 | 100 |
| `move_playingcard_away` | 100 | 100 | 100 | 100 | 99 | 99 | 100 | 100 |
| `move_stapler_pad` | 67 | 71 | 86 | 82 | 60 | 61 | 69 | 78 |
| `open_laptop` | 98 | 99 | 97 | 97 | 98 | 98 | 96 | 99 |
| `open_microwave` | 97 | 93 | 98 | 47 | 94 | 92 | 89 | 89 |
| `pick_diverse_bottles` | 84 | 88 | 91 | 92 | 88 | 88 | 86 | 88 |
| `pick_dual_bottles` | 96 | 99 | 99 | 99 | 98 | 95 | 96 | 98 |
| `place_a2b_left` | 95 | 96 | 98 | 98 | 93 | 96 | 98 | 98 |
| `place_a2b_right` | 96 | 95 | 98 | 98 | 94 | 98 | 97 | 94 |
| `place_bread_basket` | 96 | 98 | 98 | 96 | 92 | 97 | 94 | 96 |
| `place_bread_skillet` | 90 | 88 | 93 | 96 | 89 | 88 | 97 | 92 |
| `place_burger_fries` | 95 | 96 | 98 | 99 | 100 | 100 | 100 | 98 |
| `place_can_basket` | 74 | 72 | 76 | 63 | 72 | 67 | 82 | 71 |
| `place_cans_plasticbox` | 99 | 97 | 98 | 97 | 97 | 97 | 99 | 96 |
| `place_container_plate` | 98 | 99 | 97 | 98 | 98 | 96 | 98 | 100 |
| `place_dual_shoes` | 81 | 80 | 92 | 86 | 81 | 86 | 92 | 87 |
| `place_empty_cup` | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| `place_fan` | 95 | 96 | 98 | 97 | 94 | 88 | 97 | 92 |
| `place_mouse_pad` | 84 | 84 | 92 | 94 | 93 | 91 | 95 | 95 |
| `place_object_basket` | 86 | 86 | 85 | 86 | 83 | 86 | 85 | 77 |
| `place_object_scale` | 97 | 96 | 100 | 94 | 96 | 96 | 99 | 95 |
| `place_object_stand` | 98 | 99 | 99 | 97 | 98 | 96 | 97 | 95 |
| `place_phone_stand` | 100 | 100 | 100 | 100 | 100 | 100 | 99 | 99 |
| `place_shoe` | 97 | 97 | 95 | 94 | 95 | 95 | 97 | 99 |
| `press_stapler` | 97 | 96 | 96 | 98 | 100 | 100 | 98 | 100 |
| `put_bottles_dustbin` | 97 | 96 | 98 | 96 | 91 | 88 | 87 | 92 |
| `put_object_cabinet` | 91 | 94 | 91 | 91 | 89 | 95 | 90 | 84 |
| `rotate_qrcode` | 87 | 94 | 93 | 89 | 92 | 91 | 91 | 84 |
| `scan_object` | 94 | 92 | 92 | 91 | 90 | 98 | 84 | 93 |
| `shake_bottle` | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| `shake_bottle_horizontally` | 100 | 100 | 100 | 100 | 100 | 99 | 100 | 100 |
| `stack_blocks_three` | 96 | 97 | 99 | 99 | 97 | 97 | 99 | 99 |
| `stack_blocks_two` | 99 | 100 | 100 | 100 | 100 | 99 | 100 | 99 |
| `stack_bowls_three` | 78 | 80 | 80 | 72 | 83 | 75 | 85 | 83 |
| `stack_bowls_two` | 94 | 92 | 90 | 92 | 97 | 97 | 96 | 96 |
| `stamp_seal` | 79 | 81 | 87 | 98 | 84 | 80 | 92 | 97 |
| `turn_switch` | 77 | 75 | 70 | 64 | 79 | 81 | 86 | 79 |
| **mean** | 93.20 | 93.46 ± 0.33 | 94.58 ± 0.30 | 92.82 ± 0.33 | 93.56 | 93.36 ± 0.33 | 94.34 ± 0.31 | 94.00 ± 0.32 |
