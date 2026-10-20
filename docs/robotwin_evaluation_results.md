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

| | Qwen-Image-2.1 |
|---|---|
| Weights | `checkpoints/robotwin_qwenimage21_imagewam/checkpoints/weights/step_065000.pt` |
| Training | 65,000 of 69,900 scheduled steps, `gradient_clipping` 1.0 |
| Text features | Qwen3-VL, from `qwen_text_server.py` on the first GPU, 128 slots |
| Results | `ImageWAM/evaluate_results/robotwin/step_065000/qwenimage21_step065000_full/` |

The released FLUX.2-4B of Table 12 ran all 69,900 steps, most likely without gradient clipping (`docs/robotwin_training_setup.md` → Training Recipe of the Released Model).

## Runs

| | Qwen-Image-2.1 |
|---|---|
| Host | `nnmc71` GPUs 0–3 (4 × H100 NVL) |
| Workers per GPU | 2 on GPU 0 (with the text server), 3 on GPUs 1–3: 11 workers |
| Wall clock | 12.1 h, 2026-10-19 16:49 – 10-20 04:57 EDT |
| Worker time | 112.8 worker-hours, 40.6 worker-seconds per episode |
| Longest (task, setting) | `open_microwave` clean, 206 min; shortest `press_stapler` clean, 18 min |
| Peak GPU memory (sampled every minute) | 68.2 GB on GPU 0, 72.1–74.1 GB on GPUs 1–3 |
| Mean GPU utilization | 53–67% |
| Failed or resumed workers | none |

## Success Rates

| Model | Clean | Randomized |
|---|---|---|
| FLUX.2-4B, paper Table 12 | 93.20 | 93.56 |
| Qwen-Image-2.1, step 65,000 | 92.82 ± 0.33 | 94.00 ± 0.32 |

Against Table 12, Qwen-Image-2.1 is below on 17 tasks and above on 20 in clean scenes, and below on 19 and above on 16 in randomized scenes. Tasks with |z| > 3:

| Task | Setting | Table 12 | Qwen-Image-2.1 | z |
|---|---|---|---|---|
| `open_microwave` | clean | 97 | 47 | −7.9 |
| `stamp_seal` | clean | 79 | 98 | +4.2 |
| `stamp_seal` | randomized | 84 | 97 | +3.1 |

`open_microwave` draws one of two microwave models (`envs/open_microwave.py`, `model_id`). The unseen instructions name the model (`description/objects_description/044_microwave/base{0,1}.json`). Qwen-Image-2.1 success per model:

| | base0 (gray and yellow, top vents) | base1 (dark gray, slanted walls) |
|---|---|---|
| clean | 41 / 48 | 6 / 52 |
| randomized | 46 / 47 | 43 / 53 |

All 53 failed clean episodes end at the 1,500-action limit; successful ones take 153–438 actions. The open question is tracked in ISSUE-014.

Per task, success in percent:

| Task | Table 12 clean | Qwen-Image-2.1 clean | Table 12 randomized | Qwen-Image-2.1 randomized |
|---|---|---|---|---|
| `adjust_bottle` | 100 | 100 | 99 | 100 |
| `beat_block_hammer` | 98 | 99 | 99 | 98 |
| `blocks_ranking_rgb` | 96 | 99 | 99 | 97 |
| `blocks_ranking_size` | 96 | 93 | 100 | 94 |
| `click_alarmclock` | 98 | 97 | 100 | 100 |
| `click_bell` | 100 | 100 | 100 | 100 |
| `dump_bin_bigbin` | 96 | 98 | 90 | 97 |
| `grab_roller` | 100 | 100 | 100 | 100 |
| `handover_block` | 96 | 95 | 95 | 94 |
| `handover_mic` | 100 | 97 | 100 | 100 |
| `hanging_mug` | 74 | 66 | 84 | 83 |
| `lift_pot` | 100 | 100 | 100 | 100 |
| `move_can_pot` | 96 | 98 | 98 | 95 |
| `move_pillbottle_pad` | 98 | 99 | 100 | 100 |
| `move_playingcard_away` | 100 | 100 | 99 | 100 |
| `move_stapler_pad` | 67 | 82 | 60 | 78 |
| `open_laptop` | 98 | 97 | 98 | 99 |
| `open_microwave` | 97 | 47 | 94 | 89 |
| `pick_diverse_bottles` | 84 | 92 | 88 | 88 |
| `pick_dual_bottles` | 96 | 99 | 98 | 98 |
| `place_a2b_left` | 95 | 98 | 93 | 98 |
| `place_a2b_right` | 96 | 98 | 94 | 94 |
| `place_bread_basket` | 96 | 96 | 92 | 96 |
| `place_bread_skillet` | 90 | 96 | 89 | 92 |
| `place_burger_fries` | 95 | 99 | 100 | 98 |
| `place_can_basket` | 74 | 63 | 72 | 71 |
| `place_cans_plasticbox` | 99 | 97 | 97 | 96 |
| `place_container_plate` | 98 | 98 | 98 | 100 |
| `place_dual_shoes` | 81 | 86 | 81 | 87 |
| `place_empty_cup` | 100 | 100 | 100 | 100 |
| `place_fan` | 95 | 97 | 94 | 92 |
| `place_mouse_pad` | 84 | 94 | 93 | 95 |
| `place_object_basket` | 86 | 86 | 83 | 77 |
| `place_object_scale` | 97 | 94 | 96 | 95 |
| `place_object_stand` | 98 | 97 | 98 | 95 |
| `place_phone_stand` | 100 | 100 | 100 | 99 |
| `place_shoe` | 97 | 94 | 95 | 99 |
| `press_stapler` | 97 | 98 | 100 | 100 |
| `put_bottles_dustbin` | 97 | 96 | 91 | 92 |
| `put_object_cabinet` | 91 | 91 | 89 | 84 |
| `rotate_qrcode` | 87 | 89 | 92 | 84 |
| `scan_object` | 94 | 91 | 90 | 93 |
| `shake_bottle` | 100 | 100 | 100 | 100 |
| `shake_bottle_horizontally` | 100 | 100 | 100 | 100 |
| `stack_blocks_three` | 96 | 99 | 97 | 99 |
| `stack_blocks_two` | 99 | 100 | 100 | 99 |
| `stack_bowls_three` | 78 | 72 | 83 | 83 |
| `stack_bowls_two` | 94 | 92 | 97 | 96 |
| `stamp_seal` | 79 | 98 | 84 | 97 |
| `turn_switch` | 77 | 64 | 79 | 79 |
| **mean** | 93.20 | 92.82 ± 0.33 | 93.56 | 94.00 ± 0.32 |
