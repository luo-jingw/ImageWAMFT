# ImageWAM Training Recipe

Source: `papers/ImageWAM_2606.19531.pdf` (arXiv 2606.19531) and the code in `ImageWAM/`.

## Initialization

- ImageWAM uses no embodied-data pretraining. Each benchmark model is trained only on that benchmark's demonstrations, starting from the pretrained image-editing model (§4.1; P.T. ✗ in Tables 1–4 and 7).
- The Action DiT is initialized by copying weights from the editing model. For FLUX.2, the lower action layers copy the double-stream image-stream blocks and the higher layers copy the single-stream blocks (Appendix 5.1.2). Action DiT size: 642M (4B variant), 952M (9B variant). The text encoder is the Qwen3-4B or Qwen3-8B that the corresponding FLUX.2 variant uses.
- The InternData-A1 pretrain → fine-tune path in the README and in `ImageWAM/scripts/flux2/run_finetune_flux2_klein_robotwin_ee16.sh` was released after the paper and is not part of the paper's experiments.

## Real-World Setting

- Data: 4 tasks on a Dobot XTrainer dual-arm robot, about 100 trajectories per task, trained jointly as one model.
- Preprocessing follows RoboTwin. The two wrist views are resized and placed side by side, then stacked below the main view. Final input is 288×256. Future horizon is 16 frames and the action chunk length is 16.
- Training length: 30k steps (§4.1) and 10 epochs (Appendix 5.2).
- Backbone: FLUX.2 4B in §4.1, but Table 10 lists OmniGen2 for the real-world robot (18 hours, batch 16 per GPU). The paper has no real-world result for FLUX.2 9B.
- Real-world success rate (Table 4): 84.5% average over T1–T4.

## FLUX.2 9B

- Reported only on LIBERO-Plus (Table 7): 83.1% (4B) vs 85.2% (9B).
- Trained with DeepSpeed ZeRO-2. LIBERO run: 1.6 days on 8× H20, batch 12 per GPU.
- Repository config `ImageWAM/configs/task/robotwin_flux2_klein_9b_base_clean_imagewam.yaml`: batch 4 per GPU, gradient accumulation 2, 5 epochs.

## Common Hyperparameters (Table 8)

bf16, AdamW with betas (0.9, 0.95), learning rate 1e-4, weight decay 1e-2, warmup-cosine schedule with warmup equal to 5% of total steps, minimum LR 0.01× the learning rate, gradient clipping 1.0, DeepSpeed ZeRO-1 (ZeRO-2 for 9B).

## How Steps Relate to Epochs in the Code

- Dataset length is the total frame count of the training split: `BaseLeRobotDataset.__len__` with `sample_index_stride` 1. Each frame is one sample.
- Optimizer steps per epoch = ceil(ceil(frames / (batch_per_gpu × num_gpus)) / gradient_accumulation_steps) (`Trainer._estimate_total_train_steps`). `max_steps`, when set, overrides `num_epochs`.
