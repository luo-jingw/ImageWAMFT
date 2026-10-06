# Test-Time Registers for ImageWAM on LIBERO

## Question

Can a training-free register take over the attention sink of the ImageWAM-FLUX.2-4B video expert, and how does it change inference: the attention of the image and action tokens, the predicted actions, and the LIBERO success rate?

## Background: Where the Sink Is

Measured on 32 restored states (attention of the 392 reference-image queries in the prefix pass, per head and layer; residual norms per token):
- **The sink is the prompt template.** The image queries put 32–58% of their attention (layers 5–10) on the 18 template tokens of "A video recorded from a robot's point of view executing the following instruction:". These are about 4% of the keys.
- **No image token is a sink.** The largest image key takes at most about 1% of the attention. Image-token norms vary by at most 1.8× from the median, so there are no ViT-style high-norm artifact tokens.
- **The template tokens carry massive activations.**
  - In the single blocks their residual norm is 60–300× that of an image token.
  - 97% of it lies in four channels: 2769, 364, 2808 and 131.
  - The text MLP of layer 4, the last double block, writes it: an update of norm about 43,000 on the template tokens, against 4,000 on task tokens.
- **The sink keys have small values.** Their value norms are 0.1–0.35× the median, the classic sink signature.

## Mechanism Under Test

`register.py`, after Jiang et al. (2025), "Vision Transformers Don't Need Trained Registers".

**Register tokens**
- *r* tokens are appended to the text context, right after the proprio token.
- They are zero-initialized; the text input projection has no bias, so they start at zero.
- They take the next text RoPE positions and are valid keys for every query.

**Shift "neurons" (as in the paper)**
- Neurons of the layer-4 text MLP are ranked by their contribution to the massive update along its direction (`find_neurons.py`, 40 calibration states).
- The update is not sparse: 23 neurons give 50% of it, 134 give 90% and 1,057 give 99%.
- For the top-*k* neurons, the post-activation value on the register tokens is set to the neuron's largest-magnitude value over the valid text tokens. The value on every other token is set to 0.

**Shift "direction" (adapted)**
- The component of the layer-4 text MLP output along the massive direction is removed from every text token.
- The largest-magnitude component is put on the register tokens.

**Controls**
- `tokens_r*`: register tokens are added and nothing is moved.
- `ablate_*`: the neurons or the direction are removed everywhere, and no tokens are added.

## Grid

1. **Open-loop screen** (`screen_register.py`), 19 configs.
   - Configs:
     - `none`, `tokens_r1`, `tokens_r16`;
     - direction shift: `ablate_dir`, `ttr_dir_r{1,4,16}`;
     - neuron shift at *k* = 50/90/99%: `ablate_n{50,90,99}`, `ttr_n{50,90,99}_r{1,4,16}`.
   - States: the 60 evaluation states of the quantization screen (tasks 0–4 of every suite, calls 1, 3 and 6).
   - Measures:
     - action deviation from the unmodified model with the same noise, and gripper flips;
     - per-layer attention shares of the image queries (prefix) and action queries (first denoising step) on the template, task, register and image keys;
     - token norms entering the first single block.
2. **Closed loop** (`eval_register.py`), 200 episodes per config (4 suites × 10 tasks × 5 initial states).
   - Configs:
     - `ttr_n99_r16`: the paper's neuron shift, with the sink moved almost fully at the smallest deviation;
     - `ttr_dir_r16`: the direction shift, with the sink moved fully;
     - `ablate_dir`: the sink removed without registers, as a control.
   - Settings as in the token-mask and quantization studies: OSMesa, seed 42, 10 denoising steps, replanning every 12 steps.
   - At every call a paired inference of the unmodified model is recorded.
   - The unmodified success rate (100%, 200 episodes) comes from the token-mask study.

## Environment

`.venv-libero` on `nnmc71`. The screen ran on GPU 0; the closed loop runs on GPUs 0–3 with 3 workers each (`research/libero_token_mask/run_grid.sh` with `EVAL_SCRIPT=eval_register.py`).

## Outputs

- `runs/register_neurons.json`, `runs/register_direction.npy`: neuron ranking and massive direction.
- `runs/screen.json`: open-loop screen.
- `runs/<date>_register_grid/<config>/<suite>/taskXX/trial{t}.json|npz|mp4`: closed loop.
- `figures/` (`make_figures.py`), `RESULTS.md`.
