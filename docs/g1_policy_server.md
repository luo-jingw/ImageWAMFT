# G1 Policy Server

The server serves the three G1 ImageWAM runs to the physical robot and offers a local web page to switch the model and edit inference settings. Code: `deploy/g1_server/`. Design: `plan.md` → D13–D19.

| Model | Run directory |
|---|---|
| FLUX.2-klein-base-4B (`flux2_4b`) | `checkpoints/g1_lgg100_flux2_klein_4b_base_imagewam/` |
| Qwen-Image-2.1 (`qwenimage21`) | `checkpoints/g1_lgg100_qwenimage21_imagewam/` |
| FLUX.2-klein-base-9B (`flux2_9b`) | `checkpoints/g1_lgg100_flux2_klein_9b_base_imagewam/` |

## Running

Server, on a node with an idle GPU (at most 4 project GPUs per node):

```bash
tmux new-session -d -s imagewamft_g1_server \
  "GPU=0 MODELS=flux2_4b,qwenimage21,flux2_9b bash /home/user1/workspace/jingwu/ImageWAMFT/launch/g1_server.sh"
```

- Environment variables:
  - `GPU` (required);
  - `MODELS`: models loaded at start at their latest step, the first one active; default `flux2_4b`;
  - `WS_PORT` (default 8000) and `UI_PORT` (default 7860);
  - `SHARE=1` adds a Gradio share link;
  - `AUTH_FILE`: a file holding `user:password` that protects the page.
- All three models stay resident in 42.9 GiB; loading takes 13.5 s (4B), 17.5 s (Qwen-Image-2.1) and 32.5 s (9B).
- Logs: `deploy/g1_server/logs/server_<time>.log`.
- Web page: `http://<node>:7860` on the local network.
- On `nnmc71` (2026-10-05), the Gradio share link did not come up. The tunnel request and the frpc download succeeded, but no public URL appeared within 10 minutes.

Robot side, from the `tmp/g1-client` repository root:

```bash
python fastwam/main.py --iface <robot interface> --server-host <node> --server-port 8000 --prompt "<task string>"
```

- Camera preview without moving the robot: `python fastwam/preview_cameras.py --server-host <node> --server-port 8000 --fps 5`.
  - The script lives in `tmp/g1-client/fastwam/preview_cameras.py`.
  - It reads only the cameras, through the image server: no DDS and no arm authority.
  - The page shows each frame and the model input. Use it to match the scene and the head-camera view to the training data.
- `fastwam/main.py` and `fastwam/main_sync.py` are joint-space loops and match these checkpoints.
- `fastwam/main_eef.py` and `fastwam/main_mask.py` expect end-effector actions, which no G1 run produces.

The four training task strings:
- **drawer:** Use the arm nearer to the drawer to pull it open, then use the other arm to pick up the purple cube and place it inside the drawer, and finally close the drawer.
- **biomanual:** Pick up the blue cup with the nearer arm and place it on the blue coaster, then pick up the green cup with the nearer arm and place it on the now-empty green coaster.
- **simple-pick:** Pick up the pink cube with the nearer arm and place it into the green bin.
- **Stack-the-cubes:** Stack the blocks by color: put the red block in the center, then stack the blue block on the red block, then stack the yellow block on the blue block.

## Protocol

The server uses the WebSocket + msgpack protocol of `tmp/g1-client/g1_client/policy_client.py` without changes. The NumPy codec in `deploy/g1_server/msgpack_numpy.py` mirrors the client's.

| Message | Content |
|---|---|
| Metadata, sent on connect | `policy`, `model`, `checkpoint_step`, `camera_keys`, `state_dim` 16, `action_dim` 16, `action_horizon`, `action_layout`, `fps` 30, `output_mode` |
| Request | `image`: list of 3 RGB uint8 HWC views in order `cam_left_high`, `cam_left_wrist`, `cam_right_wrist`; `state`: float32[16], with 14 arm joints (rad, left then right: pitch, roll, yaw, elbow, wrist roll, pitch, yaw) and the left and right gripper; `prompt`: task string |
| Response | `actions`: float32[H, 16], absolute de-normalized targets in the state layout, either the model's chunk or a hold chunk (see Safety Controls); `actions_source` (`model` or `hold`); `output_mode`; `safety_ok` and `safety_reasons`; `server_timing` (`preprocess_ms`, `text_ms`, `model_ms`, `total_ms`); `model`; `checkpoint_step`; `num_inference_steps`; `seed` |
| Reset `{"reset": true}` | answered with `{}` |
| Camera preview `{"preview": true, "image": [three views]}` | shown on the web page with its 288×256 model input; no inference; answered with `{}` |
| Observation in the stop mode | a text frame `ServingStopped: …`; the client raises `RuntimeError` |
| Invalid request or inference error | a text frame with the traceback; the client raises `RuntimeError`, and the connection stays usable |

## Safety Controls

Set at the top of the web page and applied to the next request. A banner shows the current output mode, the safety-check settings and the robot-side address.

| Output mode | Inference | Reply to the robot | Robot (`fastwam/main.py`) |
|---|---|---|---|
| Inference only (startup default) | yes, shown on the page | a hold chunk: the request's own 16-dim state repeated H times | holds still and keeps sending observations |
| Execute | yes | the model's chunk if it passes the safety check, otherwise a hold chunk | moves |
| Stop | no | `ServingStopped` error text | raises, runs its cleanup and returns arm authority to the locomotion service |

- **Mode changes:**
  - Switching to execute requires a confirmation checkbox.
  - A stop button switches to stop immediately.
  - The controls are read again after each inference, so a switch made during an inference already applies to its reply.
- **Action safety check** (`deploy/g1_server/action_safety.py`), applied to every inferred chunk; thresholds editable on the page:
  - **Range:** every value lies within the run's training action range (`dataset_stats.json` → `action.global_min/max`), widened by 10% of the range on each side.
  - **First-step jump:** the largest arm-joint difference between the first action and the request's state is at most 0.3 rad. Grippers are excluded, because they open and close within single steps.
  - **Why 0.3 rad:** in LGG100 the recorded action-to-state difference is p50 0.035, p99 0.148 and p99.9 0.265 rad over 185,673 frames.
  - In execute mode a failed chunk is replaced by a hold chunk. The reasons appear on the page and in the request log.
- **Connections:**
  - **Port change:** the page moves the WebSocket listener to another port (1024–65535). It binds the new port first; a port that cannot be bound leaves the old listener running. Clients connected to the old port are closed and must reconnect to the new one.
  - **Client list and disconnect:** the page lists the connected clients and can close all connections; a closed client raises and runs its cleanup.
  - A restarted server starts on `WS_PORT` again, in the inference-only mode.

## Inputs

- **Image** (any input resolution). Each view is converted to a float tensor and goes through:
  1. the run's processor validation transform: ToTensor, then Resize to 240×320;
  2. a bilinear antialiased resize to the RoboTwin compact tiles: `cam_left_high` 192×256, each wrist 96×128;
  3. tiling (head on top, left and right wrist below, 288×256), `ResizeSmallestSideAspectPreserving`, `CenterCrop` and normalization to [-1, 1].

  These are the same classes the training dataset uses. The robot sends 256×320 frames, resized from 480×640. All resizes are full-frame, so no crop differs from training.
- **State**: z-score with the run's `dataset_stats.json` through the processor normalizer. Actions are de-normalized with the same statistics.
- **Text**: the task string is wrapped in `DEFAULT_PROMPT` and looked up by SHA-256 in the run's training text cache, so the four training task strings use the training features. Any other string is encoded once and cached in `deploy/g1_server/cache/<model>/`:
  - FLUX.2: by the model's Qwen3 text encoder (loaded on first use);
  - Qwen-Image-2.1: by `ImageWAM/scripts/qwenimage21/precompute_qwenimage21_text_embeds.py` in `.venv-qwenimage`, as a subprocess. That script loads the full Qwen-Image-2.1 pipeline on the server GPU for the duration.

## Web Page

- **Model**: choose a model and a checkpoint step, then load and switch to it, or unload it.
  - Loaded models stay resident on the GPU; switching between them is immediate.
  - Requests wait while a model loads.
- **Inference settings**, applied to the next request:
  - denoising steps: default 10, as in training validation;
  - sigma shift: default is the run's `infer_shift`, 5.0;
  - seed: fixed (default 42) or random per request;
  - action horizon: default 16, the trained value;
  - prompt source: the client's prompt, or a fixed task string chosen from the training tasks or typed.
- **Last request**, refreshed every 0.5 s: the three received views and the 288×256 model input. For an inference request the page also shows the action chunk plotted against the current state, the timing and the request log; camera previews do not enter the log.
- **Image preview**: a button runs the joint image and action inference on the last observation and shows the predicted frame 16 steps ahead.
- **Request record**: per inference request, the output mode, what was sent (model actions, hold in inference-only mode, hold after a failed safety check, or rejected in stop mode), the safety-check result and the largest first-step arm jump.

## Verification

- **Inputs** (`deploy/g1_server/policy_core_check.py --cpu-only`). On 12 evenly spaced samples of the four held-out episodes:
  - The server image tensor, built from the raw 480×640 frames, equals the training dataset's `video[:, 0]` (maximum absolute difference 0).
  - The server proprio equals the dataset's normalized `proprio[0]` (difference 0).
- **Model path** (`policy_core_check.py`, H100 NVL; 12 held-out samples, seed 42, 10 denoising steps).
  - `G1Policy.infer` returns the same actions as `ImageWAM.infer_action` on the dataset sample with the cached text features (maximum absolute difference 0).
  - All weights load with 0 missing and 0 unexpected MoT keys.

  | Model | Step | Load | GPU memory after load | L1 to recorded actions (rad) | Model time, median | Request total, median |
  |---|---|---|---|---|---|---|
  | FLUX.2-klein-base-4B | 27,950 | 13.5 s | 8.9 GiB | 0.026 | 156 ms | 170 ms |
  | Qwen-Image-2.1 | 27,950 | 20.5 s | 15.5 GiB | 0.027 | 237 ms | 258 ms |
  | FLUX.2-klein-base-9B | 27,950 | 34.8 s | 19.2 GiB | 0.025 | 202 ms | 215 ms |

- **FLUX.2 online text encoding vs. the training cache** (four task strings):
  - Masks are equal.
  - Relative L2 difference of the features is 0.05–0.08% (4B) and 0.07–0.11% (9B), bf16 rounding.
  - Resulting actions differ by 0.0002–0.0004 rad on average and at most 0.0085 rad.
  - The lazily loaded Qwen3 encoder raises GPU memory to 16.2 GiB (4B) and 35.1 GiB (9B).
- **Replay through the robot client** (`replay_check.py`; server started with `launch/g1_server.sh` on one H100 NVL; 40 samples, every 64 frames of the four held-out episodes). Every response is float32 [16, 16].

  | Model | Arm L1 (rad) | Gripper L1 | Camera path vs. raw frames, action L1 | Round trip, median | Server model time, median |
  |---|---|---|---|---|---|
  | FLUX.2-klein-base-4B, step 27,950 | 0.022 | 0.028 | 0.0010 | 192 ms | 184 ms |
  | Qwen-Image-2.1, step 27,950 | 0.022 | 0.036 | 0.0010 | 261 ms | 249 ms |
  | FLUX.2-klein-base-9B, step 27,950 | 0.022 | 0.031 | 0.0010 | 235 ms | 223 ms |

  - The robot camera path (256×320 INTER_AREA resize and JPEG quality 90) changes the actions by 0.001 rad on average.
  - Latency is measured while three LIBERO evaluation jobs ran on other GPUs of the node.
  - Per task, the drawer task has the largest arm error, 0.042 rad for both FLUX.2 models; the other three are 0.014–0.019 rad.
- **Web page** (driven through its Gradio API with the server running):
  - Loading through the page takes 21.0 s for Qwen-Image-2.1 and 40.3 s for the 9B model.
  - With all three models resident, 42.9 GiB is allocated, and switching to a resident model takes 0.07 s.
  - A settings change (20 steps, a random seed) appears in the next response: `num_inference_steps` 20, a new seed per request, model time 324–346 ms. Restoring 10 steps and seed 42 returns to 197 ms.
  - 240 requests were served without failure.
- **Protocol**: the unmodified `FastWAMPolicy` client connects to `PolicyWebSocketServer` and receives:
  - the metadata;
  - `{}` for a reset;
  - a float32 [H, 16] action chunk (H follows the setting);
  - `RuntimeError` for a malformed request, after which the same connection keeps working.
- **Safety controls** (2026-10-05; the server on one H100 NVL with all three models resident; FLUX.2-4B active; unmodified `FastWAMPolicy` client; held-out LGG100 samples):
  - **Inference only (startup default):** 40 of 40 replies are hold chunks equal to the request state.
  - **Confirmation:** switching to execute without the confirmation is refused.
  - **Execute,** 316 held-out samples:
    - every reply carries the model's chunk, and the safety check flags none of them;
    - the model's first-step arm jump has median 0.037, p99 0.120 and max 0.162 rad;
    - arm L1 to the recorded actions is 0.0209 rad, as before.
  - **Port change:** moving to port 8001 closes 8000, and the old connection is closed (`ConnectionClosedOK`).
  - **Stop:** the stop button makes the next request raise `ServingStopped` in the client.
  - **Without GPU,** with a stub policy:
    - a first-step jump of 0.5 rad and out-of-range values each return a hold chunk with the reason;
    - a move to an occupied port raises `OSError` and keeps the old listener;
    - closing all connections makes the client raise.
