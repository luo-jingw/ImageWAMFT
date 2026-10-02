"""ImageWAM logic for the Qwen-Image-2.1 backbone (stack "qwenimage21").

Mirrors the FLUX.2 stack in `imagewam.py` (`build_inputs_flux2`, `_build_mot_attention_mask_flux2`,
`_training_loss_flux2`, `infer_action_flux2`, `infer_video_flux2`) with the Qwen-Image-2.1 experts:

  sequence  [text (cached Qwen3-VL features, proprio token packed after the valid text) | reference | target | action]
  mask      text/reference rows -> text + reference; target rows -> text + reference + target;
            action rows -> text + reference + action; padded text columns masked (plan.md D2)
  timesteps prefix tokens use t = 0, target tokens the video timestep, action tokens the action timestep (D4)
  positions Qwen-Image-2.1 counter, action after the target block in training and inference (D5)

`QwenImage21Stack` is a stateless view over an `ImageWAM` instance; the model owns all modules and schedulers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn.functional as F

from .action_dit_qwenimage21 import ActionDiTQwenImage21
from .qwenimage21_video_expert import QwenImage21Positions, QwenImage21VideoExpert

if TYPE_CHECKING:
    from .imagewam import ImageWAM

VAE_STRIDE = 16


@dataclass
class QwenImage21Inputs:
    ref_tokens: torch.Tensor  # [B, h*w, 64] normalized VAE latents
    target_tokens: Optional[torch.Tensor]  # [B, h*w, 64] or None (inference)
    grid: tuple[int, int]  # latent (h, w)
    text_states: torch.Tensor  # [B, L_txt, 4096], proprio packed after the valid text tokens
    text_mask: torch.Tensor  # [B, L_txt] bool
    action: Optional[torch.Tensor] = None  # [B, T, action_dim]
    action_is_pad: Optional[torch.Tensor] = None
    action_dim_is_pad: Optional[torch.Tensor] = None


class QwenImage21Stack:
    def __init__(self, model: "ImageWAM") -> None:
        self.model = model
        self.video_expert: QwenImage21VideoExpert = model.video_expert
        self.action_expert: ActionDiTQwenImage21 = model.action_expert

    # ---------------------------------------------------------------- images
    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """RGB [B, 3, H, W] in [-1, 1] -> packed normalized latents [B, (H/16)*(W/16), 64] and the latent grid."""
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"`image` must be [B,3,H,W] or [3,H,W], got {tuple(image.shape)}")
        if image.shape[-2] % VAE_STRIDE or image.shape[-1] % VAE_STRIDE:
            raise ValueError(f"Qwen-Image-2.1 image dims must be multiples of {VAE_STRIDE}, got {tuple(image.shape[-2:])}")
        model = self.model
        latents = model.vae.encode(image.to(device=model.device, dtype=model.torch_dtype, non_blocking=True))
        grid = (int(latents.shape[-2]), int(latents.shape[-1]))
        return QwenImage21VideoExpert.pack_latents(latents.to(model.torch_dtype)), grid

    @torch.no_grad()
    def decode_tokens(self, tokens: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
        """Packed latents -> RGB [B, 3, H, W] float in [-1, 1]."""
        latents = QwenImage21VideoExpert.unpack_latents(tokens, grid[0], grid[1])
        image = self.model.vae.decode(latents.to(device=self.model.device, dtype=self.model.torch_dtype))
        return image.detach().float().clamp(-1, 1)

    # ---------------------------------------------------------------- text
    def _context(
        self,
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
        proprio: Optional[torch.Tensor],
        source: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context is None or context_mask is None:
            raise ValueError(
                "The Qwen-Image-2.1 stack needs cached `text_hidden_states`/`text_attention_mask` "
                "(scripts/qwenimage21/precompute_qwenimage21_text_embeds.py); online text encoding is not supported."
            )
        model = self.model
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context_mask.ndim == 1:
            context_mask = context_mask.unsqueeze(0)
        context = context.to(device=model.device, dtype=model.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=model.device, dtype=torch.bool, non_blocking=True)
        return model._append_proprio_to_context_if_enabled(
            context=context, context_mask=context_mask, proprio=proprio, source=source
        )

    # ---------------------------------------------------------------- inputs
    def build_inputs(self, sample: dict[str, Any]) -> QwenImage21Inputs:
        model = self.model
        video = sample.get("video")
        current_frame = sample.get("current_frame", sample.get("input_image"))
        next_frame = sample.get("next_frame", sample.get("target_image"))
        if video is not None:
            if video.ndim != 5:
                raise ValueError(f"`sample['video']` must be [B,C,T,H,W], got {tuple(video.shape)}")
            current_frame = video[:, :, 0] if current_frame is None else current_frame
            next_frame = video[:, :, -1] if next_frame is None else next_frame
        if current_frame is None or next_frame is None:
            raise ValueError("Qwen-Image-2.1 stack sample requires `video` or `current_frame`/`next_frame`.")
        ref_tokens, grid = self.encode_image(current_frame)
        target_tokens, target_grid = self.encode_image(next_frame)
        if target_grid != grid:
            raise ValueError(f"Reference grid {grid} and target grid {target_grid} differ")
        text_states, text_mask = self._context(
            sample.get("text_hidden_states"),
            sample.get("text_attention_mask"),
            sample.get("proprio"),
            "Qwen-Image-2.1 training sample",
        )
        action = sample["action"].to(device=model.device, dtype=model.torch_dtype, non_blocking=True)
        action_is_pad = sample.get("action_is_pad")
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=model.device, dtype=torch.bool, non_blocking=True)
        action_dim_is_pad = sample.get("action_dim_is_pad")
        if action_dim_is_pad is not None:
            action_dim_is_pad = action_dim_is_pad.to(device=model.device, dtype=torch.bool, non_blocking=True)
        return QwenImage21Inputs(
            ref_tokens=ref_tokens,
            target_tokens=target_tokens,
            grid=grid,
            text_states=text_states,
            text_mask=text_mask,
            action=action,
            action_is_pad=action_is_pad,
            action_dim_is_pad=action_dim_is_pad,
        )

    def positions(self, text_len: int, grid: tuple[int, int], action_len: int) -> QwenImage21Positions:
        return self.video_expert.build_positions(text_len, grid, grid, action_len, self.model.device)

    @staticmethod
    def attention_mask(
        text_mask: torch.Tensor,
        ref_len: int,
        target_len: int,
        action_len: int,
    ) -> torch.Tensor:
        """Boolean [B, L, L] mask (True = attend) for [text | reference | target | action] (plan.md D2)."""
        batch, txt_len = text_mask.shape
        r0 = txt_len
        x0 = r0 + ref_len
        a0 = x0 + target_len
        total = a0 + action_len
        mask = torch.zeros(batch, total, total, dtype=torch.bool, device=text_mask.device)
        mask[:, :x0, :x0] = True
        mask[:, x0:a0, :a0] = True
        mask[:, a0:, :x0] = True
        mask[:, a0:, a0:] = True
        mask[:, :, :r0] &= text_mask.to(torch.bool)[:, None, :]
        return mask

    # ---------------------------------------------------------------- training
    def training_loss(self, sample: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        model = self.model
        inputs = self.build_inputs(sample)
        target = inputs.target_tokens
        action = inputs.action
        batch = int(target.shape[0])

        noise_video = torch.randn_like(target)
        timestep_video = model.train_video_scheduler.sample_training_t(batch_size=batch, device=model.device, dtype=target.dtype)
        noisy_target = model.train_video_scheduler.add_noise(target, noise_video, timestep_video)
        target_video = model.train_video_scheduler.training_target(target, noise_video, timestep_video)

        noise_action = torch.randn_like(action)
        timestep_action = model.train_action_scheduler.sample_training_t(batch_size=batch, device=model.device, dtype=action.dtype)
        noisy_action = model.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = model.train_action_scheduler.training_target(action, noise_action, timestep_action)

        text_len = int(inputs.text_states.shape[1])
        positions = self.positions(text_len, inputs.grid, int(action.shape[1]))
        video_pre = self.video_expert.pre_dit(
            target_latents=noisy_target,
            video_timestep=model._scheduler_timestep_to_unit(timestep_video, model.train_video_scheduler),
            text_states=inputs.text_states,
            text_mask=inputs.text_mask,
            ref_latents=inputs.ref_tokens,
            positions=positions,
        )
        action_pre = self.action_expert.pre_dit(
            noisy_action, model._scheduler_timestep_to_unit(timestep_action, model.train_action_scheduler)
        )
        mask = self.attention_mask(inputs.text_mask, video_pre.ref_len, video_pre.target_len, int(action.shape[1]))
        tokens_out = model.mot(
            embeds_all={"video": video_pre.tokens, "action": action_pre.tokens},
            attention_mask=mask,
            freqs_all={"video": video_pre.rotary, "action": positions.action},
            context_all={"video": None, "action": None},
            t_mod_all={"video": video_pre.modulation, "action": action_pre.modulation},
        )
        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        video_loss_per_sample = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").flatten(1).mean(dim=1)
        video_weight = model.train_video_scheduler.training_weight(timestep_video).to(
            video_loss_per_sample.device, dtype=video_loss_per_sample.dtype
        )
        loss_video = (video_loss_per_sample * video_weight).mean()
        action_loss_per_sample = model._compute_action_loss_per_sample(
            pred_action=pred_action,
            target_action=target_action,
            action_is_pad=inputs.action_is_pad,
            action_dim_is_pad=inputs.action_dim_is_pad,
        )
        action_weight = model.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()
        loss_total = model.loss_lambda_video * loss_video + model.loss_lambda_action * loss_action
        return loss_total, {
            "loss_video": model.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": model.loss_lambda_action * float(loss_action.detach().item()),
        }

    # ---------------------------------------------------------------- inference
    def _inference_inputs(
        self,
        input_image: torch.Tensor,
        proprio: Optional[torch.Tensor],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
        source: str,
    ) -> QwenImage21Inputs:
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")
        ref_tokens, grid = self.encode_image(input_image)
        text_states, text_mask = self._context(context, context_mask, proprio, source)
        return QwenImage21Inputs(ref_tokens=ref_tokens, target_tokens=None, grid=grid, text_states=text_states, text_mask=text_mask)

    def _initial_noise(self, shape: tuple[int, ...], seed: Optional[int], rand_device: str) -> torch.Tensor:
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=rand_device, dtype=torch.float32)
        return noise.to(device=self.model.device, dtype=self.model.torch_dtype)

    def prefill_prefix(self, inputs: QwenImage21Inputs, action_len: int) -> tuple[list[dict[str, torch.Tensor]], int, QwenImage21Positions]:
        """Runs the t = 0 prefix [text | reference] through the video expert once and returns its per-block K/V."""
        model = self.model
        batch = int(inputs.ref_tokens.shape[0])
        positions = self.positions(int(inputs.text_states.shape[1]), inputs.grid, action_len)
        video_pre = self.video_expert.pre_dit(
            target_latents=None,
            video_timestep=torch.zeros(batch, device=model.device, dtype=model.torch_dtype),
            text_states=inputs.text_states,
            text_mask=inputs.text_mask,
            ref_latents=inputs.ref_tokens,
            positions=positions,
        )
        prefix_mask = self.attention_mask(inputs.text_mask, video_pre.ref_len, 0, 0)
        kv_cache = model.mot.prefill_video_cache(
            video_tokens=video_pre.tokens,
            video_freqs=video_pre.rotary,
            video_t_mod=video_pre.modulation,
            video_context_payload=None,
            video_attention_mask=prefix_mask,
        )
        return kv_cache, int(video_pre.tokens.shape[1]), positions

    @torch.no_grad()
    def infer_action(
        self,
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
    ) -> dict[str, Any]:
        model = self.model
        model.eval()
        inputs = self._inference_inputs(input_image, proprio, context, context_mask, "Qwen-Image-2.1 action inference")
        kv_cache, prefix_len, positions = self.prefill_prefix(inputs, int(action_horizon))
        full_mask = self.attention_mask(inputs.text_mask, prefix_len - int(inputs.text_states.shape[1]), 0, int(action_horizon))
        latents = self._initial_noise((1, int(action_horizon), self.action_expert.action_dim), seed, rand_device)
        timesteps, deltas = model.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps, device=model.device, dtype=latents.dtype, shift_override=sigma_shift
        )
        for step_t, step_delta in zip(timesteps, deltas):
            timestep = step_t.expand(1).to(device=model.device, dtype=latents.dtype)
            action_pre = self.action_expert.pre_dit(latents, model._scheduler_timestep_to_unit(timestep, model.infer_action_scheduler))
            tokens = model.mot.forward_action_with_video_cache(
                action_tokens=action_pre.tokens,
                action_freqs=positions.action,
                action_t_mod=action_pre.modulation,
                action_context_payload=None,
                video_kv_cache=kv_cache,
                attention_mask=full_mask,
                video_seq_len=prefix_len,
            )
            pred = self.action_expert.post_dit(tokens, action_pre)
            latents = model.infer_action_scheduler.step(pred, step_delta, latents)
        return {"action": latents[0].detach().to(device="cpu", dtype=torch.float32)}

    def forward_video_only(self, video_pre, mask: torch.Tensor) -> torch.Tensor:
        """Video expert alone over [text | reference | target] (no action tokens)."""
        mot = self.model.mot
        x = video_pre.tokens
        for block in self.video_expert.blocks:
            built = block.prepare_qkv(x, video_pre.rotary, video_pre.modulation)
            mixed = mot._mixed_attention(built.q, built.k, built.v, mask)
            x = block.apply_post(built.residual_x, mixed, built.post_state)
        return x

    @torch.no_grad()
    def infer_image(
        self,
        input_image: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
    ) -> dict[str, Any]:
        model = self.model
        model.eval()
        inputs = self._inference_inputs(input_image, proprio, context, context_mask, "Qwen-Image-2.1 image inference")
        latents = self._initial_noise(tuple(inputs.ref_tokens.shape), seed, rand_device)
        positions = self.positions(int(inputs.text_states.shape[1]), inputs.grid, 0)
        timesteps, deltas = model.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps, device=model.device, dtype=latents.dtype, shift_override=sigma_shift
        )
        for step_t, step_delta in zip(timesteps, deltas):
            timestep = step_t.expand(1).to(device=model.device, dtype=latents.dtype)
            video_pre = self.video_expert.pre_dit(
                target_latents=latents,
                video_timestep=model._scheduler_timestep_to_unit(timestep, model.infer_video_scheduler),
                text_states=inputs.text_states,
                text_mask=inputs.text_mask,
                ref_latents=inputs.ref_tokens,
                positions=positions,
            )
            mask = self.attention_mask(inputs.text_mask, video_pre.ref_len, video_pre.target_len, 0)
            tokens = self.forward_video_only(video_pre, mask)
            pred = self.video_expert.post_dit(tokens, video_pre)
            latents = model.infer_video_scheduler.step(pred, step_delta, latents)
        image = self.decode_tokens(latents, inputs.grid)
        return {"image": image[0].detach().cpu()}
