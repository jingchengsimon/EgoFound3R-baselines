from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.utils.checkpoint as checkpoint_utils
from torch import nn

from egohandmetric_prompt.models.token_layout import TokenLayout
from egohandmetric_prompt.vendor import ensure_vendor_paths


def freeze_module(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False
    return module


@dataclass(slots=True)
class VggtOmegaModules:
    aggregator: nn.Module
    camera_head: nn.Module
    dense_head: nn.Module


@dataclass(slots=True)
class VggtPrefixOutput:
    dino_feature_map: torch.Tensor
    dino_tokens: torch.Tensor
    mid_tokens: torch.Tensor
    public_tokens: torch.Tensor
    internal_tokens: torch.Tensor
    layout: TokenLayout
    patch_token_start: int
    aggregated_tokens_list: list[torch.Tensor | None]


@dataclass(slots=True)
class VggtSuffixOutput:
    aggregated_tokens_list: list[torch.Tensor | None]
    final_public_tokens: torch.Tensor
    final_internal_tokens: torch.Tensor
    final_hand_prompts: torch.Tensor
    final_metric_prompts: torch.Tensor


class VggtOmegaFrozenWrapper(nn.Module):
    def __init__(
        self,
        modules: VggtOmegaModules,
        *,
        image_height: int,
        image_width: int,
        mid_layer_index: int = 11,
        suffix_activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.aggregator = freeze_module(modules.aggregator)
        self.camera_head = freeze_module(modules.camera_head)
        self.dense_head = freeze_module(modules.dense_head)
        self.image_height = image_height
        self.image_width = image_width
        aggregator_depth = getattr(self.aggregator, "depth", None)
        if aggregator_depth is not None and (mid_layer_index < 0 or mid_layer_index >= aggregator_depth):
            raise ValueError(
                f"mid_layer_index 必须在 [0, {aggregator_depth - 1}] 范围内，当前为 {mid_layer_index}。"
            )
        self.mid_layer_index = mid_layer_index
        self.suffix_activation_checkpointing = suffix_activation_checkpointing

    @classmethod
    def from_vendor(
        cls,
        *,
        checkpoint_path: str | Path,
        image_height: int,
        image_width: int,
        mid_layer_index: int = 11,
        suffix_activation_checkpointing: bool = False,
    ) -> "VggtOmegaFrozenWrapper":
        ensure_vendor_paths()
        from vggt_omega.models.heads.camera_head import CameraHead
        from vggt_omega.models.heads.dense_head import DenseHead
        from vggt_omega.models.aggregator import Aggregator

        modules = VggtOmegaModules(
            aggregator=Aggregator(),
            camera_head=CameraHead(),
            dense_head=DenseHead(),
        )
        wrapper = cls(
            modules,
            image_height=image_height,
            image_width=image_width,
            mid_layer_index=mid_layer_index,
            suffix_activation_checkpointing=suffix_activation_checkpointing,
        )
        state_dict = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
        missing, unexpected = wrapper.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"VGGT-Omega checkpoint 加载不完整: missing={missing[:10]} unexpected={unexpected[:10]}"
            )
        return wrapper

    def train(self, mode: bool = True):
        super().train(mode)
        self.aggregator.eval()
        self.camera_head.eval()
        self.dense_head.eval()
        return self

    def _select_mid_layer_tokens(self, aggregated_tokens_list: list[torch.Tensor | None]) -> torch.Tensor:
        if 0 <= self.mid_layer_index < len(aggregated_tokens_list):
            selected = aggregated_tokens_list[self.mid_layer_index]
            if selected is not None:
                return selected
        for tokens in reversed(aggregated_tokens_list):
            if tokens is not None:
                return tokens
        raise ValueError("Aggregator 没有返回任何 cached layer tokens。")

    def _cached_tokens_device_type(self, aggregated_tokens_list: list[torch.Tensor | None]) -> str:
        for tokens in aggregated_tokens_list:
            if tokens is not None:
                return tokens.device.type
        raise ValueError("Head decode 需要至少一个 cached layer tokens。")

    def _prepare_patch_tokens(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_frames, channels, height, width = images.shape
        normalized = (images - self.aggregator._resnet_mean) / self.aggregator._resnet_std
        patch_tokens = self.aggregator.patch_embed(normalized.view(batch_size * num_frames, channels, height, width))
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
        patch_count, dino_dim = patch_tokens.shape[1:]
        patch_height = height // self.aggregator.patch_size
        patch_width = width // self.aggregator.patch_size
        dino_tokens = patch_tokens.view(batch_size, num_frames, patch_count, dino_dim)
        dino_feature_map = dino_tokens.transpose(-1, -2).reshape(
            batch_size,
            num_frames,
            dino_dim,
            patch_height,
            patch_width,
        )
        return patch_tokens, dino_tokens, dino_feature_map

    def _initial_public_tokens(self, patch_tokens: torch.Tensor, *, batch_size: int, num_frames: int) -> torch.Tensor:
        embed_dim = patch_tokens.shape[-1]
        if hasattr(self.aggregator, "camera_token") and hasattr(self.aggregator, "register_token"):
            from vggt_omega.models.aggregator import slice_expand_and_flatten

            camera_token = slice_expand_and_flatten(self.aggregator.camera_token, batch_size, num_frames)
            register_token = slice_expand_and_flatten(self.aggregator.register_token, batch_size, num_frames)
        else:
            camera_token = patch_tokens.new_zeros(batch_size * num_frames, 1, embed_dim)
            register_token = patch_tokens.new_zeros(
                batch_size * num_frames,
                self.aggregator.patch_token_start - 1,
                embed_dim,
            )
        public_tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        return public_tokens.view(batch_size, num_frames, public_tokens.shape[1], embed_dim)

    def _rope(self, *, patch_height: int, patch_width: int):
        if not hasattr(self.aggregator, "rope_embed"):
            return None
        with torch.no_grad():
            rope_sin, rope_cos = self.aggregator.rope_embed(H=patch_height, W=patch_width)
        return (
            rope_sin.to(dtype=torch.float32),
            rope_cos.to(dtype=torch.float32),
        )

    def _cache_layer_output(
        self,
        outputs: list[torch.Tensor | None],
        *,
        block_idx: int,
        frame_tokens: torch.Tensor,
        tokens: torch.Tensor,
        layout: TokenLayout,
    ) -> None:
        if block_idx not in self.aggregator.cached_layer_indices:
            return
        public_frame_tokens = layout.internal_to_public(frame_tokens)
        public_tokens = layout.internal_to_public(tokens)
        stripped_frame = public_frame_tokens[:, :, : layout.patch_token_end]
        stripped_tokens = public_tokens[:, :, : layout.patch_token_end]
        outputs[block_idx] = torch.cat([stripped_frame, stripped_tokens], dim=-1)

    def run_prefix(self, images: torch.Tensor, *, inject_layer_idx: int) -> VggtPrefixOutput:
        with torch.no_grad():
            return self._run_prefix(images, inject_layer_idx=inject_layer_idx)

    def _run_prefix(self, images: torch.Tensor, *, inject_layer_idx: int) -> VggtPrefixOutput:
        batch_size, num_frames, channels, height, width = images.shape
        if channels != 3:
            raise ValueError(f"期望 RGB 3 通道输入，实际得到 {channels}")
        patch_tokens, dino_tokens, dino_feature_map = self._prepare_patch_tokens(images)
        patch_height = dino_feature_map.shape[-2]
        patch_width = dino_feature_map.shape[-1]
        layout = TokenLayout(
            camera_token_count=1,
            register_token_count=self.aggregator.patch_token_start - 1,
            patch_token_count=patch_tokens.shape[1],
            hand_prompt_count=0,
            patch_grid_hw=(int(patch_height), int(patch_width)),
        )
        public_tokens = self._initial_public_tokens(patch_tokens, batch_size=batch_size, num_frames=num_frames)
        internal_tokens = public_tokens
        num_tokens = internal_tokens.shape[2]
        embed_dim = internal_tokens.shape[3]
        rope = self._rope(patch_height=patch_height, patch_width=patch_width)
        outputs: list[torch.Tensor | None] = [None] * self.aggregator.depth
        tokens = internal_tokens
        for block_idx in range(inject_layer_idx + 1):
            tokens_flat = tokens.view(batch_size * num_frames, num_tokens, embed_dim)
            tokens_flat, frame_tokens = self.aggregator._run_frame_block(
                tokens_flat,
                batch_size,
                num_frames,
                num_tokens,
                embed_dim,
                block_idx,
                rope,
            )
            tokens = self.aggregator._run_inter_frame_attention_block(
                tokens_flat,
                batch_size,
                num_frames,
                num_tokens,
                embed_dim,
                block_idx,
                self.aggregator.inter_frame_attention_types[block_idx] if hasattr(self.aggregator, "inter_frame_attention_types") else "global",
            )
            if tokens.ndim == 3:
                tokens = tokens.view(batch_size, num_frames, num_tokens, embed_dim)
            self._cache_layer_output(outputs, block_idx=block_idx, frame_tokens=frame_tokens, tokens=tokens, layout=layout)
        current_public_tokens = layout.internal_to_public(tokens)
        mid_layer_tokens = self._select_mid_layer_tokens(outputs)
        mid_tokens = mid_layer_tokens[:, :, self.aggregator.patch_token_start :]
        return VggtPrefixOutput(
            dino_feature_map=dino_feature_map,
            dino_tokens=dino_tokens,
            mid_tokens=mid_tokens,
            public_tokens=current_public_tokens,
            internal_tokens=tokens,
            layout=layout,
            patch_token_start=self.aggregator.patch_token_start,
            aggregated_tokens_list=outputs,
        )

    def _run_suffix_block(
        self,
        internal_tokens: torch.Tensor,
        *,
        layout: TokenLayout,
        block_idx: int,
        rope,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_frames, num_tokens, embed_dim = internal_tokens.shape
        original_patch_token_start = getattr(self.aggregator, "patch_token_start", layout.camera_register_end)
        setattr(self.aggregator, "patch_token_start", layout.internal_patch_start)
        try:
            tokens_flat = internal_tokens.view(batch_size * num_frames, num_tokens, embed_dim)
            tokens_flat, frame_tokens = self.aggregator._run_frame_block(
                tokens_flat,
                batch_size,
                num_frames,
                num_tokens,
                embed_dim,
                block_idx,
                rope,
            )
            tokens = self.aggregator._run_inter_frame_attention_block(
                tokens_flat,
                batch_size,
                num_frames,
                num_tokens,
                embed_dim,
                block_idx,
                self.aggregator.inter_frame_attention_types[block_idx] if hasattr(self.aggregator, "inter_frame_attention_types") else "global",
            )
            if tokens.ndim == 3:
                tokens = tokens.view(batch_size, num_frames, num_tokens, embed_dim)
            return tokens, frame_tokens
        finally:
            setattr(self.aggregator, "patch_token_start", original_patch_token_start)

    def _should_checkpoint_suffix_block(self, tokens: torch.Tensor) -> bool:
        return (
            self.suffix_activation_checkpointing
            and self.training
            and torch.is_grad_enabled()
            and tokens.requires_grad
        )

    def run_suffix(
        self,
        internal_tokens: torch.Tensor,
        *,
        layout: TokenLayout,
        start_layer_idx: int,
        prefix_aggregated_tokens_list: list[torch.Tensor | None],
    ) -> VggtSuffixOutput:
        if layout.patch_grid_hw is not None:
            patch_height, patch_width = layout.patch_grid_hw
        else:
            patch_height = int(layout.patch_token_count**0.5)
            patch_width = layout.patch_token_count // max(patch_height, 1)
        rope = self._rope(patch_height=patch_height, patch_width=patch_width)
        outputs = list(prefix_aggregated_tokens_list)
        tokens = internal_tokens
        for block_idx in range(start_layer_idx, self.aggregator.depth):
            if self._should_checkpoint_suffix_block(tokens):
                def run_block(block_tokens, current_block_idx=block_idx):
                    return self._run_suffix_block(
                        block_tokens,
                        layout=layout,
                        block_idx=current_block_idx,
                        rope=rope,
                    )

                tokens, frame_tokens = checkpoint_utils.checkpoint(
                    run_block,
                    tokens,
                    use_reentrant=False,
                )
            else:
                tokens, frame_tokens = self._run_suffix_block(
                    tokens,
                    layout=layout,
                    block_idx=block_idx,
                    rope=rope,
                )
            self._cache_layer_output(outputs, block_idx=block_idx, frame_tokens=frame_tokens, tokens=tokens, layout=layout)
        final_public_tokens = layout.internal_to_public(tokens)
        final_hand_prompts, final_metric_prompts = layout.split_internal_prompt_types(tokens)
        return VggtSuffixOutput(
            aggregated_tokens_list=outputs,
            final_public_tokens=final_public_tokens,
            final_internal_tokens=tokens,
            final_hand_prompts=final_hand_prompts,
            final_metric_prompts=final_metric_prompts,
        )

    def decode_camera_pose_encoding(
        self,
        aggregated_tokens_list: list[torch.Tensor | None],
        *,
        patch_token_start: int,
    ) -> torch.Tensor:
        device_type = self._cached_tokens_device_type(aggregated_tokens_list)
        with torch.autocast(device_type=device_type, enabled=False):
            return self.camera_head(aggregated_tokens_list, patch_token_start=patch_token_start)

    def decode_camera_from_pose_encoding(
        self,
        pose_encoding: torch.Tensor,
        *,
        image_size_hw: tuple[int, int],
    ) -> torch.Tensor:
        ensure_vendor_paths()
        from vggt_omega.utils.pose_enc import encoding_to_camera

        extrinsics, _ = encoding_to_camera(pose_encoding, image_size_hw, build_intrinsics=True)
        batch_size, num_frames = pose_encoding.shape[:2]
        camera_pose = torch.zeros(batch_size, num_frames, 4, 4, device=pose_encoding.device, dtype=pose_encoding.dtype)
        camera_pose[..., :3, :4] = extrinsics
        camera_pose[..., 3, 3] = 1.0
        return camera_pose

    def decode_camera(
        self,
        aggregated_tokens_list: list[torch.Tensor | None],
        *,
        patch_token_start: int,
        image_size_hw: tuple[int, int],
    ) -> torch.Tensor:
        pose_encoding = self.decode_camera_pose_encoding(
            aggregated_tokens_list,
            patch_token_start=patch_token_start,
        )
        return self.decode_camera_from_pose_encoding(pose_encoding, image_size_hw=image_size_hw)

    def build_intrinsics(
        self,
        *,
        batch_size: int,
        num_frames: int,
        image_size_hw: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
        pose_encoding: torch.Tensor | None = None,
        aggregated_tokens_list: list[torch.Tensor | None] | None = None,
        patch_token_start: int | None = None,
    ) -> torch.Tensor:
        ensure_vendor_paths()
        from vggt_omega.utils.pose_enc import encoding_to_camera

        if pose_encoding is None:
            if aggregated_tokens_list is None or patch_token_start is None:
                raise ValueError("构造 intrinsics 时需要 pose_encoding 或 aggregated_tokens_list + patch_token_start。")
            pose_encoding = self.decode_camera_pose_encoding(
                aggregated_tokens_list,
                patch_token_start=patch_token_start,
            )
        _, intrinsics = encoding_to_camera(pose_encoding, image_size_hw, build_intrinsics=True)
        return intrinsics.to(device=device, dtype=dtype)

    def decode_depth_with_conf(
        self,
        aggregated_tokens_list: list[torch.Tensor | None],
        *,
        images: torch.Tensor,
        patch_token_start: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type=images.device.type, enabled=False):
            depth, depth_conf = self.dense_head(aggregated_tokens_list, images=images, patch_token_start=patch_token_start)
        if depth.ndim == 5 and depth.shape[-1] == 1:
            depth = depth.squeeze(-1)
        if depth_conf.ndim == 5 and depth_conf.shape[-1] == 1:
            depth_conf = depth_conf.squeeze(-1)
        return depth, depth_conf

    def decode_depth(
        self,
        aggregated_tokens_list: list[torch.Tensor | None],
        *,
        images: torch.Tensor,
        patch_token_start: int,
    ) -> torch.Tensor:
        depth, _ = self.decode_depth_with_conf(
            aggregated_tokens_list,
            images=images,
            patch_token_start=patch_token_start,
        )
        return depth

    def forward(self, images: torch.Tensor):
        from egohandmetric_prompt.models.marker_model import MarkerBackboneFeatures

        prefix = self.run_prefix(images, inject_layer_idx=self.mid_layer_index)
        pose_encoding = self.decode_camera_pose_encoding(
            prefix.aggregated_tokens_list,
            patch_token_start=prefix.patch_token_start,
        )
        camera_pose = self.decode_camera_from_pose_encoding(
            pose_encoding,
            image_size_hw=images.shape[-2:],
        ).to(device=images.device, dtype=images.dtype)
        intrinsics = self.build_intrinsics(
            batch_size=images.shape[0],
            num_frames=images.shape[1],
            image_size_hw=images.shape[-2:],
            device=images.device,
            dtype=images.dtype,
            pose_encoding=pose_encoding,
        )
        depth, depth_conf = self.decode_depth_with_conf(
            prefix.aggregated_tokens_list,
            images=images,
            patch_token_start=prefix.patch_token_start,
        )
        depth = depth.to(device=images.device, dtype=images.dtype)
        depth_conf = depth_conf.to(device=images.device, dtype=images.dtype)
        return MarkerBackboneFeatures(
            dino_feature_map=prefix.dino_feature_map.to(device=images.device, dtype=images.dtype),
            dino_tokens=prefix.dino_tokens.to(device=images.device, dtype=images.dtype),
            mid_tokens=prefix.mid_tokens.to(device=images.device, dtype=images.dtype),
            camera_pose=camera_pose,
            depth=depth,
            intrinsics=intrinsics,
            camera_pose_encoding=pose_encoding.to(device=images.device, dtype=images.dtype),
            depth_conf=depth_conf,
        )
