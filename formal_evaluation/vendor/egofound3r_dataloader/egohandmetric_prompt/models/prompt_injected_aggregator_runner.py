from __future__ import annotations

from dataclasses import dataclass

import torch

from .token_layout import TokenLayout


@dataclass(slots=True)
class PromptRunOutput:
    public_tokens: torch.Tensor
    internal_tokens: torch.Tensor
    layout: TokenLayout


class PromptInjectedAggregatorRunner:
    @staticmethod
    def append_prompts(
        public_tokens: torch.Tensor,
        hand_prompts: torch.Tensor,
        *,
        patch_token_start: int = 17,
        hand_prompt_count: int | None = None,
        metric_prompt_count: int = 0,
        patch_grid_hw: tuple[int, int] | None = None,
    ) -> PromptRunOutput:
        public_tokens_with_prompts = torch.cat([public_tokens, hand_prompts], dim=2)
        resolved_hand_prompt_count = hand_prompts.shape[2] if hand_prompt_count is None else int(hand_prompt_count)
        layout = TokenLayout(
            camera_token_count=1,
            register_token_count=patch_token_start - 1,
            patch_token_count=public_tokens.shape[2] - patch_token_start,
            hand_prompt_count=resolved_hand_prompt_count,
            metric_prompt_count=int(metric_prompt_count),
            patch_grid_hw=patch_grid_hw,
        )
        internal_tokens = layout.public_to_internal(public_tokens_with_prompts)
        return PromptRunOutput(
            public_tokens=public_tokens_with_prompts,
            internal_tokens=internal_tokens,
            layout=layout,
        )

    @staticmethod
    def split_register_attention_inputs(
        internal_tokens: torch.Tensor,
        layout: TokenLayout,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        camera_register, hand_prompts, patch_tokens = layout.split_internal(internal_tokens)
        cross_frame_tokens = torch.cat([camera_register, hand_prompts], dim=2)
        return cross_frame_tokens, patch_tokens

    @staticmethod
    def extract_final_prompts(
        internal_tokens: torch.Tensor,
        layout: TokenLayout,
    ) -> torch.Tensor:
        _, hand_prompts, _ = layout.split_internal(internal_tokens)
        return hand_prompts

    @staticmethod
    def extract_final_prompt_types(
        internal_tokens: torch.Tensor,
        layout: TokenLayout,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return layout.split_internal_prompt_types(internal_tokens)
