import logging
import math
from typing import Iterable, List, Optional, Sequence, Tuple

import timm

import torch
import torch.nn as nn
import torch.nn.functional as F

from decoder import Config, Decoder


LOGGER = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """Linear layer augmented with Low-Rank Adaptation (LoRA)."""

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        train_bias: bool = False,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank = rank
        self.scaling = alpha / rank

        # Freeze original weight/bias by re-registering them as non-trainable parameters.
        self.weight = nn.Parameter(base_linear.weight.data.clone())
        self.weight.requires_grad = False

        if base_linear.bias is not None:
            self.bias = nn.Parameter(base_linear.bias.data.clone())
            self.bias.requires_grad = train_bias
        else:
            self.register_parameter("bias", None)

        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora_input = self.dropout(x)
        lora = F.linear(lora_input, self.lora_A, bias=None)
        lora = F.linear(lora, self.lora_B, bias=None)
        return base + lora * self.scaling

    @property
    def lora_parameters(self) -> Iterable[nn.Parameter]:
        yield self.lora_A
        yield self.lora_B
        if self.bias is not None and self.bias.requires_grad:
            yield self.bias


class ViLModel(nn.Module):
    """Vision-Language model for image captioning with LoRA adaptation."""

    def __init__(
        self,
        decoder_ckpt: Optional[str] = "hw3_data/p2_data/decoder_model.bin",
        vision_encoder_name: str = "vit_large_patch14_clip_224.openai",
        projector_out_dim: Optional[int] = None,
        freeze_vision: bool = True,
        freeze_decoder: bool = True,
        pad_token_id: int = 151643,
        bos_token_id: int = 151644,
        eos_token_id: int = 151645,
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        lora_targets: Sequence[str] = ("q_proj", "v_proj"),
        lora_train_bias: bool = False,
    ):
        super().__init__()

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.vision_encoder_name = vision_encoder_name

        # Vision encoder backbone from timm.
        self.vision_encoder = timm.create_model(
            vision_encoder_name,
            pretrained=True,
            num_classes=0,
            global_pool="",
        )
        self.vision_width = getattr(self.vision_encoder, "embed_dim")

        if freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            self.vision_encoder.eval()

        # Text decoder (Qwen3) provided in repo.
        self.decoder_config = Config()
        self.decoder = Decoder(self.decoder_config)

        # Decide projection output dim to match decoder hidden size by default.
        if projector_out_dim is None:
            projector_out_dim = self.decoder_config.hidden_size
        self.projector = nn.Linear(self.vision_width, projector_out_dim)

        if projector_out_dim != self.decoder_config.hidden_size:
            raise ValueError(
                f"Projector output dim ({projector_out_dim}) must match decoder hidden size "
                f"({self.decoder_config.hidden_size})."
            )

        # Load decoder checkpoint prior to LoRA injection.
        if decoder_ckpt:
            self.load_decoder_checkpoint(decoder_ckpt)

        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_targets = tuple(lora_targets)
        self.lora_train_bias = lora_train_bias
        self.lora_layers: List[LoRALinear] = []

        if self.lora_rank > 0:
            self.apply_lora_to_decoder()

        if freeze_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False
        # LoRA parameters stay trainable even when decoder is frozen.
        for module in self.lora_layers:
            for param in module.lora_parameters:
                param.requires_grad = True
        # Ensure projector always trainable.
        for param in self.projector.parameters():
            param.requires_grad = True

    def load_decoder_checkpoint(self, ckpt_path: str) -> None:
        try:
            state_dict = torch.load(ckpt_path, map_location="cpu")
            missing, unexpected = self.decoder.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                LOGGER.warning(
                    "Decoder checkpoint loaded with missing=%s, unexpected=%s",
                    missing,
                    unexpected,
                )
        except FileNotFoundError:
            LOGGER.warning("Decoder checkpoint %s not found. Proceeding without weights.", ckpt_path)

    def apply_lora_to_decoder(self) -> None:
        """Inject LoRA adapters into targeted attention projection layers."""
        for layer in self.decoder.layers:
            for target in self.lora_targets:
                module = getattr(layer.self_attn, target, None)
                if isinstance(module, nn.Linear):
                    lora_linear = LoRALinear(
                        module,
                        rank=self.lora_rank,
                        alpha=self.lora_alpha,
                        dropout=self.lora_dropout,
                        train_bias=self.lora_train_bias,
                    )
                    setattr(layer.self_attn, target, lora_linear)
                    self.lora_layers.append(lora_linear)

    def trainable_named_parameters(self) -> Iterable[Tuple[str, nn.Parameter]]:
        for name, param in self.named_parameters():
            if param.requires_grad:
                yield name, param

    def get_trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return {name: param.detach().cpu().clone() for name, param in self.trainable_named_parameters()}

    def load_trainable_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        missing = []
        param_dict = dict(self.named_parameters())
        for name, tensor in state_dict.items():
            if name not in param_dict:
                missing.append(name)
                continue
            param = param_dict[name]
            param.data.copy_(tensor.to(param.device, dtype=param.dtype))
        if missing:
            LOGGER.warning("Trainable state contains unknown parameters: %s", missing)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images into projected vision-language embeddings."""
        features = self.vision_encoder.forward_features(images)

        if isinstance(features, (tuple, list)):
            features = features[0]

        if features.dim() == 2:
            features = features.unsqueeze(1)
        elif features.dim() != 3:
            raise ValueError(f"Unexpected vision encoder output shape: {features.shape}")

        projected = self.projector(features)
        return projected

    def forward(
        self,
        images: torch.Tensor,
        caption_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning loss and logits for caption tokens.
        """
        vision_embeds = self.encode_image(images)

        text_embeds = self.decoder.embed_tokens(caption_ids)
        inputs_embeds = torch.cat([vision_embeds, text_embeds[:, :-1, :]], dim=1)

        logits = self.decoder(inputs_embeds=inputs_embeds)
        prefix_len = vision_embeds.size(1)
        text_logits = logits[:, prefix_len:, :]

        loss = F.cross_entropy(
            text_logits.reshape(-1, text_logits.size(-1)),
            caption_ids[:, 1:].reshape(-1),
            ignore_index=self.pad_token_id,
        )
        return loss, text_logits

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Greedy decode captions for a batch of images."""
        self.eval()
        device = images.device
        vision_embeds = self.encode_image(images)

        generated = torch.full(
            (images.size(0), 1), self.bos_token_id, dtype=torch.long, device=device
        )
        finished = torch.zeros(images.size(0), dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            text_embeds = self.decoder.embed_tokens(generated)
            inputs_embeds = torch.cat([vision_embeds, text_embeds], dim=1)
            logits = self.decoder(inputs_embeds=inputs_embeds)
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)
            next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_tokens], dim=1)
            finished |= next_tokens.squeeze(1).eq(self.eos_token_id)
            if finished.all():
                break

        return generated
