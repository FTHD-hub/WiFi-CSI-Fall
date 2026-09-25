"""Model variants for CSI-based fall detection experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def normalize_model_variant(name: str | None) -> str:
    if name is None:
        return "baseline"

    normalized = name.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    aliases = {
        "baseline": "baseline",
        "b0": "baseline",
        "enet": "baseline",
        "efficientnet": "baseline",
        "efficientnetb0": "baseline",
        "b0dwt": "b0-dwt",
        "b0dwtsoft": "b0-dwt-soft",
        "b0dwtmotion": "b0-dwt-motion",
        "b0dwtaug": "b0-dwt-aug",
        "dwttransformersoft": "dwt-transformer-soft",
        "dwttransformermotion": "dwt-transformer-motion",
        "dwttransformermotionaug": "dwt-transformer-motion-aug",
        "dwttransformer": "wavelet-fpnet",
        "wavelettransformer": "wavelet-fpnet",
        "waveletfpnet": "wavelet-fpnet",
        "fpnetwavelet": "wavelet-fpnet",
        "innovative": "wavelet-fpnet",
        "innovation": "wavelet-fpnet",
        "wavelet": "wavelet-fpnet",
        "waveletfpnetlegacy": "wavelet-fpnet-legacy",
        "legacywaveletfpnet": "wavelet-fpnet-legacy",
        "waveletfpnetold": "wavelet-fpnet-legacy",
        "differentialtcn": "differential-tcn",
        "dtcn": "differential-tcn",
        "difftcn": "differential-tcn",
        "motionefficientnetd1": "motion-efficientnet-d1",
        "motionenetd1": "motion-efficientnet-d1",
        "diffenetd1": "motion-efficientnet-d1",
        "motionefficientnetd12": "motion-efficientnet-d12",
        "motionenetd12": "motion-efficientnet-d12",
        "diffenetd12": "motion-efficientnet-d12",
        "motionefficientnet": "motion-efficientnet",
        "motionenet": "motion-efficientnet",
        "differentialefficientnet": "motion-efficientnet",
        "diffenet": "motion-efficientnet",
        "motionefficientnetfull": "motion-efficientnet",
        "motionenetfull": "motion-efficientnet",
        "diffenetfull": "motion-efficientnet",
    }
    if normalized in aliases:
        return aliases[normalized]
    raise ValueError(
        f'Unknown model variant "{name}". Valid values: baseline, b0-dwt, '
        'b0-dwt-soft, b0-dwt-motion, b0-dwt-aug, dwt-transformer-soft, '
        'dwt-transformer-motion, dwt-transformer-motion-aug, wavelet-fpnet, '
        'wavelet-fpnet-legacy, differential-tcn, motion-efficientnet-d1, '
        'motion-efficientnet-d12, motion-efficientnet'
    )


ABLATION_VARIANTS = {
    "A": "baseline",
    "B": "b0-dwt",
    "C": "b0-dwt-soft",
    "D": "b0-dwt-motion",
    "E": "b0-dwt-aug",
    "F": "dwt-transformer-soft",
    "G": "dwt-transformer-motion",
    "H": "dwt-transformer-motion-aug",
}


def normalize_ablation_name(name: str) -> str:
    normalized = str(name).strip().upper()
    if normalized not in ABLATION_VARIANTS:
        valid = ", ".join(ABLATION_VARIANTS)
        raise ValueError(f'Unknown ablation "{name}". Valid values: {valid}')
    return normalized


def resolve_ablation_variant(name: str) -> str:
    return ABLATION_VARIANTS[normalize_ablation_name(name)]


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must contain a state dictionary")

    for key in ("state_dict", "model_state_dict"):
        if key in checkpoint:
            checkpoint = checkpoint[key]
            break

    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint does not contain a valid state dictionary")

    return {
        key.removeprefix("module."): value
        for key, value in checkpoint.items()
    }


def infer_model_variant_from_state_dict(state_dict: dict[str, torch.Tensor]) -> str:
    if any(key.startswith("motion_enhancer.") for key in state_dict):
        return "motion-efficientnet"
    if any(key.startswith("dtcn_blocks.") or key.startswith("input_projection.") for key in state_dict):
        return "differential-tcn"
    if any(
        key.startswith(prefix)
        for prefix in ("dwt_frontend.", "transformer_encoder.", "positional_embedding.")
        for key in state_dict
    ):
        if any(key.startswith("motion_gate.") or key.startswith("fpn.") for key in state_dict):
            return "wavelet-fpnet-legacy"
        return "wavelet-fpnet"
    if any(
        key.startswith(prefix)
        for prefix in ("wavelet.", "motion_gate.", "fpn.", "stem.", "pyramid.", "fusion.")
        for key in state_dict
    ):
        return "wavelet-fpnet-legacy"
    return "baseline"


def build_checkpoint_payload(
    model: nn.Module,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"state_dict": model.state_dict()}
    model_variant = getattr(model, "model_variant", None)
    if model_variant is not None:
        payload["model_variant"] = model_variant
    if hasattr(model, "get_config"):
        config = model.get_config()
        if config:
            payload["model_config"] = config
    if extra:
        payload.update(extra)
    return payload


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = ConvBNAct(in_channels, out_channels, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.act = nn.SiLU()
        self.skip = None
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs if self.skip is None else self.skip(inputs)
        output = self.conv1(inputs)
        output = self.conv2(output)
        return self.act(output + residual)


class HaarWavelet2D(nn.Module):
    def __init__(self, channels: int = 3) -> None:
        super().__init__()
        self.channels = channels

        low = torch.tensor([1.0, 1.0], dtype=torch.float32) / math.sqrt(2.0)
        high = torch.tensor([1.0, -1.0], dtype=torch.float32) / math.sqrt(2.0)
        filters = torch.stack(
            [
                torch.outer(low, low),
                torch.outer(low, high),
                torch.outer(high, low),
                torch.outer(high, high),
            ],
            dim=0,
        ).unsqueeze(1)
        filters = filters.repeat(channels, 1, 1, 1)
        self.register_buffer("filters", filters)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if inputs.ndim != 4:
            raise ValueError("expected 4D tensor shaped [B, C, H, W]")
        if inputs.shape[1] != self.channels:
            raise ValueError(f"expected {self.channels} input channels, got {inputs.shape[1]}")

        pad_h = inputs.shape[-2] % 2
        pad_w = inputs.shape[-1] % 2
        if pad_h or pad_w:
            inputs = F.pad(inputs, (0, pad_w, 0, pad_h), mode="replicate")

        output = F.conv2d(inputs, self.filters, stride=2, groups=self.channels)
        batch_size, _, height, width = output.shape
        output = output.view(batch_size, self.channels, 4, height, width)
        ll = output[:, :, 0]
        lh = output[:, :, 1]
        hl = output[:, :, 2]
        hh = output[:, :, 3]
        return ll, lh, hl, hh


class StaticMotionEnhancement(nn.Module):
    def __init__(self, channels: int = 12, hidden_channels: int = 8) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(2, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.static_strength = nn.Parameter(torch.tensor(0.5))
        self.motion_strength = nn.Parameter(torch.tensor(1.0))

    def forward(self, ll: torch.Tensor, details: torch.Tensor) -> torch.Tensor:
        motion_energy = details.abs().mean(dim=1, keepdim=True)
        static_energy = ll.abs().mean(dim=1, keepdim=True)
        gate = self.gate(torch.cat([motion_energy, static_energy], dim=1))
        weakened_static = ll * (1.0 - torch.sigmoid(self.static_strength) * gate)
        enhanced_motion = details * (1.0 + torch.sigmoid(self.motion_strength) * gate)
        return torch.cat([weakened_static, enhanced_motion], dim=1)


class EfficientNetB0Classifier(nn.Module):
    model_variant = "baseline"

    def __init__(self, num_classes: int = 2, pretrained_backbone: bool = True) -> None:
        super().__init__()
        self.pretrained_backbone = pretrained_backbone
        weights = (
            torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
            if pretrained_backbone
            else None
        )
        self.backbone = torchvision.models.efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone(inputs)

    def get_config(self) -> dict[str, Any]:
        return {
            "backbone": "efficientnet_b0",
            "pretrained_backbone": self.pretrained_backbone,
            "classifier_head": "mlp2",
        }


class DifferentialMotionEnhancer(nn.Module):
    def __init__(
        self,
        channels: int = 3,
        *,
        use_velocity: bool = True,
        use_acceleration: bool = True,
        dynamic_gate: bool = True,
    ) -> None:
        super().__init__()
        self.use_velocity = use_velocity
        self.use_acceleration = use_acceleration
        self.dynamic_gate = dynamic_gate
        self.velocity_filter = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.acceleration_filter = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.velocity_gate_raw = nn.Parameter(torch.tensor(-1.2))
        self.acceleration_gate_raw = nn.Parameter(torch.tensor(-1.6))
        self.velocity_gain = nn.Parameter(torch.tensor(0.35))
        self.acceleration_gain = nn.Parameter(torch.tensor(0.55))
        self.acceleration_residual_raw = nn.Parameter(torch.tensor(-2.0))

    @staticmethod
    def _difference(inputs: torch.Tensor) -> torch.Tensor:
        diff = inputs[:, :, 1:, :] - inputs[:, :, :-1, :]
        return F.pad(diff, (0, 0, 1, 0))

    def _dynamic_gates(
        self,
        velocity: torch.Tensor,
        acceleration: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.dynamic_gate:
            return torch.sigmoid(self.velocity_gate_raw), torch.sigmoid(self.acceleration_gate_raw)
        velocity_energy = torch.log1p(velocity.pow(2).mean(dim=(1, 2, 3), keepdim=True))
        acceleration_energy = torch.log1p(acceleration.pow(2).mean(dim=(1, 2, 3), keepdim=True))
        velocity_gate = torch.sigmoid(self.velocity_gate_raw + self.velocity_gain * velocity_energy)
        acceleration_gate = torch.sigmoid(
            self.acceleration_gate_raw + self.acceleration_gain * acceleration_energy
        )
        return velocity_gate, acceleration_gate

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        velocity = self._difference(inputs)
        acceleration = self._difference(velocity)
        velocity_gate, acceleration_gate = self._dynamic_gates(velocity, acceleration)
        enhanced = inputs
        if self.use_velocity:
            enhanced = enhanced + velocity_gate * self.velocity_filter(velocity)
        if self.use_acceleration:
            if self.dynamic_gate:
                acceleration_scale = torch.sigmoid(self.acceleration_residual_raw)
            else:
                acceleration_scale = 1.0
            enhanced = enhanced + acceleration_scale * acceleration_gate * self.acceleration_filter(acceleration)
        return enhanced


class MotionEfficientNetB0Classifier(nn.Module):
    model_variant = "motion-efficientnet"

    def __init__(
        self,
        num_classes: int = 2,
        pretrained_backbone: bool = True,
        *,
        motion_mode: str = "full",
        model_variant: str = "motion-efficientnet",
    ) -> None:
        super().__init__()
        self.model_variant = model_variant
        self.pretrained_backbone = pretrained_backbone
        self.motion_mode = motion_mode
        self.motion_enhancer = DifferentialMotionEnhancer(
            channels=3,
            use_velocity=motion_mode in {"d1", "d12", "full"},
            use_acceleration=motion_mode in {"d12", "full"},
            dynamic_gate=motion_mode == "full",
        )
        weights = (
            torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
            if pretrained_backbone
            else None
        )
        self.backbone = torchvision.models.efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.SiLU(),
            nn.Dropout(0.35),
            nn.Linear(256, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.motion_enhancer(inputs))

    def get_config(self) -> dict[str, Any]:
        return {
            "backbone": "efficientnet_b0",
            "pretrained_backbone": self.pretrained_backbone,
            "motion_mode": self.motion_mode,
            "motion_enhancer": {
                "d1": "first_temporal_difference",
                "d12": "first_and_second_temporal_difference_static_gate",
                "full": "first_and_second_temporal_difference_dynamic_gate",
            }[self.motion_mode],
            "dynamic_gate": self.motion_mode == "full",
            "classifier_head": "mlp2",
        }


class DB4WaveletFrontend(nn.Module):
    """Time-axis db4 decomposition with optional soft-threshold denoising."""

    def __init__(
        self,
        channels: int = 3,
        *,
        soft_threshold: bool = False,
        motion_enhancement: bool = False,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.soft_threshold = soft_threshold
        self.motion_enhancement = motion_enhancement

        dec_lo = torch.tensor(
            [
                -0.010597401785069032,
                0.032883011666982945,
                0.030841381835560764,
                -0.18703481171888114,
                -0.02798376941698385,
                0.6308807679295904,
                0.7148465705529154,
                0.2303778133088964,
            ],
            dtype=torch.float32,
        ).view(1, 1, -1)
        dec_hi = torch.tensor(
            [
                -0.2303778133088964,
                0.7148465705529154,
                -0.6308807679295904,
                -0.02798376941698385,
                0.18703481171888114,
                0.030841381835560764,
                -0.032883011666982945,
                -0.010597401785069032,
            ],
            dtype=torch.float32,
        ).view(1, 1, -1)
        self.register_buffer("dec_lo", dec_lo)
        self.register_buffer("dec_hi", dec_hi)

        if soft_threshold:
            self.threshold_raw = nn.Parameter(torch.tensor([-3.0, -3.0, -3.0]))

        if motion_enhancement:
            self.motion_gate = nn.Sequential(
                nn.Conv2d(2, 8, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(8),
                nn.SiLU(),
                nn.Conv2d(8, 1, kernel_size=1),
                nn.Sigmoid(),
            )
            self.static_strength = nn.Parameter(torch.tensor(-1.5))
            self.motion_strength = nn.Parameter(torch.tensor(-1.5))

        self.projection = nn.Conv2d(channels * 5, channels, kernel_size=1, bias=False)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    @staticmethod
    def _soft_threshold(values: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        return values.sign() * F.relu(values.abs() - threshold)

    def _analysis(self, inputs: torch.Tensor, filter_bank: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = inputs.shape
        sequence = inputs.permute(0, 1, 3, 2).reshape(-1, 1, height)
        sequence = F.pad(sequence, (3, 4), mode="reflect")
        coefficients = F.conv1d(sequence, filter_bank, stride=2)
        new_height = coefficients.shape[-1]
        return coefficients.reshape(batch_size, channels, width, new_height).permute(0, 1, 3, 2)

    @staticmethod
    def _resize(values: torch.Tensor, target_height: int) -> torch.Tensor:
        return F.interpolate(
            values,
            size=(target_height, values.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.channels:
            raise ValueError(
                f"expected [B, {self.channels}, H, W] input, got {tuple(inputs.shape)}"
            )

        low1 = self._analysis(inputs, self.dec_lo)
        detail1 = self._analysis(inputs, self.dec_hi)
        low2 = self._analysis(low1, self.dec_lo)
        detail2 = self._analysis(low1, self.dec_hi)
        low3 = self._analysis(low2, self.dec_lo)
        detail3 = self._analysis(low2, self.dec_hi)

        target_height = inputs.shape[-2]
        low = self._resize(low3, target_height)
        details = [
            self._resize(detail1, target_height),
            self._resize(detail2, target_height),
            self._resize(detail3, target_height),
        ]

        if self.soft_threshold:
            thresholds = F.softplus(self.threshold_raw)
            details = [
                self._soft_threshold(detail, threshold)
                for detail, threshold in zip(details, thresholds)
            ]

        if self.motion_enhancement:
            detail_energy = torch.stack(
                [detail.abs().mean(dim=1, keepdim=True) for detail in details],
                dim=0,
            ).mean(dim=0)
            static_energy = low.abs().mean(dim=1, keepdim=True)
            gate = self.motion_gate(torch.cat([detail_energy, static_energy], dim=1))
            low = low * (
                1.0 - torch.sigmoid(self.static_strength) * gate
            )
            details = [
                detail * (1.0 + torch.sigmoid(self.motion_strength) * gate)
                for detail in details
            ]

        features = torch.cat([inputs, low, *details], dim=1)
        projected = self.projection(features)
        return inputs + torch.tanh(self.residual_scale) * projected


class DWTB0Classifier(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int = 2,
        pretrained_backbone: bool = True,
        soft_threshold: bool = False,
        motion_enhancement: bool = False,
        model_variant: str = "b0-dwt",
    ) -> None:
        super().__init__()
        self.model_variant = model_variant
        self.pretrained_backbone = pretrained_backbone
        self.soft_threshold = soft_threshold
        self.motion_enhancement = motion_enhancement
        self.dwt_frontend = DB4WaveletFrontend(
            channels=3,
            soft_threshold=soft_threshold,
            motion_enhancement=motion_enhancement,
        )
        weights = (
            torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
            if pretrained_backbone
            else None
        )
        self.backbone = torchvision.models.efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.dwt_frontend(inputs))

    def get_config(self) -> dict[str, Any]:
        return {
            "backbone": "efficientnet_b0",
            "pretrained_backbone": self.pretrained_backbone,
            "wavelet": "db4",
            "wavelet_level": 3,
            "soft_threshold": self.soft_threshold,
            "motion_enhancement": self.motion_enhancement,
            "classifier_head": "mlp2",
        }


class DWTTransformerClassifier(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int = 2,
        pretrained_backbone: bool = True,
        motion_enhancement: bool = False,
        model_variant: str = "dwt-transformer-soft",
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.model_variant = model_variant
        self.pretrained_backbone = pretrained_backbone
        self.motion_enhancement = motion_enhancement
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dwt_frontend = DB4WaveletFrontend(
            channels=3,
            soft_threshold=True,
            motion_enhancement=motion_enhancement,
        )
        weights = (
            torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
            if pretrained_backbone
            else None
        )
        backbone = torchvision.models.efficientnet_b0(weights=weights)
        self.feature_extractor = backbone.features
        self.input_proj = nn.Conv2d(1280, d_model, kernel_size=1)
        self.positional_embedding = nn.Parameter(torch.zeros(1, 64, d_model))
        nn.init.normal_(self.positional_embedding, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.3,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(self.dwt_frontend(inputs))
        features = self.input_proj(features).mean(dim=3).permute(0, 2, 1)
        sequence_length = features.shape[1]
        features = features + self.positional_embedding[:, :sequence_length]
        features = self.transformer_encoder(features)
        return self.classifier(features.mean(dim=1))

    def get_config(self) -> dict[str, Any]:
        return {
            "backbone": "efficientnet_b0",
            "pretrained_backbone": self.pretrained_backbone,
            "wavelet": "db4",
            "wavelet_level": 3,
            "soft_threshold": True,
            "motion_enhancement": self.motion_enhancement,
            "transformer_d_model": self.d_model,
            "transformer_heads": self.nhead,
            "transformer_layers": self.num_layers,
            "classifier_head": "mlp2",
        }


class WaveletFPNetClassifier(nn.Module):
    model_variant = "wavelet-fpnet-legacy"

    def __init__(
        self,
        num_classes: int = 2,
        base_channels: int = 48,
        pyramid_channels: int = 160,
        wavelet_channels: int = 3,
    ) -> None:
        super().__init__()
        self.wavelet_channels = wavelet_channels
        self.base_channels = base_channels
        self.pyramid_channels = pyramid_channels

        self.wavelet = HaarWavelet2D(channels=wavelet_channels)
        self.motion_enhance = StaticMotionEnhancement(channels=wavelet_channels * 4)

        self.stem = nn.Sequential(
            ConvBNAct(wavelet_channels * 4, base_channels),
            ResidualBlock(base_channels, base_channels),
        )
        self.stage2 = ResidualBlock(base_channels, base_channels * 2, stride=2)
        self.stage3 = ResidualBlock(base_channels * 2, base_channels * 4, stride=2)
        self.stage4 = ResidualBlock(base_channels * 4, base_channels * 8, stride=2)

        self.lateral1 = nn.Conv2d(base_channels, pyramid_channels, kernel_size=1, bias=False)
        self.lateral2 = nn.Conv2d(base_channels * 2, pyramid_channels, kernel_size=1, bias=False)
        self.lateral3 = nn.Conv2d(base_channels * 4, pyramid_channels, kernel_size=1, bias=False)
        self.lateral4 = nn.Conv2d(base_channels * 8, pyramid_channels, kernel_size=1, bias=False)

        self.smooth1 = ConvBNAct(pyramid_channels, pyramid_channels)
        self.smooth2 = ConvBNAct(pyramid_channels, pyramid_channels)
        self.smooth3 = ConvBNAct(pyramid_channels, pyramid_channels)
        self.smooth4 = ConvBNAct(pyramid_channels, pyramid_channels)

        fusion_width = pyramid_channels * 4
        self.fusion = nn.Sequential(
            nn.Linear(fusion_width, 256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.SiLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        ll, lh, hl, hh = self.wavelet(inputs)
        details = torch.cat([lh, hl, hh], dim=1)
        features = self.motion_enhance(ll, details)

        c1 = self.stem(features)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)

        p4 = self.lateral4(c4)
        p3 = self.lateral3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.lateral2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        p1 = self.lateral1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode="nearest")

        p1 = self.smooth1(p1)
        p2 = self.smooth2(p2)
        p3 = self.smooth3(p3)
        p4 = self.smooth4(p4)

        pooled = [
            F.adaptive_avg_pool2d(level, output_size=1).flatten(1)
            for level in (p1, p2, p3, p4)
        ]
        fused = torch.cat(pooled, dim=1)
        fused = self.fusion(fused)
        return self.classifier(fused)

    def get_config(self) -> dict[str, Any]:
        return {
            "base_channels": self.base_channels,
            "pyramid_channels": self.pyramid_channels,
            "wavelet_channels": self.wavelet_channels,
        }


class MultiScaleCausalConv1d(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        kernel_sizes: tuple[int, ...],
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.dilation = dilation
        self.kernel_sizes = kernel_sizes
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    groups=channels,
                    bias=False,
                )
                for kernel_size in kernel_sizes
            ]
        )
        self.mix = nn.Sequential(
            nn.Conv1d(channels * len(kernel_sizes), channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = []
        for kernel_size, branch in zip(self.kernel_sizes, self.branches):
            left_padding = (kernel_size - 1) * self.dilation
            outputs.append(branch(F.pad(inputs, (left_padding, 0))))
        return self.mix(torch.cat(outputs, dim=1))


class ChannelSqueezeExcite1d(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden_channels = max(channels // reduction, 8)
        self.gate = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weights = self.gate(inputs.mean(dim=2)).unsqueeze(-1)
        return inputs * weights


class TemporalAttentionPooling(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Conv1d(channels, channels // 2, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(channels // 2, 1, kernel_size=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.scorer(inputs), dim=2)
        return torch.sum(inputs * weights, dim=2)


class TemporalRefinementBlock(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.SiLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.act(inputs + self.block(inputs))


class LegacyCausalConv1d(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.block = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(F.pad(inputs, (self.left_padding, 0)))


class DifferentialTCNBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        kernel_sizes: tuple[int, ...] = (3, 5, 7),
        dilation: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.pre_norm = nn.BatchNorm1d(channels)
        self.diff_norm = nn.BatchNorm1d(channels)
        self.acc_norm = nn.BatchNorm1d(channels)
        self.main_branch = MultiScaleCausalConv1d(
            channels,
            kernel_sizes=kernel_sizes,
            dilation=dilation,
            dropout=dropout,
        )
        self.diff_branch = MultiScaleCausalConv1d(
            channels,
            kernel_sizes=kernel_sizes,
            dilation=dilation,
            dropout=dropout,
        )
        self.acc_branch = MultiScaleCausalConv1d(
            channels,
            kernel_sizes=kernel_sizes,
            dilation=dilation,
            dropout=dropout,
        )
        self.channel_attention = ChannelSqueezeExcite1d(channels)
        self.gate1_raw = nn.Parameter(torch.tensor(-0.1))
        self.gate2_raw = nn.Parameter(torch.tensor(-0.3))
        self.gate1_variance_gain = nn.Parameter(torch.tensor(0.35))
        self.gate2_variance_gain = nn.Parameter(torch.tensor(0.55))
        self.residual_scale = nn.Parameter(torch.tensor(0.5))
        self.output_norm = nn.BatchNorm1d(channels)
        self.output_act = nn.SiLU()

    @staticmethod
    def _difference(inputs: torch.Tensor) -> torch.Tensor:
        diff = inputs[:, :, 1:] - inputs[:, :, :-1]
        return F.pad(diff, (1, 0))

    def _dynamic_gates(
        self,
        first_diff: torch.Tensor,
        second_diff: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        velocity_energy = torch.log1p(first_diff.pow(2).mean(dim=(1, 2), keepdim=True))
        acceleration_energy = torch.log1p(second_diff.pow(2).mean(dim=(1, 2), keepdim=True))
        gate1 = torch.sigmoid(self.gate1_raw + self.gate1_variance_gain * velocity_energy)
        gate2 = torch.sigmoid(self.gate2_raw + self.gate2_variance_gain * acceleration_energy)
        return gate1, gate2

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = self.pre_norm(inputs)
        first_diff = self._difference(normalized)
        second_diff = self._difference(first_diff)
        gate1, gate2 = self._dynamic_gates(first_diff, second_diff)
        output = (
            self.main_branch(normalized)
            + gate1 * self.diff_branch(self.diff_norm(first_diff))
            + gate2 * self.acc_branch(self.acc_norm(second_diff))
        )
        output = self.output_norm(self.channel_attention(output))
        return self.output_act(inputs + torch.tanh(self.residual_scale) * output)


class DifferentialTCNClassifier(nn.Module):
    model_variant = "differential-tcn"

    def __init__(
        self,
        *,
        num_classes: int = 2,
        input_channels: int = 90,
        hidden_channels: int = 96,
        kernel_sizes: tuple[int, ...] = (3, 5),
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations
        self.dropout = dropout
        self.input_projection = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.SiLU(),
        )
        self.temporal_downsample = nn.AvgPool1d(kernel_size=2, stride=2)
        self.dtcn_blocks = nn.Sequential(
            *[
                DifferentialTCNBlock(
                    hidden_channels,
                    kernel_sizes=kernel_sizes,
                    dilation=dilation,
                    dropout=dropout,
                )
                for dilation in dilations
            ]
        )
        self.refinement = TemporalRefinementBlock(hidden_channels, dropout)
        self.attention_pool = TemporalAttentionPooling(hidden_channels)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 4, 192),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(192, 64),
            nn.SiLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, num_classes),
        )

    @staticmethod
    def _to_sequence(inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError(f"expected [B, 3, T, 30] input, got {tuple(inputs.shape)}")
        batch_size, antennas, time_steps, subcarriers = inputs.shape
        return inputs.permute(0, 1, 3, 2).reshape(batch_size, antennas * subcarriers, time_steps)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence = self._to_sequence(inputs)
        features = self.input_projection(sequence)
        features = self.temporal_downsample(features)
        features = self.refinement(self.dtcn_blocks(features))
        avg_pool = features.mean(dim=2)
        max_pool = features.amax(dim=2)
        var_pool = features.var(dim=2, unbiased=False)
        attention_pool = self.attention_pool(features)
        return self.classifier(torch.cat([avg_pool, max_pool, var_pool, attention_pool], dim=1))

    def get_config(self) -> dict[str, Any]:
        return {
            "input_channels": self.input_channels,
            "hidden_channels": self.hidden_channels,
            "kernel_sizes": list(self.kernel_sizes),
            "dilations": list(self.dilations),
            "dropout": self.dropout,
            "temporal_downsample": "avg_pool_stride_2",
            "branches": ["main", "first_difference", "second_difference"],
            "dynamic_gate": "learned_scalar_plus_velocity_and_acceleration_energy",
            "attention": ["multi_scale_temporal_conv", "channel_squeeze_excite", "temporal_attention_pooling"],
        }


def build_model(
    model_variant: str | None,
    *,
    num_classes: int = 2,
    pretrained_backbone: bool = True,
    **model_kwargs: Any,
) -> nn.Module:
    variant = normalize_model_variant(model_variant)
    if variant == "baseline":
        return EfficientNetB0Classifier(
            num_classes=num_classes,
            pretrained_backbone=pretrained_backbone,
        )
    if variant == "motion-efficientnet":
        return MotionEfficientNetB0Classifier(
            num_classes=num_classes,
            pretrained_backbone=pretrained_backbone,
            motion_mode="full",
            model_variant="motion-efficientnet",
        )
    if variant == "motion-efficientnet-d1":
        return MotionEfficientNetB0Classifier(
            num_classes=num_classes,
            pretrained_backbone=pretrained_backbone,
            motion_mode="d1",
            model_variant="motion-efficientnet-d1",
        )
    if variant == "motion-efficientnet-d12":
        return MotionEfficientNetB0Classifier(
            num_classes=num_classes,
            pretrained_backbone=pretrained_backbone,
            motion_mode="d12",
            model_variant="motion-efficientnet-d12",
        )
    if variant in {"b0-dwt", "b0-dwt-soft", "b0-dwt-motion", "b0-dwt-aug"}:
        return DWTB0Classifier(
            num_classes=num_classes,
            pretrained_backbone=pretrained_backbone,
            soft_threshold=variant in {"b0-dwt-soft"},
            motion_enhancement=variant in {"b0-dwt-motion"},
            model_variant=variant,
        )
    if variant in {"dwt-transformer-soft", "dwt-transformer-motion", "dwt-transformer-motion-aug"}:
        return DWTTransformerClassifier(
            num_classes=num_classes,
            pretrained_backbone=pretrained_backbone,
            motion_enhancement=variant in {"dwt-transformer-motion", "dwt-transformer-motion-aug"},
            model_variant=variant,
        )
    if variant == "wavelet-fpnet":
        return DWTTransformerClassifier(
            num_classes=num_classes,
            pretrained_backbone=pretrained_backbone,
            motion_enhancement=False,
            model_variant="wavelet-fpnet",
        )
    if variant == "wavelet-fpnet-legacy":
        return WaveletFPNetClassifier(
            num_classes=num_classes,
            base_channels=int(model_kwargs.get("base_channels", 48)),
            pyramid_channels=int(model_kwargs.get("pyramid_channels", 160)),
            wavelet_channels=int(model_kwargs.get("wavelet_channels", 3)),
        )
    if variant == "differential-tcn":
        return DifferentialTCNClassifier(num_classes=num_classes)
    if variant == "wavelet-transformer":
        return DWTTransformerClassifier(
            num_classes=num_classes,
            pretrained_backbone=pretrained_backbone,
            motion_enhancement=False,
            model_variant="wavelet-transformer",
        )
    raise AssertionError(f"Unhandled model variant: {variant}")
