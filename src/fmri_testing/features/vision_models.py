"""Open vision model activation interfaces."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from PIL import Image


@dataclass
class VisionFeatureExtractor:
    """Callable wrapper for layer-wise image activations."""

    model_name: str
    layers: list[str]
    device: str = "cpu"
    pretrained: bool = True

    def __post_init__(self) -> None:
        self.synthetic_dim = 128
        self.model = None
        self.transform: Callable | None = None
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self._acts: dict[str, torch.Tensor] = {}
        if self.model_name == "synthetic":
            return
        self._build_torchvision_model()

    def _build_torchvision_model(self) -> None:
        try:
            import torchvision.models as models
            from torchvision import transforms
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise ImportError("torchvision is required for non-synthetic vision models") from exc
        weights = None
        if self.model_name == "resnet50":
            if self.pretrained:
                weights = models.ResNet50_Weights.DEFAULT
            self.model = models.resnet50(weights=weights).to(self.device).eval()
            modules = dict(self.model.named_modules())
        elif self.model_name in {"vit_b_16", "vit-b-16"}:
            weights = models.ViT_B_16_Weights.DEFAULT if self.pretrained and hasattr(models, "ViT_B_16_Weights") else None
            self.model = models.vit_b_16(weights=weights).to(self.device).eval()
            modules = dict(self.model.named_modules())
        else:
            raise ValueError(f"Unsupported model {self.model_name}; use resnet50, vit_b_16, or synthetic")
        self.transform = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        for layer in self.layers:
            if layer not in modules:
                raise ValueError(f"Layer {layer} not found in {self.model_name}")
            self.handles.append(modules[layer].register_forward_hook(self._hook(layer)))

    def _hook(self, name: str) -> Callable:
        def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
            self._acts[name] = output.detach()
        return hook

    def extract_batch(self, images: list[Image.Image]) -> dict[str, np.ndarray]:
        """Extract configured activations for a batch of PIL images."""
        if self.model_name == "synthetic":
            feats = []
            for img in images:
                arr = np.asarray(img.resize((16, 16)).convert("RGB"), dtype=np.float32) / 255.0
                stats = np.concatenate([arr.mean(axis=(0, 1)), arr.std(axis=(0, 1)), arr.reshape(-1)[: self.synthetic_dim - 6]])
                feats.append(stats[: self.synthetic_dim])
            return {self.layers[0]: np.vstack(feats).astype(np.float32)}
        assert self.model is not None and self.transform is not None
        batch = torch.stack([self.transform(img.convert("RGB")) for img in images]).to(self.device)
        self._acts.clear()
        with torch.no_grad():
            _ = self.model(batch)
        out = {}
        for layer, act in self._acts.items():
            if act.ndim == 4:
                act = torch.nn.functional.adaptive_avg_pool2d(act, 1).flatten(1)
            else:
                act = act.flatten(1)
            out[layer] = act.cpu().numpy().astype(np.float32)
        return out


def optional_model_available(name: str) -> bool:
    """Report whether optional no-login model backends can be imported."""
    if name == "open_clip_vit_b_32":
        try:
            import open_clip  # noqa: F401
            return True
        except Exception:
            return False
    if name == "dinov2":
        return hasattr(torch, "hub")
    return name in {"resnet50", "vit_b_16", "synthetic"}
