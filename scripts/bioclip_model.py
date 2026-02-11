# bioclip_model.py

import torch
import torch.nn as nn
import open_clip


class BioCLIPClassifier(nn.Module):
    """
    BioCLIP backbone + linear head classifier
    - supports freeze/unfreeze backbone
    - includes CLIP-style logit_scale for normalized features
    """

    def __init__(
        self,
        model_name: str = "hf-hub:imageomics/bioclip",
        num_classes: int = 10,
        freeze_backbone: bool = False,
        use_logit_scale: bool = True,
        logit_scale_init: float = 1 / 0.07,   # CLIP default temperature ~0.07
        max_logit_scale: float = 100.0,
    ):
        super().__init__()
        print(f"Loading BioCLIP backbone: {model_name}")
        self.backbone, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(model_name)

        # Determine embedding dim robustly
        embed_dim = None
        # common: model.visual.output_dim
        if hasattr(self.backbone, "visual") and hasattr(self.backbone.visual, "output_dim"):
            embed_dim = int(self.backbone.visual.output_dim)
        # fallback: dummy forward
        if embed_dim is None:
            with torch.no_grad():
                dummy = torch.randn(1, 3, 224, 224)
                feats = self.backbone.encode_image(dummy)
                embed_dim = int(feats.shape[-1])

        print(f"Detected embedding dimension: {embed_dim}")

        self.classifier = nn.Linear(embed_dim, num_classes)

        self.use_logit_scale = use_logit_scale
        self.max_logit_scale = float(max_logit_scale)

        if self.use_logit_scale:
            # logit_scale is stored in log-space in OpenAI CLIP
            init = torch.log(torch.tensor(float(logit_scale_init)))
            self.logit_scale = nn.Parameter(init.clone().detach())
        else:
            self.register_parameter("logit_scale", None)

        # Freeze backbone if requested
        self.set_backbone_trainable(not freeze_backbone)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = bool(trainable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.encode_image(x)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        logits = self.classifier(feats)

        if self.use_logit_scale and self.logit_scale is not None:
            scale = self.logit_scale.exp().clamp(max=self.max_logit_scale)
            logits = logits * scale

        return logits

    def get_transforms(self):
        return self.preprocess_train, self.preprocess_val
