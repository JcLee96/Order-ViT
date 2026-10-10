"""Order-ViT: a timm backbone with a categorical head and a pairwise order head."""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

GREATER, EQUAL, LESS = 0, 1, 2


def mlp_head(in_features, out_features):
    """Three-layer MLP head used for both the class and the order classifier."""
    return nn.Sequential(
        nn.Linear(in_features, 512),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(512, 256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(256, out_features),
    )


def order_labels(y_a, y_b):
    """Order relation of a to b: 0 if a > b, 1 if a == b, 2 if a < b."""
    return torch.where(y_a > y_b, GREATER, torch.where(y_a == y_b, EQUAL, LESS)).long()


class OrderViT(nn.Module):
    """Backbone features feed a grade classifier and a 3-way (>, =, <) pair comparator."""

    def __init__(self, backbone="vit_base_r50_s16_384", num_classes=4, pretrained=True, **timm_kwargs):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, **timm_kwargs)
        dim = self.backbone.num_features
        self.cls_head = mlp_head(dim, num_classes)
        self.order_head = mlp_head(dim * 2, 3)

    def features(self, x):
        """Pooled pre-logit backbone features."""
        return self.backbone(x.float())

    def compare(self, feat_a, feat_b):
        """Order logits for each (a, b) feature pair."""
        return self.order_head(torch.cat([feat_a, feat_b], dim=1))

    def forward(self, x):
        """Grade logits and features."""
        feats = self.features(x)
        return self.cls_head(feats), feats


def order_vit_loss(model, x, y, order_weight=1.0):
    """Joint loss: CE on grades + weighted CE on (x_i, x_perm(i)) order relations within the batch."""
    logits, feats = model(x)
    perm = torch.randperm(len(y), device=y.device)
    order_target = order_labels(y, y[perm])
    order_logits = model.compare(feats, feats[perm])
    cls_loss = F.cross_entropy(logits, y)
    order_loss = F.cross_entropy(order_logits, order_target)
    return {
        "loss": cls_loss + order_weight * order_loss,
        "cls_loss": cls_loss,
        "order_loss": order_loss,
        "logits": logits,
        "order_logits": order_logits,
        "order_target": order_target,
        "features": feats,
    }


def load_weights(model, path, map_location="cpu"):
    """Load a checkpoint saved by train.py or a legacy Lightning checkpoint (keys are remapped)."""
    state = torch.load(path, map_location=map_location, weights_only=False)
    state = state.get("model", state.get("state_dict", state))
    renames = {"model.": "backbone.", "discriminator_layer1.": "cls_head.", "discriminator_layer2.": "order_head."}
    remapped = {}
    for key, value in state.items():
        for old, new in renames.items():
            if key.startswith(old):
                key = new + key[len(old):]
                break
        remapped[key] = value
    own = model.state_dict()
    remapped = {k: v for k, v in remapped.items() if k in own and own[k].shape == v.shape}
    missing = [k for k in own if k not in remapped]
    if missing:
        raise RuntimeError(f"Checkpoint {path} is missing {len(missing)} weights, e.g. {missing[:3]}")
    model.load_state_dict(remapped)
    return model
