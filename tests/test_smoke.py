"""Smoke test on a tiny random dataset: build, train a few steps, evaluate and vote."""
import math
import os
import sys
import tempfile

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_split, make_loader  # noqa: E402
from src.engine import (  # noqa: E402
    build_optimizer, build_reference_bank, evaluate, predict, train_one_epoch, vote_scores, voting_evaluate,
)
from src.model import OrderViT, load_weights, order_labels  # noqa: E402

NUM_CLASSES = 4
TINY_MODEL = {"backbone": "vit_tiny_patch16_224", "num_classes": NUM_CLASSES, "pretrained": False, "img_size": 32}


def make_dummy_dataset(root, per_split=8, seed=0):
    """Write random RGB patches + split CSVs (colon TMA layout) and a Test II style folder under `root`."""
    rng = np.random.default_rng(seed)
    image_root, split_dir, test2_root = (os.path.join(root, d) for d in ("colon_tma", "splits", "colon_test2"))
    os.makedirs(split_dir, exist_ok=True)
    for split in ("train", "valid", "test"):
        rows = []
        for i in range(per_split):
            label = i % NUM_CLASSES
            rel = f"case_{split}/{i:04d}_{label}.jpg"
            os.makedirs(os.path.join(image_root, f"case_{split}"), exist_ok=True)
            cv2.imwrite(os.path.join(image_root, rel), rng.integers(0, 256, (48, 48, 3), dtype=np.uint8))
            rows.append({"path": rel, "class": label})
        pd.DataFrame(rows).to_csv(os.path.join(split_dir, f"{split}.csv"), index=False)
    for i in range(per_split):
        folder = os.path.join(test2_root, "wsi_0", "region_0")
        os.makedirs(folder, exist_ok=True)
        img = rng.integers(0, 256, (40, 40, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(folder, f"patch_{i:04d}_{i % NUM_CLASSES + 1}.png"), img)
    return {"image_root": image_root, "split_dir": split_dir, "test2_root": test2_root,
            "img_size": 32, "batch_size": 4, "num_workers": 0}


def test_order_labels():
    """Order relation codes: 0 for >, 1 for =, 2 for <."""
    got = order_labels(torch.tensor([2, 1, 0]), torch.tensor([1, 1, 3]))
    assert got.tolist() == [0, 1, 2]


def test_vote_scores():
    """Relative voting spreads a '>' vote over all higher classes."""
    scores = vote_scores([np.array([0]), np.array([]), np.array([]), np.array([])], NUM_CLASSES, "relative")
    assert np.allclose(scores, [0, 1 / 3, 1 / 3, 1 / 3])


def test_end_to_end():
    """Train two steps, evaluate, vote and reload weights on dummy data."""
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with tempfile.TemporaryDirectory() as root:
        data_cfg = make_dummy_dataset(root)
        train_items = load_split(data_cfg, "train")
        assert len(train_items) == 8 and len(load_split(data_cfg, "test2")) == 8

        loader = make_loader(train_items, data_cfg, train=True)
        x, y = next(iter(loader))
        assert x.shape == (4, 3, 32, 32) and x.dtype == torch.uint8

        model = OrderViT(**TINY_MODEL).to(device)
        logits, feats = model(x.to(device))
        assert logits.shape == (4, NUM_CLASSES)
        assert model.compare(feats, feats).shape == (4, 3)

        train_cfg = {"lr": 1e-4, "weight_decay": 5e-4, "scheduler": "CosineAnnealingWarmRestarts",
                     "t_0": 20, "t_max": 10, "min_lr": 1e-6}
        optimizer, scheduler = build_optimizer(model, train_cfg)
        stats = train_one_epoch(model, loader, optimizer, device, order_weight=1.0)
        scheduler.step()
        assert math.isfinite(stats["loss"]) and stats["loss"] > 0

        metrics = evaluate(model, make_loader(load_split(data_cfg, "valid"), data_cfg), device, NUM_CLASSES)
        assert math.isfinite(metrics["loss"]) and 0 <= metrics["acc"] <= 1
        assert metrics["confusion_matrix"].shape == (NUM_CLASSES, NUM_CLASSES)

        ref_feats, ref_probs, ref_targets = predict(model, make_loader(train_items, data_cfg), device)
        assert ref_feats.shape == (8, model.backbone.num_features) and ref_probs.shape == (8, NUM_CLASSES)
        bank = build_reference_bank(ref_feats, ref_probs, ref_targets, min_prob=0.0)
        vote_cfg = {"key": "sub_prob", "threshold": 1.0, "num_sample": 2,
                    "vote_score_way": "relative", "combine": "weighted_sum"}
        test2_loader = make_loader(load_split(data_cfg, "test2"), data_cfg)
        voted = voting_evaluate(model, test2_loader, bank, vote_cfg, device)
        assert voted["num_revoted"] == 8 and 0 <= voted["acc"] <= 1

        ckpt = os.path.join(root, "best.pt")
        torch.save({"model": model.state_dict()}, ckpt)
        reloaded = load_weights(OrderViT(**TINY_MODEL), ckpt).to(device).eval()
        model.eval()
        with torch.no_grad():
            assert torch.allclose(reloaded(x.to(device))[0], model(x.to(device))[0], atol=1e-5)
        print(f"train loss {stats['loss']:.4f} | val loss {metrics['loss']:.4f} acc {metrics['acc']:.3f}"
              f" | voting acc {voted['acc']:.3f} (origin {voted['origin_acc']:.3f})")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--make-data":
        cfg = make_dummy_dataset(sys.argv[2])
        print(" ".join(f"data.{k}={v}" for k, v in cfg.items()))
        sys.exit(0)
    test_order_labels()
    test_vote_scores()
    test_end_to_end()
    print("smoke test passed")
