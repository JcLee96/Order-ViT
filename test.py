"""Evaluate an Order-ViT checkpoint, optionally with comparison-based voting against reference images."""
import argparse
import json

import torch

from src.config import load_config
from src.data import load_split, make_loader
from src.engine import build_reference_bank, evaluate, predict, set_seed, voting_evaluate
from src.model import OrderViT, load_weights


def parse_args():
    """CLI: checkpoint, test split, voting switch and `key.sub=value` overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/colon.yaml")
    parser.add_argument("--ckpt", required=True, help="best.pt from train.py (legacy Lightning .ckpt also works)")
    parser.add_argument("--split", default="test", choices=["valid", "test", "test2"])
    parser.add_argument("--voting", action="store_true", help="re-decide uncertain samples by order voting")
    parser.add_argument("overrides", nargs="*", help="e.g. voting.num_sample=30")
    return parser.parse_args()


def main():
    """Print accuracy, macro F1, QWK and the confusion matrix (with and without voting)."""
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = cfg["data"]

    model = OrderViT(**{**cfg["model"], "pretrained": False})
    model = load_weights(model, args.ckpt).to(device).eval()
    test_loader = make_loader(load_split(data_cfg, args.split), data_cfg)

    if args.voting:
        ref_items = load_split(data_cfg, "train") + load_split(data_cfg, "valid")
        feats, probs, targets = predict(model, make_loader(ref_items, data_cfg), device)
        bank = build_reference_bank(feats, probs, targets, cfg["voting"]["min_prob"])
        print("reference images per class:", [len(b) for b in bank])
        metrics = voting_evaluate(model, test_loader, bank, cfg["voting"], device, cfg["seed"])
    else:
        metrics = evaluate(model, test_loader, device, cfg["model"]["num_classes"])

    print(args.split, json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"}, indent=2, default=float))
    print("confusion matrix\n", metrics["confusion_matrix"])


if __name__ == "__main__":
    main()
