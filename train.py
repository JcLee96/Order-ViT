"""Train Order-ViT (grade classification + order learning) and keep the best checkpoint by val accuracy."""
import argparse
import json
import os

import torch

from src.config import load_config
from src.data import load_split, make_loader
from src.engine import build_optimizer, evaluate, set_seed, train_one_epoch
from src.model import OrderViT


def parse_args():
    """CLI: config path plus optional `key.sub=value` overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/colon.yaml")
    parser.add_argument("overrides", nargs="*", help="e.g. train.epochs=1 data.batch_size=8")
    return parser.parse_args()


def main():
    """Train, validate each epoch, save `best.pt` / `last.pt`, then report test metrics of the best model."""
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["output_dir"], exist_ok=True)

    data_cfg, train_cfg = cfg["data"], cfg["train"]
    train_loader = make_loader(load_split(data_cfg, "train"), data_cfg, train=True)
    valid_loader = make_loader(load_split(data_cfg, "valid"), data_cfg)

    model = OrderViT(**cfg["model"]).to(device)
    optimizer, scheduler = build_optimizer(model, train_cfg)
    scaler = torch.amp.GradScaler("cuda") if train_cfg["amp"] and device.type == "cuda" else None
    num_classes = cfg["model"]["num_classes"]

    best_acc = -1.0
    for epoch in range(train_cfg["epochs"]):
        tr = train_one_epoch(model, train_loader, optimizer, device, train_cfg["order_weight"], scaler)
        va = evaluate(model, valid_loader, device, num_classes, train_cfg["order_weight"])
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(va["loss"])
        else:
            scheduler.step()
        print(
            f"epoch {epoch:03d} | train loss {tr['loss']:.4f} acc {tr['acc']:.4f} order_acc {tr['order_acc']:.4f}"
            f" | val loss {va['loss']:.4f} acc {va['acc']:.4f} f1 {va['f1_macro']:.4f} qwk {va['qwk']:.4f}"
            f" order_acc {va['order_acc']:.4f}"
        )
        state = {"model": model.state_dict(), "epoch": epoch, "config": cfg}
        torch.save(state, os.path.join(cfg["output_dir"], "last.pt"))
        if va["acc"] > best_acc:
            best_acc = va["acc"]
            torch.save(state, os.path.join(cfg["output_dir"], "best.pt"))

    best = torch.load(os.path.join(cfg["output_dir"], "best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    test = evaluate(model, make_loader(load_split(data_cfg, "test"), data_cfg), device, num_classes)
    summary = {k: v for k, v in test.items() if k != "confusion_matrix"}
    print("test", json.dumps(summary, indent=2))
    print("confusion matrix\n", test["confusion_matrix"])


if __name__ == "__main__":
    main()
