"""Training loop, evaluation metrics and comparison-based voting inference."""
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

from src.model import EQUAL, GREATER, LESS, order_vit_loss


def set_seed(seed):
    """Seed python, numpy and torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model, cfg):
    """Adam optimizer and an epoch-level LR scheduler from the `train` config."""
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    name = cfg["scheduler"]
    if name == "CosineAnnealingWarmRestarts":
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=cfg["t_0"], T_mult=1, eta_min=cfg["min_lr"])
    elif name == "CosineAnnealingLR":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["t_max"], eta_min=cfg["min_lr"])
    elif name == "ReduceLROnPlateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    else:
        raise ValueError(f"Unknown scheduler: {name}")
    return opt, sched


def classification_metrics(preds, targets, num_classes):
    """Accuracy, macro F1, quadratic weighted kappa and confusion matrix."""
    labels = list(range(num_classes))
    return {
        "acc": accuracy_score(targets, preds),
        "f1_macro": f1_score(targets, preds, labels=labels, average="macro", zero_division=0),
        "qwk": cohen_kappa_score(targets, preds, labels=labels, weights="quadratic"),
        "confusion_matrix": confusion_matrix(targets, preds, labels=labels),
    }


def train_one_epoch(model, loader, optimizer, device, order_weight, scaler=None):
    """One epoch of joint classification + order learning; returns mean losses and accuracies."""
    model.train()
    totals = {"loss": 0.0, "cls_loss": 0.0, "order_loss": 0.0, "acc": 0.0, "order_acc": 0.0}
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            out = order_vit_loss(model, x, y, order_weight)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(out["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out["loss"].backward()
            optimizer.step()
        totals["loss"] += out["loss"].item()
        totals["cls_loss"] += out["cls_loss"].item()
        totals["order_loss"] += out["order_loss"].item()
        totals["acc"] += (out["logits"].argmax(1) == y).float().mean().item()
        totals["order_acc"] += (out["order_logits"].argmax(1) == out["order_target"]).float().mean().item()
    return {k: v / max(len(loader), 1) for k, v in totals.items()}


@torch.no_grad()
def predict(model, loader, device):
    """Run the model over a loader; returns features, class probabilities and targets as numpy."""
    model.eval()
    feats, probs, targets = [], [], []
    for x, y in loader:
        logits, f = model(x.to(device))
        feats.append(f.float().cpu())
        probs.append(F.softmax(logits.float(), dim=1).cpu())
        targets.append(y)
    return torch.cat(feats).numpy(), torch.cat(probs).numpy(), torch.cat(targets).numpy()


@torch.no_grad()
def evaluate(model, loader, device, num_classes, order_weight=1.0):
    """Validation/test pass: loss (incl. in-batch order loss), order accuracy and grading metrics."""
    model.eval()
    losses, order_hits, order_total, preds, targets = [], 0, 0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = order_vit_loss(model, x, y, order_weight)
        losses.append(out["loss"].item())
        order_hits += (out["order_logits"].argmax(1) == out["order_target"]).sum().item()
        order_total += len(y)
        preds.append(out["logits"].argmax(1).cpu())
        targets.append(y.cpu())
    metrics = classification_metrics(torch.cat(preds).numpy(), torch.cat(targets).numpy(), num_classes)
    metrics["loss"] = float(np.mean(losses))
    metrics["order_acc"] = order_hits / max(order_total, 1)
    return metrics


def build_reference_bank(feats, probs, targets, min_prob=0.9):
    """Per class, keep features of reference images classified correctly (and with max prob > min_prob)."""
    preds = probs.argmax(1)
    keep = (preds == targets) & (probs.max(1) > min_prob)
    return [feats[keep & (targets == c)] for c in range(probs.shape[1])]


def sample_references(bank, num_sample, rng):
    """Randomly draw `num_sample` reference features per class."""
    return [refs[rng.permutation(len(refs))[:num_sample]] for refs in bank]


def vote_scores(results, num_classes, way="relative"):
    """Turn order predictions against references of each class into class vote scores."""
    scores = np.zeros(num_classes)
    for ref_class, relations in enumerate(results):
        for r in relations:
            if r == EQUAL:
                scores[ref_class] += 1
                continue
            targets = range(ref_class + 1, num_classes) if r == GREATER else range(ref_class)
            if len(targets) == 0:
                continue
            for c in targets:
                scores[c] += 1.0 / len(targets) if way == "relative" else 1.0
    return scores


def entropy(p):
    """Shannon entropy (nats) of a probability vector."""
    p = np.asarray(p, dtype=np.float64)
    p = p / p.sum()
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum())


def uncertainty(probs, key):
    """Per-sample uncertainty key: `sub_prob` (top1 - top2), `prob` (top1) or `ent` (entropy)."""
    if key == "sub_prob":
        top2 = np.sort(probs, axis=1)[:, -2:]
        return top2[:, 1] - top2[:, 0]
    if key == "prob":
        return probs.max(1)
    if key == "ent":
        return np.array([entropy(p) for p in probs])
    raise ValueError(f"Unknown key: {key}")


def is_uncertain(values, key, threshold):
    """Samples to re-decide by voting: entropy above, or (sub-)probability below the threshold."""
    return values > threshold if key == "ent" else values < threshold


@torch.no_grad()
def vote_predict(model, feats, probs, references, cfg, device):
    """Refine uncertain predictions by comparing their features with class reference features."""
    num_classes = probs.shape[1]
    preds = probs.argmax(1).copy()
    uncertain = np.where(is_uncertain(uncertainty(probs, cfg["key"]), cfg["key"], cfg["threshold"]))[0]
    for idx in uncertain:
        query = torch.from_numpy(feats[idx]).to(device).unsqueeze(0)
        results = []
        for refs in references:
            if len(refs) == 0:
                results.append(np.empty(0, dtype=np.int64))
                continue
            ref = torch.from_numpy(refs).to(device)
            results.append(model.compare(query.expand(len(ref), -1), ref).argmax(1).cpu().numpy())
        scores = vote_scores(results, num_classes, cfg["vote_score_way"])
        if scores.sum() == 0:
            continue
        prob_vote = scores / scores.sum()
        if cfg["combine"] == "weighted_sum":
            e_cls, e_vote = entropy(probs[idx]), entropy(prob_vote)
            denom = e_cls + e_vote
            w_cls, w_vote = (0.5, 0.5) if denom == 0 else (1 - e_cls / denom, 1 - e_vote / denom)
            preds[idx] = np.argmax(probs[idx] * w_cls + prob_vote * w_vote)
        elif cfg["combine"] == "total_probs":
            preds[idx] = np.argmax((probs[idx] + prob_vote) / 2)
        else:
            preds[idx] = np.argmax(prob_vote)
    return preds, len(uncertain)


@torch.no_grad()
def voting_evaluate(model, test_loader, bank, cfg, device, seed=42):
    """Test-time voting: per batch, sample references and re-decide uncertain samples."""
    feats, probs, targets = predict(model, test_loader, device)
    rng = np.random.default_rng(seed)
    num_classes = probs.shape[1]
    batch = test_loader.batch_size
    preds = np.empty_like(targets)
    n_uncertain = 0
    for start in range(0, len(targets), batch):
        sl = slice(start, start + batch)
        refs = sample_references(bank, cfg["num_sample"], rng)
        preds[sl], n = vote_predict(model, feats[sl], probs[sl], refs, cfg, device)
        n_uncertain += n
    base = classification_metrics(probs.argmax(1), targets, num_classes)
    voted = classification_metrics(preds, targets, num_classes)
    voted["origin_acc"] = base["acc"]
    voted["num_revoted"] = n_uncertain
    voted["num_changed"] = int((preds != probs.argmax(1)).sum())
    return voted
