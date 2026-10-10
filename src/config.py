"""YAML config loading with `key.sub=value` command-line overrides."""
import yaml


def load_config(path, overrides=()):
    """Load a YAML config and apply dotted overrides such as `train.epochs=1`."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides:
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    return cfg
