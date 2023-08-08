import logging
import warnings
from typing import List, Sequence
import pytorch_lightning as pl
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
import pandas as pd
import math
import random
import torch
from functools import reduce
from torch.nn.functional import pairwise_distance
from torch.nn import _reduction as _Reduction
from typing import Callable, Optional
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from torchmetrics.functional import pairwise_euclidean_distance
import time
import numpy as np


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
            "debug",
            "info",
            "warning",
            "error",
            "exception",
            "fatal",
            "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


log = get_logger(__name__)


def extras(config: DictConfig) -> None:
    """Applies optional utilities, controlled by config flags.

    Utilities:
    - Ignoring python warnings
    - Rich config printing
    """

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # pretty print config tree using Rich library if <config.print_config=True>
    if config.get("print_config"):
        log.info("Printing config tree with Rich! <config.print_config=True>")
        print_config(config, resolve=True)


@rank_zero_only
def print_config(
        config: DictConfig,
        print_order: Sequence[str] = (
                "datamodule",
                "model",
                "callbacks",
                "logger",
                "trainer",
        ),
        resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        print_order (Sequence[str], optional): Determines in what order config components are printed.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    quee = []

    for field in print_order:
        quee.append(field) if field in config else log.info(
            f"Field '{field}' not found in config"
        )

    for field in config:
        if field not in quee:
            quee.append(field)

    for field in quee:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = config[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.log", "w") as file:
        rich.print(tree, file=file)


@rank_zero_only
def log_hyperparameters(
        config: DictConfig,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        trainer: pl.Trainer,
        callbacks: List[pl.Callback],
        logger: List[pl.loggers.LightningLoggerBase],
) -> None:  # sourcery skip: merge-dict-assign
    """Controls which config parts are saved by Lightning loggers.

    Additionaly saves:
    - number of model parameters
    """

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["model"] = config["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["datamodule"] = config["datamodule"]
    hparams["trainer"] = config["trainer"]

    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)


def finish(
        config: DictConfig,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        trainer: pl.Trainer,
        callbacks: List[pl.Callback],
        logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, pl.loggers.wandb.WandbLogger):
            import wandb

            wandb.finish()


def bring_colon_dataset_csv(datatype, stage=None):
    # Directories
    path = f"/path/to/LJC/data/colon_tma/{datatype}/"
    # path = "/path/to/LJC/data/colon_tma/COLON_PATCHES_1024/"
    if stage != "fit" and stage is not None:
        return pd.read_csv(path + "test.csv")
    df_train = pd.read_csv(path + "train.csv")
    df_val = pd.read_csv(path + "valid.csv")
    return df_train, df_val


def bring_dataset_colontest2_csv(stage=None):
    # Directories
    # path = "/path/to/LJC/data/colon_45WSIs_1144_08_step05_05/"

    # if stage != "fit" and stage is not None:
    #     return pd.read_csv(f"{path}test.csv")
    # df_train = pd.read_csv(f"{path}train.csv")
    # df_val = pd.read_csv(f"{path}valid.csv")
    # return df_train, df_val

    data_path = pd.read_csv('/path/to/LJC/data/colon_45WSIs_1144_08_step05_05/colon_test2_all_data.csv')
    return data_path


def bring_gastric_dataset_csv(stage=None):
    # Directories
    path = "/path/to/LJC/data/gastric/"

    if stage != "fit" and stage is not None:
        return pd.read_csv(f"{path}class4_step10_ds_test.csv")
    df_train = pd.read_csv(f"{path}class4_step05_ds_train.csv")
    df_val = pd.read_csv(f"{path}class4_step10_ds_valid.csv")
    return df_train, df_val


def bring_gastric_dataset_csv2(stage=None):
    # Directories
    path = "/path/to/LJC/data/gastric/"

    if stage != "fit" and stage is not None:
        return pd.read_csv(f"{path}class4_step10_ds_test2.csv")
    df_train = pd.read_csv(f"{path}class4_step05_ds_train2.csv")
    df_val = pd.read_csv(f"{path}class4_step10_ds_valid2.csv")
    return df_train, df_val


def bring_agcc_dataset_csv(stage=None):
    # Directories
    path = f"/data1/ljc/AGGC22/"

    if stage != "fit" and stage is not None:
        return pd.read_csv(f"{path}test.csv")
    df_train = pd.read_csv(f"{path}train.csv")
    df_val = pd.read_csv(f"{path}valid.csv")
    return df_train, df_val


def bring_agcc_dataset_csv(stage=None):
    # Directories
    path = f"/data1/ljc/AGGC22/"

    if stage != "fit" and stage is not None:
        return pd.read_csv(f"{path}test.csv")
    df_train = pd.read_csv(f"{path}train.csv")
    df_val = pd.read_csv(f"{path}valid.csv")
    return df_train, df_val


def tensor2np(data: torch.Tensor):
    """convert tensor to numpy array"""
    return data if isinstance(data, np.ndarray) else data.detach().cpu().numpy()


def get_shuffled_label(x, y):
    """get shuffled label from x and y

    1) shuffle the pair (x,y)
    2) get the shuffled indices and targets

    """
    pair = list(enumerate(list(pair) for pair in zip(x, y)))
    pair = random.sample(pair, len(pair))
    indices, pair = zip(*pair)
    indices = list(indices)
    pair = list(pair)
    shuffle_y = [i[1] for i in pair]
    shuffle_y = torch.stack(shuffle_y, dim=0)

    return indices, shuffle_y


def cal_relative(n, r, class_score):
    """
    calculate voting score relatively

    r: the compare result (0: >, 1: =, 2: <)
    n: the class of data

    This method considers the relative value
    ex) if r==0 & n==0, then there are 3 classes (1,2,3) that is higher than 0
        so add 1/3 for each class
        if r==2 & n==2, then there are 2 classes(0,1) that is lower than 2
        so add 1/2 for each class
    """
    if r == 0:
        score = 1 / (3 - n)
        for idx in range(n + 1, 4):
            class_score[idx] += score
        return class_score
    if r == 1:
        class_score[n] += 1
        return class_score
    if r == 2:
        score = 1 / n
        for idx in range(n):
            class_score[idx] += score
        return class_score


def cal_absolute(n, r, class_score):
    """
    calculate voting score absolutely

    r: the compare result (0: >, 1: =, 2: <)
    n: the class of data

    This method considers the absolute value

    add 1 to the class that satisfy the condition
    """
    if r == 0:
        for idx in range(n + 1, 4):
            class_score[idx] += 1
        return class_score
    if r == 1:
        class_score[n] += 1
        return class_score
    if r == 2:
        for idx in range(n):
            class_score[idx] += 1
        return class_score


def vote_results(results: List, way):
    class_score = [0] * 4
    # class_score[4] is other case
    if way == 'absolute':

        for r0, r1, r2, r3 in zip(results[0], results[1], results[2], results[3]):
            class_score = cal_absolute(0, r0, class_score)
            class_score = cal_absolute(1, r1, class_score)
            class_score = cal_absolute(2, r2, class_score)
            class_score = cal_absolute(3, r3, class_score)

        return class_score

    if way == 'relative':
        for r0, r1, r2, r3 in zip(results[0], results[1], results[2], results[3]):
            class_score = cal_relative(0, r0, class_score)
            class_score = cal_relative(1, r1, class_score)
            class_score = cal_relative(2, r2, class_score)
            class_score = cal_relative(3, r3, class_score)

        return class_score


def dist_indexing(y, shuffle_y, y_idx_groupby, dist_matrix):
    indices = []
    for i, (yV, shuffleV) in enumerate(zip(y, shuffle_y)):
        if yV == shuffleV:
            indices.append(
                y_idx_groupby[yV][dist_matrix[i][y_idx_groupby[yV]].argmax()]
            )
        elif yV > shuffleV:
            flatten_ = reduce(lambda a, b: a + b, y_idx_groupby[:yV])
            indices.append(flatten_[dist_matrix[i][flatten_].argmin()])
        else:
            flatten_ = reduce(lambda a, b: a + b, y_idx_groupby[yV + 1:])
            indices.append(flatten_[dist_matrix[i][flatten_].argmin()])
    return indices


def params_freeze(model):
    model.blocks[22:].requires_grad_(False)
    for name, param in model.named_parameters():
        param.requires_grad = "head" in name
        param.requires_grad = "norm" in name


def get_distmat_heatmap(df, targets):
    df = pd.DataFrame(df.detach().cpu().numpy())
    plt.clf()
    plt.figure(figsize=(30, 30))
    confmat_heatmap = sns.heatmap(
        data=df,
        cmap="RdYlGn",
        annot=True,
        annot_kws={"size": 15},
        fmt=".2f",
        xticklabels=targets.detach().cpu().numpy(),
        yticklabels=targets.detach().cpu().numpy(),
        cbar=False,
    )

    confmat_heatmap.xaxis.set_label_position("top")
    plt.yticks(rotation=0)
    confmat_heatmap.tick_params(axis="x", which="both", bottom=False)

    return confmat_heatmap.get_figure()


def get_confmat(df):
    df = pd.DataFrame(df.detach().cpu().numpy())
    plt.clf()  # ADD THIS LINE
    plt.figure(figsize=(10, 10))
    confmat_heatmap = sns.heatmap(
        data=df,
        cmap="RdYlGn",
        annot=True,
        fmt=".3f",
        cbar=False,
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted label")

    confmat_heatmap.xaxis.set_label_position("top")
    plt.yticks(rotation=0)
    confmat_heatmap.tick_params(axis="x", which="both", bottom=False)

    return confmat_heatmap.get_figure()


def get_feature_df(features, targets):
    cols = [f"feature_{i}" for i in range(features.shape[1])]
    df = pd.DataFrame(features.detach().cpu().numpy(), columns=cols)
    label_dict = {0: "BN_0", 1: "WD_1", 2: "MD_2", 3: "PD_3"}
    df["LABEL"] = targets.detach().cpu().numpy()
    df["LABEL"] = df["LABEL"].map(label_dict)

    return df


def triplet_margin_with_distance_loss(
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        *,
        distance_function: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        margin: float = 1.0,
        swap: bool = False,
        reduction: str = "mean",
) -> torch.Tensor:
    r"""
    See :class:`~torch.nn.TripletMarginWithDistanceLoss` for details.
    """
    if torch.jit.is_scripting():
        raise NotImplementedError(
            "F.triplet_margin_with_distance_loss does not support JIT scripting: "
            "functions requiring Callables cannot be scripted."
        )

    distance_function = (
        distance_function if distance_function is not None else pairwise_distance
    )

    positive_dist = distance_function(anchor, positive)
    negative_dist = distance_function(anchor, negative)

    if swap:
        swap_dist = distance_function(positive, negative)
        negative_dist = torch.min(negative_dist, swap_dist)

    output = torch.clamp(positive_dist - negative_dist + margin, min=0.0)

    reduction_enum = _Reduction.get_enum(reduction)
    if reduction_enum == 1:
        return output.mean()
    elif reduction_enum == 2:
        return output.sum()
    else:
        return output


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.cdist(inputs, inputs, compute_mode="donot_use_mm_for_euclid_dist")

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y), dist
        # This is same as " max( D(a,p)-D(a,n) + margin, 0 ) "


def get_max(lst):
    return torch.max(lst).unsqueeze(0)


def get_min(lst):
    return torch.min(lst).unsqueeze(0)


class TripletLossWithGL(nn.Module):
    """Triplet loss with hard positive/negative mining and Greater/Less.
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3):
        super(TripletLossWithGL, self).__init__()
        self.margin = margin

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)

        # Make a pairwise distance matrix
        dist = torch.cdist(
            inputs, inputs, 2, compute_mode="donot_use_mm_for_euclid_dist"
        )
        # Make a relationship mask based on the anchor
        mask_eq = torch.mul(targets.expand(n, n).eq(targets.expand(n, n).t()), 1)  # =
        mask_gt = torch.mul(targets.expand(n, n).gt(targets.expand(n, n).t()), 2)  # <
        mask_lt = torch.mul(targets.expand(n, n).lt(targets.expand(n, n).t()), 3)  # >
        mask = mask_eq + mask_gt + mask_lt

        hard_ap, hard_an_g, hard_an_l = [], [], []
        # hard_ap: anchor positive, hard_an_g: anchor negative greater, hard_an_l: anchor negative less
        cnt = 0
        for i in range(n):
            # Get a list of positive & negative values
            dists_ap = dist[i][mask[i] == 1]
            dists_an_g = dist[i][mask[i] == 2]
            dists_an_l = dist[i][mask[i] == 3]
            if len(dists_ap) == 1:
                cnt += 1
            # For each anchor, find the hardest positive and negative
            hard_ap.append(get_max(dists_ap))

            if len(dists_an_g) == 0:
                hard_an_g.append(torch.tensor([10000], device=dists_an_g.device))
            else:
                hard_an_g.append(get_min(dists_an_g))

            if len(dists_an_l) == 0:
                hard_an_l.append(torch.tensor([10000], device=dists_an_l.device))
            else:
                hard_an_l.append(get_min(dists_an_l))

        hard_ap = torch.cat(hard_ap)
        hard_an_g = torch.cat(hard_an_g)
        hard_an_l = torch.cat(hard_an_l)

        # Compute Triplet loss
        loss = 0
        loss += F.relu(
            hard_ap - abs(hard_an_g - hard_an_l) + self.margin
        ).mean()  # " max( D(a,p) - |D(a,n>)-D(a,n<)| + margin, 0 ) "
        # loss += F.relu(hard_ap - hard_an_g + self.margin).mean()  # " max( D(a,p) - D(a,n>) + margin, 0 ) "
        # loss += F.relu(hard_ap - hard_an_l + self.margin).mean()  # " max( D(a,p) - D(a,n<) + margin, 0 ) "

        # return torch.div(loss, 3), dist, cnt
        return loss, dist, cnt


def mrmr_base(K, relevance_func, redundancy_func,
              relevance_args={}, redundancy_args={},
              denominator_func=np.mean, only_same_domain=False,
              return_scores=False, show_progress=True):
    """General function for mRMR algorithm.
    Parameters
    ----------
    K: int
        Maximum number of features to select. The length of the output is *at most* equal to K
    relevance_func: callable
        Function for computing Relevance.
        It must return a pandas.Series containing the relevance (a number between 0 and +Inf)
        for each feature. The index of the Series must consist of feature names.
    redundancy_func: callable
        Function for computing Redundancy.
        It must return a pandas.Series containing the redundancy (a number between -1 and 1,
        but note that negative numbers will be taken in absolute value) of some features (called features)
        with respect to a variable (called target_variable).
        It must have *at least* two parameters: "target_variable" and "features".
        The index of the Series must consist of feature names.
    relevance_args: dict (optional, default={})
        Optional arguments for relevance_func.
    redundancy_args: dict (optional, default={])
        Optional arguments for redundancy_func.
    denominator_func: callable (optional, default=numpy.mean)
        Synthesis function to apply to the denominator of MRMR score.
        It must take an iterable as input and return a scalar.
    only_same_domain: bool (optional, default=False)
        If False, all the necessary redundancy coefficients are computed.
        If True, only features belonging to the same domain are compared.
        Domain is defined by the string preceding the first underscore:
        for instance "cusinfo_age" and "cusinfo_income" belong to the same domain, whereas "age" and "income" don't.
    return_scores: bool (optional, default=False)
        If False, only the list of selected features is returned.
        If True, a tuple containing (list of selected features, relevance, redundancy) is returned.
    show_progress: bool (optional, default=True)
        If False, no progress bar is displayed.
        If True, a TQDM progress bar shows the number of features processed.
    Returns
    -------
    selected_features: list of str
        List of selected features.
    """

    FLOOR = .001

    relevance = relevance_func(**relevance_args)
    features = relevance[relevance.fillna(0) > 0].index.to_list()
    relevance = relevance.loc[features]
    redundancy = pd.DataFrame(FLOOR, index=features, columns=features)
    K = min(K, len(features))
    selected_features = []
    not_selected_features = features.copy()

    for i in tqdm(range(K), disable=not show_progress):

        score_numerator = relevance.loc[not_selected_features]

        if i > 0:

            last_selected_feature = selected_features[-1]

            if only_same_domain:
                not_selected_features_sub = [c for c in not_selected_features if
                                             c.split('_')[0] == last_selected_feature.split('_')[0]]
            else:
                not_selected_features_sub = not_selected_features

            if not_selected_features_sub:
                redundancy.loc[not_selected_features_sub, last_selected_feature] = redundancy_func(
                    target_column=last_selected_feature,
                    features=not_selected_features_sub,
                    **redundancy_args
                ).fillna(FLOOR).abs().clip(FLOOR)
                score_denominator = redundancy.loc[not_selected_features, selected_features].apply(
                    denominator_func, axis=1).replace(1.0, float('Inf'))

        else:
            score_denominator = pd.Series(1, index=features)

        score = score_numerator / score_denominator

        best_feature = score.index[score.argmax()]
        selected_features.append(best_feature)
        not_selected_features.remove(best_feature)

    if not return_scores:
        return selected_features
    else:
        return (selected_features, relevance, redundancy)


class CEOLoss(nn.Module):
    """
    Args:
        num_classes (int): number of classes.
    """

    def __init__(self, num_classes=4):
        super(CEOLoss, self).__init__()
        self.num_classes = num_classes
        self.level = torch.arange(-num_classes + 1, num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()

    # def cross_entropy_loss(self, pred, target):
    #     c = 1e-8
    #     if pred.ndim == 1:
    #         target = target.reshape(1, target.size)
    #         y = pred.reshape(1, pred.size)
    #
    #     batch_size = pred.shape[0]
    #     return -np.sum(target * np.log(pred + c)) / batch_size

    def NLLLoss(self, logs, targets):
        out = torch.zeros_like(targets, dtype=torch.float)
        for i in range(len(targets)):
            out[i] = logs[i][targets[i]]
        return -out.sum() / len(out)

    def forward(self, x, y, logits_4cls, class_y):
        """"
        Args:
            x (tensor): Regression/ordinal output, size (B), type: float
            y (tensor): Ground truth,  size (B), type: int/long

        Returns:
            CEOLoss: Cross-Entropy Ordinal loss
        """
        acc_loss = 0
        for i in range(len(x)):
            loss = torch.log(torch.sum(torch.exp(x[i]))) - x[i, y[i].item()]  # x[i, y[i].item()]
            acc_loss += loss

        acc_loss /= len(x)

        log_softmax = torch.nn.LogSoftmax(dim=1)
        x_log = log_softmax(logits_4cls)
        result_loss = self.NLLLoss(x, y)

        loss_4cls = self.criterion(logits_4cls, class_y)

        levels = self.level.repeat(len(class_y), 1).cuda()
        logit = x.repeat(len(self.level), 1).permute(1, 0)
        logit = torch.abs(logit - levels)  # 분모: denominator 분자: numerator

        return F.cross_entropy(-logit, class_y, reduction='mean')

def inverse_huber_loss(target, output, T):
    absdiff = torch.abs(output - target)
    C = T * torch.max(absdiff).item()
    return torch.mean(torch.where(absdiff < C, absdiff, (absdiff * absdiff + C * C) / (2 * C)))