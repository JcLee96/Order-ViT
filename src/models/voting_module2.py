from typing import Any, List
import torch
import torch.nn as nn
import timm
from pytorch_lightning import LightningModule
from torchmetrics import ConfusionMatrix, F1Score, CohenKappa, Accuracy
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd

from src.datamodules.ubc_datamodule import UbcDataset_pd
from src.datamodules.harvard_datamodule import HarvardDataset_pd
from src.datamodules.colon_datamodule import ColonDataset
from src.datamodules.gastric_datamodule import GastricDataset
import pymrmr
import timm_change

import copy
from scipy.stats import entropy
import operator
from src.utils import (
    vote_results,
    get_shuffled_label,
    get_confmat,
    tensor2np,
    # KMeans,
    # bring_kmeans_trained_feature,
)
import wandb

# min, max = np.zeros((4,4)), np.zeros((4,1))
min, max = [[], [], [], [], [], [], [], [], [], [], [], []], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

class VotingLitModule(LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        name="vit_base_patch16_224",
        pretrained=True,
        scheduler="ReduceLROnPlateau",
        threshold=0.8,
        num_sample=10,
        key="ent",
        sampling="random",
        decide_by_total_probs=False,
        weighted_sum=False,
        module_type="voting",
        seed=42,
        vote_score_way="vote_score_way",
        implementation_model=False,
        class_cnt=4,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        # self.save_feature = []
        # self.save_new_label = []
        self.model = timm.create_model(
                    self.hparams.name, pretrained=self.hparams.pretrained, num_classes=class_cnt)


        self.discriminator_layer1 = nn.Sequential(
            nn.Linear(self.model.classifier.in_features, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, class_cnt),
        ) if 'net' in self.hparams.name else nn.Sequential(
            nn.Linear(self.model.head.in_features, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, class_cnt),
        )
        self.discriminator_layer2 = nn.Sequential(
            nn.Linear(self.model.classifier.in_features * 2, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 3),
        ) if 'net' in self.hparams.name else nn.Sequential(
            nn.Linear(self.model.head.in_features * 2, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 3),
        )

        self.criterion = torch.nn.CrossEntropyLoss()
        self.test_acc = Accuracy()
        self.confusion_matrix = ConfusionMatrix(num_classes=class_cnt)
        self.confusion_matrix_fix_label = ConfusionMatrix(num_classes=class_cnt)
        self.f1_score = F1Score(num_classes=class_cnt, average="macro")
        self.fix_f1_score = F1Score(num_classes=class_cnt, average="macro")
        self.cohen_kappa = CohenKappa(num_classes=class_cnt, weights="quadratic")
        # self.cnt = SumMetric()

    def forward(self, x):  # 4 classification
        return self.discriminator_layer1(self.get_features(x.float()))

    def get_features(self, x):
        """get features from timm models

        Since densenet code is quite different from vit models, the extract part is different
        """
        features = (
            self.model.global_pool(self.model.forward_features(x.float()))
            if "densenet" in self.hparams.name
            else self.model.forward_features(x.float())
        )
        features = (
            features
            if "densenet" in self.hparams.name
            else self.model.forward_head(features, pre_logits=True)
        )
        return features

    def get_comparison_list(self, origin, shuffle):
        """make the comparison(answer) sheet

        origin > shffle -> 0
        origin = shffle -> 1
        origin < shffle -> 2
        """
        comparison = []
        for i, j in zip(origin.tolist(), shuffle.tolist()):
            if i > j:
                comparison.append(0)
            elif i == j:
                comparison.append(1)
            else:
                comparison.append(2)
        return torch.tensor(comparison, device=self.device)

    def shuffle_batch(self, x, y):
        """shuffle batch and get comparison and shuffled targets"""

        indices, shuffle_y = get_shuffled_label(x, y)
        comparison = self.get_comparison_list(y, shuffle_y)

        return indices, comparison, shuffle_y

    def get_convinced(self, x, dataloader):
        """get data which max_probs is bigger than 0.9 and prediction is correct

        max_probs: Maximum of the classification probability values

        """
        convinced = []
        for img, label in dataloader:
            img = img.type_as(x)
            logits = self.forward(img)
            preds = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)
            max_probs = torch.max(probs, 1)

            escape = False
            for i, v in enumerate(max_probs.values):
                if v > 0.9 and preds[i] == label[0]:
                    convinced.append(img[i])
                    if len(convinced) == self.hparams.num_sample:
                        escape = True
                        break
            if escape:
                break
        return convinced

    def check_which_dataset(self):
        """check which dataset it is"""
        dataset_name = str(self.trainer.datamodule.__class__).lower()
        if "ubc" in dataset_name:
            return "ubc"
        elif "harvard" in dataset_name:
            return "harvard"
        elif "colon" in dataset_name:
            return "colon"
        elif "gastric" in dataset_name:
            return "gastric"
        else:
            raise ValueError("Dataset name is not correct")

    def get_trained_dataset(self):
        """get trained dataset by datamodule name
        1) check which dataset it is
        2) get path and label depending on dataset

        """
        data_type = self.check_which_dataset()
        if data_type in ["colon", "gastric"]:
            paths = self.trainer.datamodule.train_dataloader().dataset.image_id
            labels = self.trainer.datamodule.train_dataloader().dataset.labels
        elif data_type in ["harvard", "ubc"]:
            paths = np.array(
                [
                    path
                    for path, label in self.trainer.datamodule.train_dataloader().dataset.pair_list
                ]
            )
            labels = np.array(
                [
                    label
                    for path, label in self.trainer.datamodule.train_dataloader().dataset.pair_list
                ]
            )

        return paths, labels, data_type

    def get_dataclass(self, data_type):
        if data_type == "colon":
            return ColonDataset
        elif data_type == "gastric":
            return GastricDataset
        elif data_type == "harvard":
            return HarvardDataset_pd
        elif data_type == "ubc":
            return UbcDataset_pd

    def bring_trained_feature(self, mode):
        """
        bring trained features, predictions, targets from a npy file

        There are 3 options

        K: numbers of samples

        - 1) random

            randomly select K features by the condition

            condition: 4 classification prediction is correct

        - 2) trust

            randomly select K features by the condition

            condition: 4 classification prediction is correct and the max_probs is bigger than 0.9

        - 3) kmeans

            randomly select K features by the condition

            condition: K features that is centroids in the Kmeans method

        return value : a list of features which index number is same as the class number

        ex) list[0] is class 0 features, list[1] is class 1 features ...

        """
        path = "/path/to/LJC/data/voting"
        # name = f"{self.trainer.datamodule.__class__.__name__.lower()[:-10]}_{self.hparams.name}_{self.trainer.datamodule.hparams.data_ratio}_seed_{self.hparams.seed}.npy"
        name = "colon_vit_base_r50_s16_384_1.0_seed_42.npy"
        features = np.load(f"{path}/features/{name}")
        preds = np.load(f"{path}/preds/{name}")
        targets = np.load(f"{path}/targets/{name}")

        if mode == "random":
            random_idxs = [
                np.random.choice(
                    np.where((preds == i) & (targets == i))[0],
                    self.hparams.num_sample,
                    replace=False,
                )
                for i in range(4)
            ]
            return [features[random_idxs[i]] for i in range(4)]

        elif mode == "trust":
            max_probs = np.load(f"{path}/max_probs/{name}")
            random_idxs = [
                np.random.choice(
                    np.where((preds == i) & (max_probs > 0.9) & (targets == i))[0],
                    self.hparams.num_sample,
                    replace=False,
                )
                for i in range(4)
            ]
            return [features[random_idxs[i]] for i in range(4)]

        # if mode == "kmeans":
        #     return bring_kmeans_trained_feature(features, targets, preds)

        elif mode == "mRMR":

            random_idxs = np.zeros((4, self.hparams.num_sample))
            for i in range(4):
                random_idxs[i] = np.load(path + "/0.9_feature/lda_verified_choice_id_{0}.npy".format(self.hparams.num_sample)
                            )[self.hparams.num_sample * i:self.hparams.num_sample * (i + 1)].reshape(-1)


            return [features[random_idxs[i].astype(int)] for i in range(4)]

    def bring_random_trained_data(self, x):
        """
        get the trained data
        This function starts from the dataset setup to filtering every batch
        So it takes a long time
        """
        self.trainer.datamodule.setup()
        train_img_path, train_img_labels, data_type = self.get_trained_dataset()
        CustomDataset = self.get_dataclass(data_type)
        random_idx_labels = [
            np.random.choice(
                np.where(train_img_labels == i)[0],
                self.hparams.num_sample,
                replace=False,
            )
            for i in range(4)
        ]
        random_10_train_paths = [train_img_path[random_idx_labels[i]] for i in range(4)]
        random_10_train_labels = [
            train_img_labels[random_idx_labels[i]] for i in range(4)
        ]

        df = [
            pd.DataFrame(
                {
                    "path": random_10_train_paths[i],
                    "label": random_10_train_labels[i],
                    "class": random_10_train_labels[i],
                }
            )
            for i in range(4)
        ]

        dataloaders = [
            DataLoader(
                CustomDataset(df[i], self.trainer.datamodule.train_transform),
                batch_size=10,
                num_workers=0,
                pin_memory=False,
                drop_last=False,
                shuffle=False,
            )
            for i in range(4)
        ]

        imgs = []
        for i in range(4):
            img, _ = next(iter(dataloaders[i]))
            img = img.type_as(x)
            imgs.append(img)
        # bring data from train loader and convert to tensor

        return imgs

    def bring_convinced_trained_data(self, x):
        """bring convinced data we trained
        1) load the trained data
        2) bring dataloaders by class
        3) get convinced data by class

        x is used for converting dtype
        """
        self.trainer.datamodule.setup()
        train_img_path, train_img_labels, data_type = self.get_trained_dataset()
        CustomDataset = self.get_dataclass(data_type)

        idx_labels = [np.where(train_img_labels == i)[0] for i in range(4)]
        train_paths = [train_img_path[idx_labels[i]] for i in range(4)]
        train_labels = [train_img_labels[idx_labels[i]] for i in range(4)]
        df = [
            pd.DataFrame(
                {
                    "path": train_paths[i],
                    "label": train_labels[i],
                    "class": train_labels[i],
                }
            )
            for i in range(4)
        ]
        dataloaders = [
            DataLoader(
                CustomDataset(df[i], self.trainer.datamodule.train_transform),
                batch_size=16,
                num_workers=0,
                pin_memory=False,
                drop_last=False,
                shuffle=True,
            )
            for i in range(4)
        ]

        return [self.get_convinced(x, dataloaders[i]) for i in range(4)]

    def compare_test_with_trained(self, feature, trained_features):
        """compare the uncertain data with trained data by class

        1) concatenate uncertain data feature and trained data features
        2) The concatenated features go to discriminator_layer2

        return value:
        comparison results by class

        """
        result_list = []
        for trained_feature in trained_features:
            concat_features = torch.cat(
                (
                    feature.unsqueeze(0),
                    torch.from_numpy(trained_feature).type_as(feature).unsqueeze(0),
                ),
                dim=1,
            )
            logits_compare = self.discriminator_layer2(concat_features)
            preds_compare = torch.argmax(logits_compare, dim=1)
            result_list.append(preds_compare)
        return torch.cat(result_list)

    def voting(
        self,
        entropies,
        max_probs,
        features,
        trained_features,
        probs,
        preds,
        y,
    ):
        """
        voting function

        1) get the threshold key value (entropy or probability)
        2) Check whether the threshold condition is satisfied
            (a) if the key value is entropy, then the condition is "value > threshold"
            (b) if the key value is probability, then the condition is "value < threshold"
        3) If satisfied, go to compare_test_with_trained function and we get the comparison results
                         and calculate the comparison results by the calcluation method (relative or absolute)
        4) There are 3 options to get optimal class by the voting method
            (A) Weighted sum
            (B) Total probability
            (C) None (do nothing to vote score)
        5) switch the original prediction by the classifer to new prediction by the voting method
        """
        cnt_correct_diff = 0
        ops = {"ent": operator.gt, "prob": operator.lt, "sub_prob": operator.lt}
        # gt: > , lt: <
        # threshold_key = entropies if self.hparams.key == "ent" else max_probs.values
        sorted_prob = []
        for i in range(len(probs)):
            sorted_data = sorted(probs[i].cpu().numpy())
            sorted_prob.append(sorted_data[-1] - sorted_data[-2])
        if self.hparams.key == "ent":
            threshold_key = entropies
        elif self.hparams.key == "prob":
            threshold_key = max_probs.values
        else:
            threshold_key = sorted_prob

        # get key data by key
        for idx, value in enumerate(threshold_key):
            if ops[self.hparams.key](value, self.hparams.threshold):
                # if key=='ent' --> value > threshold
                # if key=='prob' --> value < threshold
                results = [
                    self.compare_test_with_trained(features[idx], trained_features[i])
                    for i in range(4)
                ]
                vote_score = vote_results(results, way=self.hparams.vote_score_way)  # vote_score

                if self.hparams.weighted_sum or self.hparams.decide_by_total_probs:
                    total_score = sum(vote_score)
                    prob_vote = np.array([i / total_score for i in vote_score])
                    if self.hparams.weighted_sum:
                        vote_cls = self.get_vote_cls_by_weighted_sum(
                            entropies, probs, idx, prob_vote, vote_score, self.hparams.key
                        )
                    elif self.hparams.decide_by_total_probs:
                        vote_cls = self.get_vote_cls_by_total_probs(
                            probs, idx, prob_vote
                        )
                else:
                    # origin
                    # vote_cls = np.argmax(vote_score)
                    # add (normalization)
                    total_score = sum(vote_score)
                    vote_cls = np.argmax(np.array([i / total_score for i in vote_score]))

                if tensor2np(preds)[idx] != vote_cls and vote_cls == y[idx]:
                    cnt_correct_diff += 1
                # This is for the counting that the vote_label is the same as the true label, but differnt as predict label
                preds[idx] = torch.Tensor([vote_cls]).type_as(y)

        return cnt_correct_diff, preds

    def get_vote_cls_by_total_probs(self, probs_4cls, idx, prob_vote):
        """
                        P(4classification) + P(vote)
        vote score = ------------------------------------
                                        2
        """
        add_probs_4cls_vote = (tensor2np(probs_4cls[idx]) + prob_vote) / 2

        return np.argmax(add_probs_4cls_vote)

    def get_vote_cls_by_weighted_sum(
        self, entropy_4cls, probs_4cls, idx, prob_vote, vote_cnt, key
    ):
        """
                                            e**(-entropy(4classification))
        weight_4classification = ---------------------------------------------------------
                                    e**(-entropy(4classification)) + e**(-entropy(vote))

                                     e**(-entropy(vote))
        weight_vote =  -------------------------------------------------------
                        e**(-entropy(4classification)) + e**(-entropy(vote))

        vote score = weight_4classification * P(4classification) + weight_vote * P(vote)

        w_4cls = np.exp(-e_4cls) / (np.exp(-e_4cls) + np.exp(-e_vote))
        w_vote = np.exp(-e_vote) / (np.exp(-e_4cls) + np.exp(-e_vote))
        """

        """
        normalization weighted sum method
                                            e**(-entropy(4classification))
        weight_4classification = ---------------------------------------------------------
                                    e**(-entropy(4classification)) + e**(-entropy(vote))

                                     e**(-entropy(vote))
        weight_vote =  -------------------------------------------------------
                        e**(-entropy(4classification)) + e**(-entropy(vote))

        vote score = weight_4classification * P(4classification) + weight_vote * P(vote)

        w_4cls = 1 - (e_4cls / (e_4cls + e_vote))
        w_vote = 1 - (e_vote / (e_4cls + e_vote))
        """
        if key == 'ent':
            e_4cls, e_vote = entropy_4cls[idx], entropy(prob_vote)
        else:
            e_4cls, e_vote = list(map(lambda i: entropy(i), tensor2np(probs_4cls)))[idx], entropy(prob_vote)

        # if e_4cls <= e_vote:
        #     return np.argmax(vote_cnt)
        # w_4cls = np.exp(-e_4cls) / (np.exp(-e_4cls) + np.exp(-e_vote))
        # w_vote = np.exp(-e_vote) / (np.exp(-e_4cls) + np.exp(-e_vote))
        w_4cls = 1 - (e_4cls / (e_4cls + e_vote))
        w_vote = 1 - (e_vote / (e_4cls + e_vote))
        voting_classify = tensor2np(probs_4cls[idx]) * w_4cls + prob_vote * w_vote
        return np.argmax(voting_classify)

    def step(self, batch, min, max):
        x, y = batch
        features = self.get_features(x)
        logits_4cls = self.discriminator_layer1(features)
        loss_4cls = self.criterion(logits_4cls, y)
        preds_4cls = torch.argmax(logits_4cls, dim=1)
        probs_4cls = torch.softmax(logits_4cls, dim=1)
        max_probs_4cls = torch.max(probs_4cls, 1)
        origin_preds_4cls = copy.deepcopy(preds_4cls)
        entropy_4cls = list(map(lambda i: entropy(i), tensor2np(probs_4cls)))


        trained_features = self.bring_trained_feature(mode=self.hparams.sampling)
        cnt_correct_diff, new_preds_4cls = self.voting(
            entropy_4cls,
            max_probs_4cls,
            features,
            trained_features,
            probs_4cls,
            preds_4cls,
            y,
        )
        # self.save_feature.append(features)
        # self.save_new_label.append(new_preds_4cls)
        cnt_diff = sum(x != y for x, y in zip(origin_preds_4cls, new_preds_4cls))
        return (
            # features,
            loss_4cls,
            origin_preds_4cls,
            new_preds_4cls,
            y,
            cnt_diff,
            cnt_correct_diff,
            min,
            max,
        )

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        """
        orgin_acc: Accuracy that do not use voting method
        new_acc: Accuracy that do use voting method
        """
        global min, max


        (
            # features,
            loss,
            origin_preds_4cls,
            new_preds_4cls,
            target_4cls,
            cnt_diff,
            cnt_correct_diff,
            min,
            max,
        ) = self.step(batch, min, max)

        self.confusion_matrix(new_preds_4cls, target_4cls)
        self.f1_score(new_preds_4cls, target_4cls)
        self.cohen_kappa(new_preds_4cls, target_4cls)

        origin_acc = self.test_acc(origin_preds_4cls, target_4cls)
        new_acc = self.test_acc(new_preds_4cls, target_4cls)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/origin_acc", origin_acc, on_step=False, on_epoch=True)
        self.log("test/new_acc", new_acc, on_step=False, on_epoch=True)

        return {
            # "features": features,
            "loss": loss,
            "origin_acc": origin_acc,
            "new_acc": new_acc,
            "origin_preds": origin_preds_4cls,
            "new_preds_4cls": new_preds_4cls,
            "targets": target_4cls,
            "cnt_diff": cnt_diff,
            "cnt_correct_diff": cnt_correct_diff,
            "min": min,
            "max": max,
        }

    def test_epoch_end(self, outputs):
        """
        cnt_diff: The number of changes compared to previous predictions
        cnt_correct_diff: The number of correct answers in cnt_diff
        """

        # features = np.vstack([tensor2np(i["features"]) for i in outputs])
        # targets = np.concatenate([tensor2np(i["new_preds_4cls"]) for i in outputs],axis=None)
        #
        # path1 = '/path/to/LJC/data/voting/features/'
        # path2 = '/path/to/LJC/data/voting/targets/'
        # np.save(f'{path1}{self.trainer.datamodule.__class__.__name__.lower()[:-10]}_{self.hparams.name}_{self.trainer.datamodule.hparams.data_ratio}_seed_{self.hparams.seed}_3fig.npy',tensor2np(torch.cat(self.save_feature)))
        # np.save(f'{path2}{self.trainer.datamodule.__class__.__name__.lower()[:-10]}_{self.hparams.name}_{self.trainer.datamodule.hparams.data_ratio}_seed_{self.hparams.seed}_3fig.npy',tensor2np(torch.cat(self.save_new_label)))
        cm = self.confusion_matrix.compute()
        f1 = self.f1_score.compute()
        qwk = self.cohen_kappa.compute()
        plot_img = get_confmat(cm)
        self.logger.experiment.log({"test/conf_matrix": wandb.Image(plot_img)})

        self.log("test/f1_macro", f1, on_step=False, on_epoch=True)
        self.log("test/wqKappa", qwk, on_step=False, on_epoch=True)

        cnt_diff = sum(i["cnt_diff"] for i in outputs).sum()
        cnt_correct_diff = sum(i["cnt_correct_diff"] for i in outputs)

        self.log(
            "test/cnt_diff", cnt_diff, on_epoch=True, on_step=False, reduce_fx="sum"
        )
        self.log(
            "test/cnt_correct_diff",
            cnt_correct_diff,
            on_epoch=True,
            on_step=False,
            reduce_fx="sum",
        )
        self.test_acc.reset()
        self.confusion_matrix.reset()
        self.f1_score.reset()
        self.cohen_kappa.reset()
