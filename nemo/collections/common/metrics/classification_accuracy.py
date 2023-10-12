# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import re
import string
from collections import Counter
from typing import List, Union

import torch
from torchmetrics import Metric

__all__ = ['TopKClassificationAccuracy']


class TopKClassificationAccuracy(Metric):
    """
    This metric computes numerator and denominator for Overall Accuracy between logits and labels.
    When doing distributed training/evaluation the result of res=TopKClassificationAccuracy(logits, labels) calls
    will be all-reduced between all workers using SUM operations.
    Here contains two numbers res=[correctly_predicted, total_samples]. Accuracy=correctly_predicted/total_samples.

    If used with PytorchLightning LightningModule, include correct_count and total_count inside validation_step results.
    Then aggregate (sum) then at the end of validation epoch to correctly compute validation WER.

    Example:
        def validation_step(self, batch, batch_idx):
            ...
            correct_count, total_count = self._accuracy(logits, labels)
            self.val_outputs = {'val_loss': loss_value, 'val_correct_count': correct_count, 'val_total_count': total_count}
            return self.val_outputs

        def on_validation_epoch_end(self):
            ...
            val_loss_mean = torch.stack([x['val_loss'] for x in self.val_outputs]).mean()
            correct_counts = torch.stack([x['val_correct_counts'] for x in self.val_outputs])
            total_counts = torch.stack([x['val_total_counts'] for x in self.val_outputs])

            topk_scores = compute_topk_accuracy(correct_counts, total_counts)

            tensorboard_log = {'val_loss': val_loss_mean}
            for top_k, score in zip(self._accuracy.top_k, topk_scores):
                tensorboard_log['val_epoch_top@{}'.format(top_k)] = score
            
            self.val_outputs.clear()  # free memory
            return {'log': tensorboard_log}

    Args:
        top_k: Optional list of integers. Defaults to [1].

    Returns:
        res: a torch.Tensor object with two elements: [correct_count, total_count]. To correctly compute average
        accuracy, compute acc=correct_count/total_count
    """

    full_state_update = True

    def __init__(self, top_k=None, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        if top_k is None:
            top_k = [1]

        self.top_k = top_k
        self.add_state(
            "correct_counts_k", default=torch.zeros(len(self.top_k)), dist_reduce_fx='sum', persistent=False
        )
        self.add_state("total_counts_k", default=torch.zeros(len(self.top_k)), dist_reduce_fx='sum', persistent=False)

    @torch.no_grad()
    def top_k_predicted_labels(self, logits: torch.Tensor) -> torch.Tensor:
        max_k = max(self.top_k)
        _, predictions = logits.topk(max_k, dim=1, largest=True, sorted=True)
        return predictions

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            predictions = self.top_k_predicted_labels(logits)
            predictions = predictions.t()
            correct = predictions.eq(labels.view(1, -1)).expand_as(predictions)

            correct_counts_k = []
            total_counts_k = []

            for k in self.top_k:
                correct_k = correct[:k].reshape(-1).long().sum()
                total_k = labels.shape[0]

                correct_counts_k.append(correct_k)
                total_counts_k.append(total_k)

            self.correct_counts_k = torch.tensor(correct_counts_k, dtype=labels.dtype, device=labels.device)
            self.total_counts_k = torch.tensor(total_counts_k, dtype=labels.dtype, device=labels.device)

    def compute(self):
        """
        Computes the top-k accuracy.

        Returns:
            A list of length `K`, such that k-th index corresponds to top-k accuracy
            over all distributed processes.
        """
        if not len(self.correct_counts_k) == len(self.top_k) == len(self.total_counts_k):
            raise ValueError("length of counts must match to topk length")

        if self.top_k == [1]:
            return [self.correct_counts_k.float() / self.total_counts_k]

        else:
            top_k_scores = compute_topk_accuracy(self.correct_counts_k, self.total_counts_k)

            return top_k_scores

    @property
    def top_k(self) -> List[int]:
        return self._top_k

    @top_k.setter
    def top_k(self, value: List[int]):
        if value is None:
            value = [1]

        if type(value) == int:
            value = [value]

        if type(value) != list:
            value = list(value)

        self._top_k = value


def compute_topk_accuracy(correct_counts_k, total_counts_k):
    """
    Computes the top-k accuracy
    Args:
        correct_counts: Tensor of shape [K], K being the top-k parameter.
        total_counts: Tensor of shape [K], and K being the top-k parameter.
    Returns:
        A list of length `K`, such that k-th index corresponds to top-k accuracy
        over all distributed processes.
    """
    top_k_scores = []

    for ki in range(len(correct_counts_k)):
        correct_count = correct_counts_k[ki].item()
        total_count = total_counts_k[ki].item()
        top_k_scores.append(correct_count / float(total_count))

    return top_k_scores


class ExactStringPerCategoryMatchMetric(Metric):
    def __init__(self, categories=[], dist_sync_on_step=False, *args, **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.categories = set(categories)

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        for category in categories:
            self.add_state(f"{category}_total", default=torch.tensor(0), dist_reduce_fx="sum")
            self.add_state(f"{category}_correct", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: str, target: str, category: str = None):
        if pred == target:
            self.correct += 1
        self.total += 1
        if category is None:
            return
        if category in self.categories:
            val = getattr(self, f"{category}_total")
            setattr(self, f"{category}_total", val + 1)
            if pred == target:
                val = getattr(self, f"{category}_correct")
                setattr(self, f"{category}_correct", val + 1)
        else:
            logging.warn(f'{category} is not in the pre-defined list')

    def compute(self):
        results = {}
        results['acc'] = self.correct.float() / self.total
        for category in self.categories:
            results[category] = getattr(self, f"{category}_correct") / getattr(self, f"{category}_total")
        for category in self.categories:
            results[f"{category}_total"] = getattr(self, f"{category}_total")
        return results


class ExactStringMatchMetric(Metric):
    def __init__(self, dist_sync_on_step=False, *args, **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: str, target: str):
#        print("$$$1 pred", pred)
#        print("$$$2 target", target)
        if pred == target:
            self.correct += 1
        self.total += 1

    def compute(self):
        return self.correct.float() / self.total


import numpy as np
from collections import defaultdict
from seqeval.metrics.sequence_labeling import get_entities
from seqeval.metrics.v1 import _prf_divide

def extract_tp_actual_correct(y_true, y_pred, suffix, *args):
    entities_true = defaultdict(set)
    entities_pred = defaultdict(set)
    for type_name, start, end in get_entities(y_true, suffix):
        entities_true[type_name].add((start, end))
    for type_name, start, end in get_entities(y_pred, suffix):
        entities_pred[type_name].add((start, end))

    target_names = sorted(set(entities_true.keys()) | set(entities_pred.keys()))

    tp_sum = np.array([], dtype=np.int32)
    pred_sum = np.array([], dtype=np.int32)
    true_sum = np.array([], dtype=np.int32)
    for type_name in target_names:
        entities_true_type = entities_true.get(type_name, set())
        entities_pred_type = entities_pred.get(type_name, set())
        tp_sum = np.append(tp_sum, len(entities_true_type & entities_pred_type))
        pred_sum = np.append(pred_sum, len(entities_pred_type))
        true_sum = np.append(true_sum, len(entities_true_type))

    return pred_sum, tp_sum, true_sum


class MyF1Score(Metric):
    higher_is_better = True
    def __init__(self, dist_sync_on_step=False, *args, **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("pred_sum", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("tp_sum", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("true_sum", default=torch.tensor(0), dist_reduce_fx="sum")

        self.add_state("good", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("bad", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: str, target: str):
        target = target.split(" ")
        pred = pred.split(" ")

        if target[0] not in ["O", "B", "I"]:
            target = target[1::]
            pred = pred[1::]

        assert isinstance(target, list)
        assert isinstance(pred, list)

        if len(pred) > len(target):
            pred = pred[0:len(target)]

        if len(target) > len(pred):
            pred = pred + (len(target) - len(pred))*['O']

        # check ground truth
        for t in target:
            if t not in ["O", "B", "I"]:
                raise ValueError(f"No such target `{t}`")
        assert len(target) == len(pred)

        #print("$$$$$$$$$$$$$$ MyF1Score target1", target, type(target), len(target))
        #print("$$$$$$$$$$$$$$ MyF1Score pred1", pred, type(pred), len(pred))

        try:
            pred_sum, tp_sum, true_sum = extract_tp_actual_correct(y_true=target, y_pred=pred, suffix=False)

            #print("pred_sum", pred_sum)
            #print("tp_sum", tp_sum)
            #print("true_sum", true_sum)
            #print("size pred_sum", pred_sum.size)
            #if pred_sum.size > 0:
            self.pred_sum += pred_sum.sum()
            self.tp_sum += tp_sum.sum()
            self.true_sum += true_sum.sum()
        except Exception as e:
            print(f"WARNING: extract_tp_actual_correct failed with: {e}")
            print("$$$$$$$$$$$$$$ MyF1Score Exception target", target, type(target), len(target))
            print("$$$$$$$$$$$$$$ MyF1Score Exception pred  ", pred, type(pred), len(pred))

    def compute(self):
        average = None
        warn_for = ('precision', 'recall', 'f-score')
        zero_division = 0
        beta = 1.0
        beta2 = 1.0

        # Divide, and on zero-division, set scores and/or warn according to
        # zero_division:
        precision = _prf_divide(
            numerator=np.asarray([self.tp_sum.cpu()]),
            denominator=np.asarray([self.pred_sum.cpu()]),
            metric='precision',
            modifier='predicted',
            average=average,
            warn_for=warn_for,
            zero_division=zero_division
        )
        recall = _prf_divide(
            numerator=np.asarray([self.tp_sum.cpu()]),
            denominator=np.asarray([self.true_sum.cpu()]),
            metric='recall',
            modifier='true',
            average=average,
            warn_for=warn_for,
            zero_division=zero_division
        )

        # if tp == 0 F will be 1 only if all predictions are zero, all labels are
        # zero, and zero_division=1. In all other case, 0
        if np.isposinf(beta):
            f_score = recall
        else:
            denom = beta2 * precision + recall

            denom[denom == 0.] = 1  # avoid division by 0
            f_score = (1 + beta2) * precision * recall / denom

        return f_score


class TokenF1Score(Metric):
    """Taken from the official evaluation script for v1.1 of the SQuAD dataset"""

    def __init__(self, dist_sync_on_step=False, *args, **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: str, target: Union[str, List[str]]):
        if isinstance(target, str):
            self.correct += self.f1_score(pred, target)
        elif isinstance(target, list):
            self.correct += max([self.f1_score(pred, tgt) for tgt in target])
        self.total += 1

    def compute(self):
        return self.correct.float() / self.total

    def f1_score(self, prediction, ground_truth):
        prediction_tokens = self.normalize(prediction).split()
        ground_truth_tokens = self.normalize(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def normalize(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))
