import torch
from f1 import compute_f1
from typing import Optional

class NERMetrics():
    """
    Computes precision, recall, and micro-averaged F1 from a list of predicted and gold labels.
    """
    def __init__(self, number_of_classes: int, none_label: int=0):
        self.number_of_classes = number_of_classes
        self.none_label = none_label
        self.reset()

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        predictions = predictions.cpu()
        gold_labels = gold_labels.cpu()
        mask = mask.cpu()
        for i in range(self.number_of_classes):
            if i == self.none_label:
                continue
            self._true_positives += ((predictions==i)*(gold_labels==i)*mask.bool()).sum().item()
            self._false_positives += ((predictions==i)*(gold_labels!=i)*mask.bool()).sum().item()
            self._true_negatives += ((predictions!=i)*(gold_labels!=i)*mask.bool()).sum().item()
            self._false_negatives += ((predictions!=i)*(gold_labels==i)*mask.bool()).sum().item()

    def get_metric(self, reset=False):
        """
        Returns
        -------
        A tuple of the following metrics based on the accumulated count statistics:
        precision : float
        recall : float
        f1-measure : float
        """
        predicted = self._true_positives + self._false_positives
        gold = self._true_positives + self._false_negatives
        matched = self._true_positives
        precision, recall, f1_measure = compute_f1(predicted, gold, matched)

        # Reset counts if at end of epoch.
        if reset:
            self.reset()

        return precision, recall, f1_measure

    def reset(self):
        self._true_positives = 0
        self._false_positives = 0
        self._true_negatives = 0
        self._false_negatives = 0
