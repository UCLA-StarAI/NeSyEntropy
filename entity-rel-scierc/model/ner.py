import torch
from torch.nn import functional as F
import torch.nn as nn
from ner_metrics import NERMetrics
from typing import Any, Dict, List, Optional, Callable


class NERTagger(torch.nn.Module):

    def __init__(self):
        super(NERTagger, self).__init__()
        self.n_labels = 8
        self.feedfwd = nn.Sequential(
                    nn.Linear(768, 150),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(150, 150),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(150, self.n_labels - 1) # We do not score the null label
                )
        self.ner_metric = NERMetrics(self.n_labels, 0)
        self.loss = torch.nn.CrossEntropyLoss(reduction="sum")

    def forward(self,  # type: ignore
                spans: torch.IntTensor,
                span_mask: torch.IntTensor,
                span_embeddings: torch.IntTensor,
                sentence_lengths: torch.Tensor,
                ner_labels: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        ner_scores = self.feedfwd(span_embeddings)

        # Give large negative score to masked out elements
        mask = span_mask.unsqueeze(-1)
        ner_scores = util.replace_masked_values(ner_scores, mask.bool(), -1e20)

        # The dummy scores are the score for the null label
        dummy_dims = [ner_scores.size(0), ner_scores.size(1), 1]
        dummy_scores = ner_scores.new_zeros(*dummy_dims)
        ner_scores = torch.cat((dummy_scores, ner_scores), -1)

        _, predicted_ner = ner_scores.max(2)

        predictions = self.predict(ner_scores.detach().cpu(),
                                   spans.detach().cpu(),
                                   span_mask.detach().cpu(),
                                   metadata)
        output_dict = {"predictions": predictions}

        if ner_labels is not None:
            metrics = self._ner_metrics[self._active_namespace]
            metrics(predicted_ner, ner_labels, span_mask)
            ner_scores_flat = ner_scores.view(-1, self._n_labels[self._active_namespace])
            ner_labels_flat = ner_labels.view(-1)
            mask_flat = span_mask.view(-1).bool()

            loss = self._loss(ner_scores_flat[mask_flat], ner_labels_flat[mask_flat])
            output_dict["loss"] = loss

        return output_dict

    def predict(self, ner_scores, spans, span_mask, metadata):
        # TODO(dwadden) Make sure the iteration works in documents with a single sentence.
        # Zipping up and iterating iterates over the zeroth dimension of each tensor; this
        # corresponds to iterating over sentences.
        predictions = []
        zipped = zip(ner_scores, spans, span_mask, metadata)
        for ner_scores_sent, spans_sent, span_mask_sent, sentence in zipped:
            predicted_scores_raw, predicted_labels = ner_scores_sent.max(dim=1)
            softmax_scores = F.softmax(ner_scores_sent, dim=1)
            predicted_scores_softmax, _ = softmax_scores.max(dim=1)
            ix = (predicted_labels != 0) & span_mask_sent.bool()

            predictions_sent = []
            zip_pred = zip(predicted_labels[ix], predicted_scores_raw[ix],
                           predicted_scores_softmax[ix], spans_sent[ix])
            for label, label_score_raw, label_score_softmax, label_span in zip_pred:
                label_str = self.vocab.get_token_from_index(label.item(), self._active_namespace)
                span_start, span_end = label_span.tolist()
                ner = [span_start, span_end, label_str, label_score_raw.item(),
                       label_score_softmax.item()]
                prediction = document.PredictedNER(ner, sentence, sentence_offsets=True)
                predictions_sent.append(prediction)

            predictions.append(predictions_sent)

        return predictions

    def get_metrics(self, reset: bool = False):
        "Loop over the metrics for all namespaces, and return as dict."
        res = {}
        for namespace, metrics in self._ner_metrics.items():
            precision, recall, f1 = metrics.get_metric(reset)
            prefix = namespace.replace("_labels", "")
            to_update = {f"{prefix}_precision": precision,
                         f"{prefix}_recall": recall,
                         f"{prefix}_f1": f1}
            res.update(to_update)

        res_avg = {}
        for name in ["precision", "recall", "f1"]:
            values = [res[key] for key in res if name in key]
            res_avg[f"MEAN__ner_{name}"] = sum(values) / len(values) if values else 0
            res.update(res_avg)

        return res

if __name__ == "__main__":
    ner_tagger = NERTagger()
