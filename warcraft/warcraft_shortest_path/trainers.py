import random

from comb_modules.utils import cached_vertex_grid_to_edges, cached_vertex_grid_to_edges_grid_coords
import time
from abc import ABC, abstractmethod

import torch
from comb_modules.losses import HammingLoss
from comb_modules.dijkstra import ShortestPath
from logger import Logger
from models import get_model
from utils import AverageMeter, optimizer_from_string, customdefaultdict
from decorators import to_tensor, to_numpy
from . import metrics
from .metrics import compute_metrics
import numpy as np
from collections import defaultdict
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from .visualization import draw_paths_on_image
from data.sl_constraints import get_neighbors

def get_trainer(trainer_name):
    trainers = {"Baseline": BaselineTrainer, "DijkstraOnFull": DijkstraOnFull}
    return trainers[trainer_name]

# Semantic Loss
import sys
sys.path.insert(0, '..') #Compute mpe
sys.path.insert(0, '../pypsdd')

from compute_mpe import CircuitMPE

idxs_6x6 = [[] for i in range(9)]
for i in range(6):
    for j in range(6):
        for a in range(3):
            for b in range(3):
                idxs_6x6[a*3+b].append((i+a*3)*12+j+b*3)

cmpes_6x6 = [CircuitMPE(f'data/6x6_tiles/{i}.vtree', f'data/6x6_tiles/{i}.sdd') for i in range(9)]
def sl_6x6_tiles(outputu):
    sl = torch.tensor(0.0).cuda()
    entropy = torch.tensor(0.0).cuda()
    for i in range(9):
        cur_cmpe = cmpes_6x6[i]
        lit_weights = [[1.0-outputu[x], outputu[x]] for x in idxs_6x6[i]]

        wmc_tf = cur_cmpe.get_tf_ac(lit_weights)

        sl = sl - torch.log(wmc_tf).mean()

        entropy = entropy +  cur_cmpe.Shannon_entropy(lit_weights).mean()

    return (sl, entropy)

class ShortestPathAbstractTrainer(ABC):
    def __init__(
        self,
        *,
        train_iterator,
        test_iterator,
        metadata,
        use_cuda,
        batch_size,
        optimizer_name,
        optimizer_params,
        model_params,
        fast_mode,
        neighbourhood_fn,
        preload_batch,
        lr_milestone_1,
        lr_milestone_2,
        use_lr_scheduling,
        sl_weight,
        entreg_weight
    ):

        self.fast_mode = fast_mode
        self.use_cuda = use_cuda
        self.optimizer_params = optimizer_params
        self.batch_size = batch_size
        self.test_iterator = test_iterator
        self.train_iterator = train_iterator
        self.metadata = metadata
        self.grid_dim = int(np.sqrt(self.metadata["output_features"]))
        self.neighbourhood_fn = neighbourhood_fn
        self.preload_batch = preload_batch
        self.sl_weight=sl_weight
        self.entreg_weight=entreg_weight

        self.model = None
        self.build_model(**model_params)

        if self.use_cuda:
            self.model.to("cuda")
        self.optimizer = optimizer_from_string(optimizer_name)(self.model.parameters(), **optimizer_params)
        self.use_lr_scheduling = use_lr_scheduling
        if use_lr_scheduling:
            self.scheduler = MultiStepLR(self.optimizer, milestones=[lr_milestone_1, lr_milestone_2], gamma=0.1)
        self.epochs = 0
        self.train_logger = Logger(scope="training", default_output="tensorboard")
        self.val_logger = Logger(scope="validation", default_output="tensorboard")

    def train_epoch(self):
        self.epochs += 1
        batch_time = AverageMeter("Batch time")
        data_time = AverageMeter("Data time")
        cuda_time = AverageMeter("Cuda time")
        avg_loss = AverageMeter("Loss")
        avg_accuracy = AverageMeter("Accuracy")
        avg_perfect_accuracy = AverageMeter("Perfect Accuracy")

        avg_metrics = customdefaultdict(lambda k: AverageMeter("train_"+k))


        self.model.train()

        end = time.time()

        iterator = self.train_iterator.get_epoch_iterator(batch_size=self.batch_size, number_of_epochs=1, device='cuda' if self.use_cuda else 'cpu', preload=self.preload_batch)
        for i, data in enumerate(iterator):
            input, true_path, true_weights = data["images"], data["labels"],  data["true_weights"]

            if i == 0:
                self.log(data, train=True)
            cuda_begin = time.time()
            cuda_time.update(time.time()-cuda_begin)

            # measure data loading time
            data_time.update(time.time() - end)

            loss, accuracy, last_suggestion = self.forward_pass(input, true_path, train=True, i=i)

            suggested_path = last_suggestion["suggested_path"]

            batch_metrics = metrics.compute_metrics(true_paths=true_path,
            suggested_paths=suggested_path, true_vertex_costs=true_weights)

            # update batch metrics
            {avg_metrics[k].update(v, input.size(0)) for k, v in batch_metrics.items()}
            assert len(avg_metrics.keys()) > 0

            avg_loss.update(loss.item(), input.size(0))
            avg_accuracy.update(accuracy.item(), input.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if self.fast_mode:
                break

        meters = [batch_time, data_time, cuda_time, avg_loss, avg_accuracy]
        meter_str = "\t".join([str(meter) for meter in meters])
        print(f"Epoch: {self.epochs}\t{meter_str}")

        if self.use_lr_scheduling:
            self.scheduler.step()
        self.train_logger.log(avg_loss.avg, "loss")
        self.train_logger.log(avg_accuracy.avg, "accuracy")
        for key, avg_metric in avg_metrics.items():
            self.train_logger.log(avg_metric.avg, key=key)

        return {
            "train_loss": avg_loss.avg,
            "train_accuracy": avg_accuracy.avg,
            **{"train_"+k: avg_metrics[k].avg for k in avg_metrics.keys()}
        }

    def evaluate(self, print_paths=False):
        avg_metrics = defaultdict(AverageMeter)

        self.model.eval()

        iterator = self.test_iterator.get_epoch_iterator(batch_size=self.batch_size, number_of_epochs=1, shuffle=False, device='cuda' if self.use_cuda else 'cpu', preload=self.preload_batch)

        for i, data in enumerate(iterator):
            input, true_path, true_weights = (
                data["images"].contiguous(),
                data["labels"].contiguous(),
                data["true_weights"].contiguous(),
            )

            if self.use_cuda:
                input = input.cuda()
                true_path = true_path.cuda()

            loss, accuracy, last_suggestion = self.forward_pass(input, true_path, train=False, i=i)
            suggested_path = last_suggestion["suggested_path"]
            if print_paths:
                torch.set_printoptions(profile="full")
                print("test", i)
                print(true_path)
                print(suggested_path)
            data.update(last_suggestion)
            if i == 0:
                indices_in_batch = random.sample(range(self.batch_size), 4)
                for num, k in enumerate(indices_in_batch):
                    self.log(data, train=False, k=k, num=num)

            evaluated_metrics = metrics.compute_metrics(true_paths=true_path,
            suggested_paths=suggested_path, true_vertex_costs=true_weights)
            avg_metrics["loss"].update(loss.item(), input.size(0))
            avg_metrics["accuracy"].update(accuracy.item(), input.size(0))
            for key, value in evaluated_metrics.items():
                avg_metrics[key].update(value, input.size(0))

            if self.fast_mode:
                break

        for key, avg_metric in avg_metrics.items():
            self.val_logger.log(avg_metric.avg, key=key)
        avg_metrics_values = dict([(key, avg_metric.avg) for key, avg_metric in avg_metrics.items()])
        return avg_metrics_values

    @abstractmethod
    def build_model(self, **kwargs):
        pass

    @abstractmethod
    def forward_pass(self, input, true_shortest_paths, train, i):
        pass

    def log(self, data, train, k=None, num=None):
        logger = self.train_logger if train else self.val_logger
        if not train:
            image = self.metadata['denormalize'](data["images"][k]).squeeze().astype(np.uint8)
            suggested_path = data["suggested_path"][k].squeeze()
            labels = data["labels"][k].squeeze()

            suggested_path_im = torch.ones((3, *suggested_path.shape))*255*suggested_path.cpu()
            labels_im = torch.ones((3, *labels.shape))*255*labels.cpu()
            image_with_path = draw_paths_on_image(image=image, true_path=labels, suggested_path=suggested_path, scaling_factor=10)

            logger.log(labels_im.data.numpy().astype(np.uint8), key=f"shortest_path_{num}", data_type="image")
            logger.log(suggested_path_im.data.numpy().astype(np.uint8), key=f"suggested_path_{num}", data_type="image")
            logger.log(image_with_path, key=f"full_input_with_path{num}", data_type="image")


class BaselineTrainer(ShortestPathAbstractTrainer):
    def build_model(self, model_name, arch_params):
        grid_dim = int(np.sqrt(self.metadata["output_features"]))
        self.model = get_model(
            model_name, out_features=self.metadata["output_features"], in_channels=self.metadata["num_channels"], arch_params=arch_params
        )

    def forward_pass(self, input, label, train, i):
        output = self.model(input)
        output = torch.sigmoid(output)
        outputu = torch.unbind(output, axis=1)
        flat_target = label.view(label.size()[0], -1)

        # Cross-Entropy
        criterion = torch.nn.BCELoss()
        loss = criterion(output, flat_target).mean()


        if train:
            # Semantic Loss 
            semantic_loss, entropy = sl_6x6_tiles(outputu)

            loss += self.sl_weight * semantic_loss
            loss += self.entreg_weight * entropy
            print("loss", loss, "semantic loss", semantic_loss*self.sl_weight, "entropy", entropy*self.entreg_weight)

        accuracy = (output.round() * flat_target).sum() / flat_target.sum()

        suggested_path = output.view(label.shape).round()
        last_suggestion = {"vertex_costs": None, "suggested_path": suggested_path}

        return loss, accuracy, last_suggestion
