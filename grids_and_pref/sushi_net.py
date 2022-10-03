# python sushi_net.py --entropy_circuit --layers 3 --units 25 --iters 20000 --data test.data --wmc 0.25 --entropy_weight 0.01

import argparse
from torch import nn
import torch.nn.functional as F
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../pypsdd')

import numpy as np
from numpy.random import permutation

from sushi_data import SushiData, to_pairwise_comp
from compute_mpe import CircuitMPE

import torch
import random
from torch.distributions.bernoulli import Bernoulli


torch.manual_seed(1234)
random.seed(1234)
np.random.seed(1234)

FLAGS = None

class Net(nn.Module):

    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, FLAGS.units)
        self.fc2 = nn.Linear(FLAGS.units, FLAGS.units)
        self.fc3 = nn.Linear(FLAGS.units, FLAGS.units)
        self.output = nn.Linear(FLAGS.units, output_size) 

    def forward(self, x):
        x = self.fc1(x)
        x = F.sigmoid(x)

        x = self.fc2(x)
        x = F.sigmoid(x)

        x = self.fc3(x)
        x = F.sigmoid(x)

        output = self.output(x)
        return output

def main():

    # Import data
    sushi_data = SushiData('sushi.soc')
    input_size = sushi_data.train_data.shape[1]
    output_size = sushi_data.train_labels.shape[1]


    # Create the model
    model = Net(input_size, output_size).cuda()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=FLAGS.l2_decay)

    # Get supervised part (rest is unsupervised)
    perm = permutation(sushi_data.train_data.shape[0])
    sup_train_inds = perm[:int(sushi_data.train_data.shape[0] * FLAGS.give_labels)]
    unsup_train_inds = perm[int(sushi_data.train_data.shape[0] * FLAGS.give_labels):]

    # Mask out the loss for the unsupervised samples
    ce_weights = torch.zeros([sushi_data.train_data.shape[0], 1]).cuda()
    ce_weights[sup_train_inds, :] = 1

    # Create CircuitMPE instance for predictions
    cmpe = CircuitMPE('permutation-4.vtree', 'permutation-4.sdd')
  
    prev_loss = 1e15
    max_coherent = 0

    # Load data
    X = torch.tensor(sushi_data.train_data).float().cuda()
    y = torch.tensor(sushi_data.train_labels).cuda()

    for i in range(FLAGS.iters):

        # train
        model.train()
        model.zero_grad()


        # Forward
        output = model(X)

        # Cross_entropy loss
        cross_entropy = criterion(output, y) * ce_weights

        # Add semantic loss
        outputu = torch.unbind(torch.sigmoid(output), axis=1)
        wmc = cmpe.get_tf_ac([[1.0 - ny,ny] for ny in outputu])
        semantic_loss = -torch.log(wmc)


        if FLAGS.entropy_all:
            m = Bernoulli(torch.sigmoid(output))
            entropy = m.entropy().sum(dim=-1)
            loss = 1.0 * cross_entropy.sum() + FLAGS.wmc * semantic_loss.sum() + FLAGS.entropy_weight * entropy.sum()

        elif FLAGS.entropy_circuit:
            entropy = cmpe.Shannon_entropy()
            loss = 1.0 * cross_entropy.sum() + FLAGS.wmc * semantic_loss.sum() + FLAGS.entropy_weight * entropy.mean()

        else:
            loss = 1.0 * cross_entropy.sum() + FLAGS.wmc * semantic_loss.sum()

        # Backward and step
        loss.backward()
        optimizer.step()

        # Every 1k iterations check accuracy
        if i % 1 == 0:

            model.eval()
            print("After %d iterations" % i)
            X_valid =  torch.tensor(sushi_data.valid_data).float().cuda()
            y_valid =  torch.LongTensor(sushi_data.valid_labels).cuda()

            valid_out = torch.sigmoid(model(X_valid))
            preds  = valid_out.round().long()

            # Percentage that are exactly right
            exactly_correct = torch.all(preds == y_valid, dim=1)
            print(exactly_correct.sum())
            percent_exactly_correct = exactly_correct.sum().to(dtype=torch.float)/exactly_correct.size(0)
            print("Percentage of validation that are exactly right: %f" % (percent_exactly_correct * 100))
            if max_coherent < percent_exactly_correct * 100:
                max_coherent = percent_exactly_correct * 100
                print("Saving new best model")
                torch.save(model.state_dict(), './best_sl_' + str(FLAGS.wmc) + '.pt')
            print("max so far: ", max_coherent)

            # Percentage of individual labels that are right
            individual_correct = (preds == y_valid).sum()
            percent_individual_correct = individual_correct.to(dtype=torch.float) / len(preds.flatten()) 
            print("Percentage of individual labels in validation that are right: %f" % (percent_individual_correct * 100))

            # Percentage of predictions that satisfy the constraint
            wmc = [cmpe.weighted_model_count([(1-p, p) for p in datum]) for datum in np.array(valid_out.cpu().detach() + 0.5, int)]
            print("Percentage of predictions that satisfy the constraint %f", 100*sum(wmc)/len(wmc))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='test.data',
                      help='Input data file to use')
    parser.add_argument('--units', type=int, default=100,
                      help='Number of units per hidden layer')
    parser.add_argument('--layers', type=int, default=3,
                      help='Number of hidden layers')
    parser.add_argument('--wmc', type=float, default=0.0,
                      help='Coefficient of WMC in loss')
    parser.add_argument('--entropy_weight', type=float, default=0.5,
                      help='Coefficient of entropy in loss')
    parser.add_argument('--iters', type=int, default=10000,
                      help='Number of minibatch steps to do')
    parser.add_argument('--relu', action='store_true',
                      help='Use relu hidden units instead of sigmoid')
    parser.add_argument('--early_stopping', action='store_true',
                      help='Enable early stopping - quit when validation loss is increasing')
    parser.add_argument('--give_labels', type=float, default=1.0,
                      help='Percentage of training examples to use labels for (1.0 = supervised)')
    parser.add_argument('--use_unlabeled', action='store_true',
                      help='Use this flag to enable semi supervised learning with WMC')
    parser.add_argument('--l2_decay', type=float, default=0.0,
                      help='L2 weight decay coefficient')
    parser.add_argument('--entropy_all', default=False, action='store_true',
                      help='Calculate the entropy on the entire output distribution')
    parser.add_argument('--entropy_circuit', default=False, action='store_true',
                      help='Calcualte the entropy on the distribution over the models of the circuit')

    FLAGS, unparsed = parser.parse_known_args()
    main()
