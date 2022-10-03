import os
from datetime import datetime
import time
import numpy as np
import random
from shutil import copyfile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
torch.backends.cudnn.enabled = False

from data.loader import DataLoader, CombinedDataLoader
from model.rnn import RelationModel
from utils import scorer, constant, helper, torch_utils 
from utils.vocab import Vocab
from utils.utils import parse_arguments

# Classification Report
from sklearn.metrics import f1_score, classification_report

import eval_script
import itertools

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# A map from class ids to labels
id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()])

def flatten(ll):
        return [val for l in ll for val in l]

# Get the predictions of the current model (or should it be the best model?)
# and satisfy the constraints using ILP
def get_ilp_predictions(model, dataset):
    rel_preds, ner_preds = eval_script.evaluate(None, model=model, dataset=dataset, gpu_id=opt['gpu_id'], ilp=True)
    return rel_preds, ner_preds
    

# Evaluate model on the provided dataset
def evaluate(model, dataset):
    
    # (0        , 1        , 2         , 3       , 4      , 5       , 6            , 7       )
    # (rel_preds, ner_preds, ner_labels, subj_idx, obj_idx, rel_loss, semantic_loss, ner_loss)
    accumulators = [[], [], [], [], [], [], [], []]
    for i, batch in enumerate(dataset):
        for l, el in zip(accumulators, model.predict(batch)): l.extend(el)

    rel_preds, ner_preds, ner_labels, subj_idx, obj_idx, rel_loss, semantic_loss, ner_loss = accumulators
    
    # average losses over batches
    rel_loss = torch.tensor(rel_loss).mean()
    ner_loss = torch.tensor(ner_loss).mean()
    semantic_loss = torch.tensor(semantic_loss).mean()

    # Calculate validation set f1-score
    predictions = np.array([id2label[int(p)] for p in rel_preds])
    rel_f1 = f1_score(dataset.gold(), predictions, labels=np.unique(dataset.gold()[np.where(dataset.gold() != 'NA')]), average='micro')
    ner_f1 = f1_score(flatten(ner_labels), flatten(ner_preds), average='micro')

    # Calculate triples f1-score
    relation_gold = [constant.LABEL_TO_ID[gold] for gold in dataset.gold()]
    predicted_triples = tuple(zip(rel_preds, [sentence[subj_idx[i]] for i, sentence in enumerate(ner_preds)], [sentence[obj_idx[i]] for i, sentence in enumerate(ner_preds)]))
    gt_triples = tuple(zip(relation_gold, [sentence[subj_idx[i]] for i, sentence in enumerate(ner_labels)], [sentence[obj_idx[i]] for i, sentence in enumerate(ner_labels)]))

    predicted_triples = [str(pred) for pred in predicted_triples]
    gt_triples = [str(gt) for gt in gt_triples]
    triples_f1 = f1_score(gt_triples, predicted_triples, average='micro')

    print("NER_loss: ", ner_loss, " Rel_f1: ", rel_f1, " NER_f1: ", ner_f1, " triples_f1:", triples_f1)
    return (rel_f1+ner_f1)/2, (rel_loss, semantic_loss)

# Train model for one epoch on the provided dataset 
def train(model, dataset, opt, epoch):

    # Unpack labelled and unlabelled dataset
    labelled_dataset, unlabelled_dataset = dataset

    # The dataset the greater number of elements determines the number of iterations on our epoch
    dataset = labelled_dataset if len(labelled_dataset) > len(unlabelled_dataset) else unlabelled_dataset
    
    # logging
    format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.1f} * {:.6f}, {:.2f} * {:.6f})'

    # Track training metrics
    losses = torch.Tensor()

    # Iterate over all batches in the training set
    for i in range(len(dataset)):

        # Get the labelled and unlabelled batches
        # there are more unlabelled than lablled batches,
        # and we therefore cycle through the labelled batches
        # i.e. one epoch of unlabelled bat)ches = several epochs
        # of labelled batches
        ul_batch = unlabelled_dataset.__getitem__(i % len(unlabelled_dataset))
        l_batch = labelled_dataset.__getitem__(i % len(labelled_dataset))

        # A training batch is just a concatenation of labelled and unlablled batches
        batch = (l_batch, ul_batch)

        # Update the model through a forward-backward step
        rel_loss, semantic_loss = model.update(batch, epoch)

        # Update training metrics
        losses = torch.cat((losses, torch.tensor([[rel_loss , semantic_loss]], dtype=torch.float)), dim=0)

        # logging
        if i % opt['log_step'] == 0:
            print(format_str.format(datetime.now(), i, len(dataset), epoch, opt['num_epoch'],
                opt['rel_weight'] * rel_loss + opt['semantic_weight'] * semantic_loss,
                opt['rel_weight'],  rel_loss, opt['semantic_weight'],  semantic_loss))


if __name__ == "__main__":

    # Parse Arguments
    args = parse_arguments()
    opt = vars(args)
    
    # Set print options
    torch.set_printoptions(sci_mode=False)

    # idk if I crashed Leffe or something, but I set this just in case :P
    # torch.set_num_threads(8)

    # Set device
    if opt['cpu']:
        opt['cuda'] = False

    elif opt['cuda']:
        torch.cuda.set_device(opt['gpu_id'])

    # Set random seed
    torch.manual_seed(opt['seed'])
    np.random.seed(opt['seed'])
    random.seed(opt['seed'])

    # Set num_class
    opt['num_class'] = len(constant.LABEL_TO_ID)

    # Split the data into labelled/unlabelled
    from data.create_labelled_unlabelled import create_labelled, create_unlabelled, create_ilp
    create_labelled(opt['data_dir'], opt['num_samples'], opt['seed'])
    create_unlabelled(opt['data_dir'], opt['transductive'] and not opt['ilp'])
    create_ilp(opt['data_dir'], opt['transductive'], name='ilp.json' + str(opt['num_samples']))

    # load vocab
    vocab_file = opt['vocab_dir'] + '/vocab.pkl'
    vocab = Vocab(vocab_file, load=True)
    opt['vocab_size'] = vocab.size
    emb_file = opt['vocab_dir'] + '/embedding.npy'
    emb_matrix = np.load(emb_file)
    emb_matrix = None

    # Mine relations from dataset
    #from data.mine_relations_big import mine_relations
    #from data.mine_kb import mine_kb
    #mine_kb(opt['data_dir'], '.')
    #mine_relations(opt['data_dir'], 'model/')

    # load data; under the semi-supervised setting we make use
    # of both labelled as well as unlabelled data, hence the
    # labelled_batch and unlabelled_batch
    print("Loading data from {} with batch size {}...".format(opt['data_dir'], opt['batch_size']))
    labelled_batch = DataLoader(opt['data_dir'] + '/labelled.json', int(opt['batch_size']), opt, vocab, evaluation=False)
    dev_batch = DataLoader(opt['data_dir'] + '/dev.json', opt['batch_size'], opt, vocab, evaluation=True)

    ILP = False
    unlabelled_batch = DataLoader(opt['data_dir'] + '/unlabelled.json', int(opt['batch_size']), opt, vocab, evaluation=False)

    # The available classes are used by scikit-learn to calculate the f1-score
    train_classes = np.unique(labelled_batch.gold()[np.where(labelled_batch.gold() != 'NA')])
    dev_classes = np.unique(dev_batch.gold()[np.where(dev_batch.gold() != 'NA')])

    # The train and test classes should differ from the gold classes only by the NO_RELATION relation
    assert (len(dev_classes) == len(np.unique(dev_batch.gold()))) and (len(train_classes) == len(np.unique(labelled_batch.gold())))

    # Create model directory
    model_id = opt['id'] if len(opt['id']) > 1 else '0' + opt['id']
    model_save_dir = opt['save_dir'] + '/' + model_id
    opt['model_save_dir'] = model_save_dir
    helper.ensure_dir(model_save_dir, verbose=True)

    # save config
    helper.save_config(opt, model_save_dir + '/config.json', verbose=True)
    vocab.save(model_save_dir + '/vocab.pkl')
    file_logger = helper.FileLogger(model_save_dir + '/' + opt['log'], header="#epoch   train_loss  dev_loss    train_f1    dev_f1  t_rel_loss  t_semantic_loss d_rel_loss  d_semantic_loss")

    # print model info
    helper.print_config(opt)

    # model
    if opt['finetune'] != None:
        model_file = opt['finetune']
        opt = torch_utils.load_config(model_file)
        model = RelationModel(opt)
        model.load(model_file)
        dev_f1, dev_losses = evaluate(model, dev_batch)

    else:
        model = RelationModel(opt, emb_matrix=emb_matrix)

    # Set up the trackers
    dev_f1_history = []

    # Initial accuracy on validation set
    epoch = 0

    # Evaluate on train set
    print('################################################## Train ##################################################')
    train_f1, train_losses = evaluate(model, labelled_batch)

    # Evaluate on dev set
    print('################################################## Valid ##################################################')
    dev_f1, dev_losses = evaluate(model, dev_batch)
    
    # Calculate metrics
    train_loss = opt['rel_weight'] * train_losses[0] + opt['semantic_weight'] * train_losses[1]
    dev_loss = opt['rel_weight'] * dev_losses[0] + opt['semantic_weight'] * dev_losses[1]

    # Print current epoch
    print("epoch {}: train_loss = {:.6f} ({:.1f} * {:.6f} + {:.3f} * {:.6f}), dev_loss = {:.6f} ({:.1f} * {:.6f} + {:.3f} * {:.6f}), train_f1 = {:.4f}, dev_f1 = {:.4f}"\
            .format(epoch, train_loss, opt['rel_weight'], train_losses[0], opt['semantic_weight'], train_losses[1],
                dev_loss, opt['rel_weight'], dev_losses[0], opt['semantic_weight'], dev_losses[1],
                train_f1.mean(), dev_f1))

    # Log to file
    file_logger.log("{} {:.6f}  {:.6f}  {:.4f}  {:.4f}  {:.6f}  {:.6f}  {:.6f}  {:.6f}"
            .format(epoch, train_loss, dev_loss, train_f1.mean(),\
            dev_f1, train_losses[0], train_losses[1],\
            dev_losses[0], dev_losses[1]))
    
    print(opt['THRESHOLD'])
    num_ilp = 0
    for epoch in range(1, opt['num_epoch'] + 1):

        # Train for one epoch on the train set
        if opt['ilp'] and epoch >= opt['THRESHOLD']+1:
            train(model, (CombinedDataLoader(labelled_batch, unlabelled_batch), unlabelled_batch), opt, epoch)
        else:
            train(model, (labelled_batch, unlabelled_batch), opt, epoch)

        # Evaluate on train set
        print('################################################## Train ##################################################')
        train_f1, train_losses = evaluate(model, labelled_batch)

        # Evaluate on dev set
        print('################################################## Valid ##################################################')
        dev_f1, dev_losses = evaluate(model, dev_batch)
        
        # Calculate metrics
        train_loss = opt['rel_weight'] * train_losses[0] + opt['semantic_weight'] * train_losses[1]
        dev_loss = opt['rel_weight'] * dev_losses[0] + opt['semantic_weight'] * dev_losses[1]

        # Print current epoch
        print("epoch {}: train_loss = {:.6f} ({:.1f} * {:.6f} + {:.3f} * {:.6f}), dev_loss = {:.6f} ({:.1f} * {:.6f} + {:.3f} * {:.6f}), train_f1 = {:.4f}, dev_f1 = {:.4f}"\
                .format(epoch, train_loss, opt['rel_weight'], train_losses[0], opt['semantic_weight'], train_losses[1],
                    dev_loss, opt['rel_weight'], dev_losses[0], opt['semantic_weight'], dev_losses[1],
                    train_f1.mean(), dev_f1))

        # Log to file
        file_logger.log("{} {:.6f}  {:.6f}  {:.4f}  {:.4f}  {:.6f}  {:.6f}  {:.6f}  {:.6f}"
                .format(epoch, train_loss, dev_loss, train_f1.mean(),\
                dev_f1, train_losses[0], train_losses[1],\
                dev_losses[0], dev_losses[1]))

        # Save every save_epoch epochs
        if epoch % opt['save_epoch'] == 0:
            model_file = model_save_dir + '/checkpoint_epoch_{}.pt'.format(epoch)
            model.save(model_file, epoch)

        # Save best model
        if epoch == 1 or dev_f1 > max(dev_f1_history):

            # Save new best model
            model.save( model_save_dir + '/best_model.pt', epoch)
            print("new best model saved.")

            if opt['ilp'] and epoch >= opt['THRESHOLD']:

                # If we've reached our ILP threshold, break
                if num_ilp >= 5:
                    print("Done with 5 cycles of ILP")
                    break

                # hack for first time we do ilp
                if not ILP:
                    unlabelled_batch = DataLoader(opt['data_dir'] + '/ilp.json' + str(opt['num_samples']), int(opt['batch_size']), opt, vocab, evaluation=False)
                    ILP = True

                # produce pseudo-labels using best_model + ilp
                rel_preds, ner_preds = get_ilp_predictions(model, unlabelled_batch)
                unlabelled_batch.adjust(rel_preds, ner_preds)
                num_ilp += 1
                print("Num ilp", num_ilp)

        # Update validation accuracy history
        dev_f1_history += [dev_f1]

        # Anneal the lr
        model.scheduler.step(dev_f1.mean())

        # Check for nans
        if torch.isnan(dev_loss):
            break

        # Check for early Stopping
        model.early_stopping(dev_f1)
        if model.early_stopping.early_stop:
            print("Early stopping...")
            break

        print("\n")
    print("Training ended with {} epochs. Best accuracy on validation: {}".format(epoch, max(dev_f1_history)))
