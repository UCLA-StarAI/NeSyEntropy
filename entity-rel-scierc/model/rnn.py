"""
A rnn model for relation extraction, written in pytorch.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import constant, torch_utils
from model import layers

# For Semantic Loss
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../pypsdd')
from compute_mpe import CircuitMPE

# Early Stopping
from utils.early_stopping import EarlyStopping

id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()])
id2ner = dict([(v,k) for k,v in constant.NER_TO_ID.items()])
np.set_printoptions(precision=8)


from tnorm.constraint import ACE05Constraint

class RelationModel(object):
    """ A wrapper class for the training and evaluation of models. """

    def __init__(self, opt, emb_matrix=None):
        
        # Set options, model, loss and parameters
        self.opt = opt
        self.model = PositionAwareRNN(opt, emb_matrix)
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.cmpe = CircuitMPE('dataset/sciERC/disj_constraint.vtree', 'dataset/sciERC/disj_constraint.sdd')
        self.early_stopping  = EarlyStopping(patience=20)

        # Check if we're using the GPU
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()

        # Set the optimizer we're using
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])
        self.scheduler = ReduceLROnPlateau(self.optimizer, min_lr=0.0001, mode='max', verbose='True', patience=10, factor=0.1)
        
        # check constraints once for entropy regularization
        # if opt['entreg_weight']:
        self.sat_idxs = self.get_satisfying_indexes()

    def tnorm_loss(self, logits, ner_logits, masks, subj_idx, obj_idx, batch_size):

        # We need to retrieve the indices of the objects
        # this is accomplished by adding the PAD masks
        # to each of the subj_idx and obj_idx, where we
        # end up with a non-zero tensor except at the index
        # of the subject and object respectively.

        subj_idx = subj_idx + masks.long()
        subj_idx = (subj_idx==0).nonzero()[:,1]

        obj_idx = obj_idx + masks.long()
        obj_idx = (obj_idx==0).nonzero()[:,1]

        relation_probs = F.softmax(logits, -1)
        ner_probs = F.softmax(ner_logits, -1).view(batch_size, -1, len(constant.NER_TO_ID))

        # Separate subject and object probabilities as they are
        # treated as seperate propositional variables by the constraint
        subj_probs = ner_probs[np.arange(batch_size), subj_idx]
        obj_probs = ner_probs[np.arange(batch_size), obj_idx]

        return ACE05Constraint(relation_probs, subj_probs, obj_probs) 


    def semantic_loss(self, logits, ner_logits, masks, subj_idx, obj_idx, batch_size):

        # We need to retrieve the indices of the objects
        # this is accomplished by adding the PAD masks
        # to each of the subj_idx and obj_idx, where we
        # end up with a non-zero tensor except at the index
        # of the subject and object respectively.
        subj_idx = subj_idx + masks.long()
        subj_idx = (subj_idx==0).nonzero()[:,1]

        obj_idx = obj_idx + masks.long()
        obj_idx = (obj_idx==0).nonzero()[:,1]

        # Semantic loss makes use of probabilities
        # so we need to normalize our logits
        relation_probs = F.softmax(logits, -1)

        ner_probs = F.softmax(ner_logits, -1).view(batch_size, -1, len(constant.NER_TO_ID))

        # Separate subject and object probabilities as they are
        # treated as seperate propositional variables by the constraint
        subj_probs = ner_probs[np.arange(batch_size), subj_idx]
        obj_probs = ner_probs[np.arange(batch_size), obj_idx]

        # Concatenate the probabilities in a manner which respects
        # the ordering (used by semantic loss for assigning values to
        # the variables). P.S: We do not have a constraint for relation
        # 'NA'
        probs = torch.cat((relation_probs[:, 1:], subj_probs, obj_probs), dim=1)
        probs = torch.unbind(probs, dim=1)

        # Create CircuitMPE instance for our predictions
        wmc = self.cmpe.get_tf_ac([[1.0 - p,p] for p in probs])
        wmc.clamp_(min=0.0005, max=1.0)

        return -torch.log(wmc)

    def all_prod(self, t1, t2):
        # Returns vector of products of all possible pairs of elements from t1 and t2
        # t1 and t2 should be 1 dimensional tensors
        return torch.matmul(t1.view(-1,1),t2.view(1,-1)).flatten()

    def get_satisfying_indexes(self):
        print("Finding satisfying indexes")
        kb = np.load('dataset/sciERC/kb.npy', allow_pickle=True).item()
        # kb = np.load('dataset/ace05/kb.npy', allow_pickle=True).item()
        n_rels = len(constant.LABEL_TO_ID)
        n_ners = len(constant.NER_TO_ID)
        idxs = [] 
        for i in range(n_rels):
            for j in range(n_ners):
                for k in range(n_ners):
                    rel = id2label[i]
                    entity_str = '(' + id2ner[j] + str(1) + ' & ' + id2ner[k] + str(2) + ')'
                    if rel in kb and entity_str in kb[rel]:
                        idx = i*n_ners**2 + j*n_ners + k
                        idxs.append( idx )
        print("Found %d"%(len(idxs)))
        return idxs

    def categorical_semantic_loss(self, logits, ner_logits, masks, subj_idx, obj_idx, batch_size):
        subj_idx = subj_idx + masks.long()
        subj_idx = (subj_idx==0).nonzero()[:,1]

        obj_idx = obj_idx + masks.long()
        obj_idx = (obj_idx==0).nonzero()[:,1]

        relation_probs = F.softmax(logits, -1)
        num_ners = len(constant.NER_TO_ID)
        ner_probs = F.softmax(ner_logits, -1).view(batch_size, -1, num_ners)

        subj_probs = ner_probs[np.arange(batch_size), subj_idx]
        obj_probs = ner_probs[np.arange(batch_size), obj_idx]

        cat_sl = torch.zeros(batch_size)
        if self.opt['cuda']:
            cat_sl = cat_sl.cuda()
            
        for i in range(batch_size):
            # index goes rel, subj, obj
            joint_dist = self.all_prod(relation_probs[i,:],self.all_prod(subj_probs[i,:],obj_probs[i,:]))
            # ent_reg_loss -= torch.sum(joint_dist * torch.log(joint_dist))
            sat_joint_dist = joint_dist[self.sat_idxs]
            cat_sl[i] = torch.sum(sat_joint_dist)
        
        return -torch.log(cat_sl)

    def entropy_regularization_loss(self, logits, ner_logits, masks, subj_idx, obj_idx, batch_size):
        # old brute force entropy regularization loss 
        subj_idx = subj_idx + masks.long()
        subj_idx = (subj_idx==0).nonzero()[:,1]

        obj_idx = obj_idx + masks.long()
        obj_idx = (obj_idx==0).nonzero()[:,1]

        relation_probs = F.softmax(logits, -1)
        num_ners = len(constant.NER_TO_ID)
        ner_probs = F.softmax(ner_logits, -1).view(batch_size, -1, num_ners)

        subj_probs = ner_probs[np.arange(batch_size), subj_idx]
        obj_probs = ner_probs[np.arange(batch_size), obj_idx]

        ent_reg_loss = torch.tensor(0.0)
        if self.opt['cuda']:
            ent_reg_loss = ent_reg_loss.cuda()
            
        for i in range(batch_size):
            # index goes rel, subj, obj
            joint_dist = self.all_prod(relation_probs[i,:],self.all_prod(subj_probs[i,:],obj_probs[i,:]))
            # sat_joint_dist = joint_dist[self.sat_idxs]
            # sat_joint_dist /= torch.sum(sat_joint_dist)
            # ent_reg_loss += torch.distributions.Categorical(probs=sat_joint_dist).entropy()
            ent_reg_loss += torch.distributions.Categorical(probs=joint_dist).entropy()
        
        return ent_reg_loss

    def entropy_bf(self, logits, ner_logits, masks, subj_idx, obj_idx, batch_size):

        # We need to retrieve the indices of the objects
        # this is accomplished by adding the PAD masks
        # to each of the subj_idx and obj_idx, where we
        # end up with a non-zero tensor except at the index
        # of the subject and object respectively.
        subj_idx = subj_idx + masks.long()
        subj_idx = (subj_idx==0).nonzero()[:,1]

        obj_idx = obj_idx + masks.long()
        obj_idx = (obj_idx==0).nonzero()[:,1]

        # Semantic loss makes use of probabilities
        # so we need to normalize our logits
        relation_probs = F.softmax(logits, -1)

        ner_probs = F.softmax(ner_logits, -1).view(batch_size, -1, len(constant.NER_TO_ID))

        # Separate subject and object probabilities as they are
        # treated as seperate propositional variables by the constraint
        subj_probs = ner_probs[np.arange(batch_size), subj_idx]
        obj_probs = ner_probs[np.arange(batch_size), obj_idx]

        # Concatenate the probabilities in a manner which respects
        # the ordering (used by semantic loss for assigning values to
        # the variables). P.S: We do not have a constraint for relation
        # 'NA'
        probs = torch.cat((relation_probs[:, 1:], subj_probs, obj_probs), dim=1)
        probs = torch.unbind(probs, dim=1)

        # Weighted Model Count
        wmc = self.cmpe.get_tf_ac([[1.0 - p,p] for p in probs]).log()

        # literal weights
        literal_weights = [[(1.0 - p).clamp(min=1e-16).log(), p.clamp(min=1e-16).log()] for p in probs]

        # Create CircuitMPE instance for our predictions
        # mode is of the form: [0011..., 0111..., 1011..., ...]
        models = self.cmpe.get_models()

        # Calculate the probability of each of the models
        models_probs = []
        for model in models:

            # Aggregate the probability of a single model
            model_prob = 0
            for i in range(1, len(model) + 1):
                model_prob += literal_weights[i - 1][model[i]]

            # Aggregate the probability of the current model
            # (normalized by the weighted model count)
            models_probs += [model_prob - wmc]

        models_probs = torch.stack(models_probs) #shape: models x batch_size
        entropy = -(models_probs.exp() * models_probs).sum(dim=0) #shape: batch_size
        return entropy

    def entropy(self, logits, ner_logits, masks, subj_idx, obj_idx, batch_size):

        # We need to retrieve the indices of the objects
        # this is accomplished by adding the PAD masks
        # to each of the subj_idx and obj_idx, where we
        # end up with a non-zero tensor except at the index
        # of the subject and object respectively.
        subj_idx = subj_idx + masks.long()
        subj_idx = (subj_idx==0).nonzero()[:,1]

        obj_idx = obj_idx + masks.long()
        obj_idx = (obj_idx==0).nonzero()[:,1]

        # Semantic loss makes use of probabilities
        # so we need to normalize our logits
        relation_probs = F.softmax(logits, -1)

        ner_probs = F.softmax(ner_logits, -1).view(batch_size, -1, len(constant.NER_TO_ID))

        # Separate subject and object probabilities as they are
        # treated as seperate propositional variables by the constraint
        subj_probs = ner_probs[np.arange(batch_size), subj_idx]
        obj_probs = ner_probs[np.arange(batch_size), obj_idx]

        # Concatenate the probabilities in a manner which respects
        # the ordering (used by semantic loss for assigning values to
        # the variables). P.S: We do not have a constraint for relation
        # 'NA'
        probs = torch.cat((relation_probs[:, 1:], subj_probs, obj_probs), dim=1)
        probs = torch.unbind(probs, dim=1)

        # Create CircuitMPE instance for our predictions
        # print([[1.0 - p,p] for p in probs])
        # print(torch.tensor([[1.0 - p,p] for p in probs]))
        wmc = self.cmpe.get_torch_ac([[1.0 - p,p] for p in probs])
        entropy = self.cmpe.Shannon_entropy()
        
        return entropy

    def entropy_stable(self, logits, ner_logits, masks, subj_idx, obj_idx, batch_size):

        # We need to retrieve the indices of the objects
        # this is accomplished by adding the PAD masks
        # to each of the subj_idx and obj_idx, where we
        # end up with a non-zero tensor except at the index
        # of the subject and object respectively.
        subj_idx = subj_idx + masks.long()
        subj_idx = (subj_idx==0).nonzero()[:,1]

        obj_idx = obj_idx + masks.long()
        obj_idx = (obj_idx==0).nonzero()[:,1]

        # Semantic loss makes use of probabilities
        # so we need to normalize our logits
        relation_probs = F.softmax(logits, -1)

        ner_probs = F.softmax(ner_logits, -1).view(batch_size, -1, len(constant.NER_TO_ID))

        # Separate subject and object probabilities as they are
        # treated as seperate propositional variables by the constraint
        subj_probs = ner_probs[np.arange(batch_size), subj_idx]
        obj_probs = ner_probs[np.arange(batch_size), obj_idx]

        # Concatenate the probabilities in a manner which respects
        # the ordering (used by semantic loss for assigning values to
        # the variables). P.S: We do not have a constraint for relation
        # 'NA'
        probs = torch.cat((relation_probs[:, 1:], subj_probs, obj_probs), dim=1)
        probs = torch.unbind(probs, dim=1)

        # Create CircuitMPE instance for our predictions
        wmc = self.cmpe.get_tf_ac([[1.0 - p,p] for p in probs])
        self.cmpe.generate_torch_ac_stable()
        entropy = self.cmpe.Shannon_entropy_stable()

        return entropy

    def entropy_kld(self, logits, ner_logits, masks, subj_idx, obj_idx, batch_size):

        # We need to retrieve the indices of the objects
        # this is accomplished by adding the PAD masks
        # to each of the subj_idx and obj_idx, where we
        # end up with a non-zero tensor except at the index
        # of the subject and object respectively.
        subj_idx = subj_idx + masks.long()
        subj_idx = (subj_idx==0).nonzero()[:,1]

        obj_idx = obj_idx + masks.long()
        obj_idx = (obj_idx==0).nonzero()[:,1]

        # Semantic loss makes use of probabilities
        # so we need to normalize our logits
        relation_probs = F.softmax(logits, -1)

        ner_probs = F.softmax(ner_logits, -1).view(batch_size, -1, len(constant.NER_TO_ID))

        # Separate subject and object probabilities as they are
        # treated as seperate propositional variables by the constraint
        subj_probs = ner_probs[np.arange(batch_size), subj_idx]
        obj_probs = ner_probs[np.arange(batch_size), obj_idx]

        # Concatenate the probabilities in a manner which respects
        # the ordering (used by semantic loss for assigning values to
        # the variables). P.S: We do not have a constraint for relation
        # 'NA'
        probs = torch.cat((relation_probs[:, 1:], subj_probs, obj_probs), dim=1)
        probs = torch.unbind(probs, dim=1)

        # Create CircuitMPE instance for our predictions
        entropy = []
        for i in range(batch_size):
            wmc = self.cmpe.get_norm_ac([[1.0 - p[i].item(), p[i].item()] for p in probs])
            entropy += [self.cmpe.entropy_kld()]

        return entropy

    def renyi_entropy(self, logits, ner_logits, masks, subj_idx, obj_idx, batch_size, Q=2):

        # We need to retrieve the indices of the objects
        # this is accomplished by adding the PAD masks
        # to each of the subj_idx and obj_idx, where we
        # end up with a non-zero tensor except at the index
        # of the subject and object respectively.
        subj_idx = subj_idx + masks.long()
        subj_idx = (subj_idx==0).nonzero()[:,1]

        obj_idx = obj_idx + masks.long()
        obj_idx = (obj_idx==0).nonzero()[:,1]

        # Semantic loss makes use of probabilities
        # so we need to normalize our logits
        relation_probs = F.softmax(logits.clamp(min=1e-16), -1)

        ner_probs = F.softmax(ner_logits.clamp(min=1e-16), -1).view(batch_size, -1, len(constant.NER_TO_ID))

        # Separate subject and object probabilities as they are
        # treated as seperate propositional variables by the constraint
        subj_probs = ner_probs[np.arange(batch_size), subj_idx]
        obj_probs = ner_probs[np.arange(batch_size), obj_idx]

        # Concatenate the probabilities in a manner which respects
        # the ordering (used by semantic loss for assigning values to
        # the variables). P.S: We do not have a constraint for relation
        # 'NA'
        probs = torch.cat((relation_probs[:, 1:], subj_probs, obj_probs), dim=1)
        probs = torch.unbind(probs, dim=1)

        # Create CircuitMPE instance for our predictions
        wmc = self.cmpe.get_torch_ac([[1.0 - p,p] for p in probs])
        entropy = self.cmpe.Renyi_entropy(Q)

        return entropy

    def update(self, batch, epoch, unsort=True):
        """ Run a step of forward and backward model update. """

        #torch.autograd.set_detect_anomaly(True) 

        # Unpack the batch into labelled and unlabelled
        l_batch, ul_batch = batch

        # If we're using the GPU, copy the batch and labels to the GPU
        if self.opt['cuda']:
            l_inputs = [b.cuda() for b in l_batch[:7]]
            labels = l_batch[7].cuda()

            ul_inputs = [b.cuda() for b in ul_batch[:7]]

        else:
            l_inputs = [b for b in l_batch[:7]]
            labels = l_batch[7]

            ul_inputs = [b for b in ul_batch[:7]]

        # Initialize Losses
        rel_loss = torch.tensor(0)
        semantic_loss = torch.tensor(0)
        ner_loss = torch.tensor(0)

        # Set the model to be in training mode, and zero out the gradient
        self.model.train()
        self.optimizer.zero_grad()

        # ----- Labelled Data Handling -----
        # Do the forward pass on the labelled inputs
        logits, _, ner_logits_l, subj_idx, obj_idx = self.model(l_inputs)

        # Calculate the relation loss
        rel_loss = self.criterion(logits, labels) 

        # Calculate the named-entity recognition loss:
        # a. We flatten the logits (batch_size*max_len, num_classes) 
        # and labels (batch_size*max_len)
        ner_logits_l = ner_logits_l.view(-1, len(constant.NER_TO_ID))
        ner_labels = l_inputs[3].view(-1)
        masks = ~l_inputs[1].view(-1)
    
        # b. Get the valid indices i.e. non-PAD tokens
        valid_indices = torch.nonzero(masks)
        valid_indices = valid_indices.flatten()


        # c. Select only the logits/labels for the valid tokens
        ner_logits_l = ner_logits_l[valid_indices]
        ner_labels = ner_labels[valid_indices]
        
        ner_loss = self.criterion(ner_logits_l, ner_labels)

        # ----- Unlabelled Data Handling -----

        # Do the forward pass on the unlabelled inputs
        logits, _, ner_logits_ul, subj_idx, obj_idx = self.model(ul_inputs)

        # Record the current batch size
        batch_size = ner_logits_ul.shape[0]

        # Calculate constraint losses
        semantic_loss = self.semantic_loss(logits, ner_logits_ul,
                ul_inputs[1], subj_idx, obj_idx, batch_size)
        # semantic_loss = self.categorical_semantic_loss(logits, ner_logits_ul,
        #         ul_inputs[1], subj_idx, obj_idx, batch_size)

        # Aggregate different losses
        loss = self.opt['semantic_weight'] * semantic_loss.sum()+\
                self.opt['rel_weight'] * rel_loss.sum() + self.opt['ner_weight'] * ner_loss.sum()

        if self.opt['tnorm_weight']:
            tnorm_loss = self.tnorm_loss(logits, ner_logits_ul,
                    ul_inputs[1], subj_idx, obj_idx, batch_size)
            loss = loss + self.opt['tnorm_weight'] * tnorm_loss.sum()

        if self.opt['entreg_weight']:
            # entreg_loss = self.renyi_entropy(logits, ner_logits_ul,
            #         ul_inputs[1], subj_idx, obj_idx, batch_size).sum()
            # entreg_loss = self.entropy_regularization_loss(logits, ner_logits_ul,
            #         ul_inputs[1], subj_idx, obj_idx, batch_size)
            entreg_loss = self.entropy(logits, ner_logits_ul,
                    ul_inputs[1], subj_idx, obj_idx, batch_size).sum()
            loss = loss + self.opt['entreg_weight'] * entreg_loss

            # Uncomment the below block to make sure that the entropies calculated different match
            #print("############################ Begin entropy calculation ############################:")
            #entropy = self.entropy(logits, ner_logits_ul, ul_inputs[1], subj_idx, obj_idx, batch_size)
            #print("entropy:", entropy.sum())
            #
            #with torch.no_grad():
            #    entropy_stable = self.entropy_stable(logits, ner_logits_ul, ul_inputs[1], subj_idx, obj_idx, batch_size)
            #    print("entropy_stable:", entropy_stable.sum())

            #with torch.no_grad():
            #    entropy_kld = self.entropy_kld(logits, ner_logits_ul, ul_inputs[1], subj_idx, obj_idx, batch_size)
            #    print("entropy_kld:", sum(entropy_kld))

            #with torch.no_grad():
            #    entropy_bf = self.entropy_bf(logits, ner_logits_ul, ul_inputs[1], subj_idx, obj_idx, batch_size)
            #    print("entropy_bf:", entropy_bf.sum())
            #print("############################ End entropy calculation ############################:")

        # backward
        loss.backward()

        # Assert no nans
        params = [p for p in self.model.parameters() if p.grad is not None]
        for p in params:
            assert(~torch.isnan(p.grad).any())

        # clip lstm gradient
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])

        # Take one step in the direction of the gradient
        self.optimizer.step()

        return rel_loss.detach().clone().mean(), semantic_loss.detach().clone().mean()

    def predict(self, batch, unsort=True, evaluate=False):
        """ Run forward prediction. If unsort is True, recover the original order of the batch. """

        # If we're using the GPU, copy the batch and labels to the GPU
        if self.opt['cuda']:
            inputs = [b.cuda() for b in batch[:7]]
            labels = batch[7].cuda()
        else:
            inputs = [b for b in batch[:7]]
            labels = batch[7]

        orig_idx = batch[8]

        # forward
        self.model.eval()

        # Run a forward pass to generate the predictions
        logits, _, ner_logits, subj_idx, obj_idx = self.model(inputs)

        # Record current batch size
        batch_size = ner_logits.shape[0]

        # Initialize losses
        rel_loss = torch.tensor(0)
        semantic_loss = torch.tensor(0)
        ner_loss = torch.tensor(0)

        # Calculate the loss
        rel_loss = self.criterion(logits, labels) 
        semantic_loss = self.semantic_loss(logits, ner_logits, inputs[1], subj_idx, obj_idx, batch_size)

        # Calculate the named-entity recognition loss:
        # a. We flatten the logits (batch_size*max_len, num_classes) 
        # and labels (batch_size*max_len)
        ner_logits_l = ner_logits.view(-1, len(constant.NER_TO_ID))
        ner_labels = inputs[3].view(-1)
        masks = ~inputs[1].view(-1)
    
        # b. Get the valid indices i.e. non-PAD tokens
        valid_indices = torch.nonzero(masks)
        valid_indices = valid_indices.flatten()

        # c. Select only the logits/labels for the valid tokens
        ner_logits_l = ner_logits_l[valid_indices]
        ner_labels = ner_labels[valid_indices]

        ner_loss = self.criterion(ner_logits_l, ner_labels)

        # Apply softmax to obtain class-probabilities and final predictions
        probs = F.softmax(logits).data.cpu().numpy().tolist()
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()

        ############ Probs ##########
        
        # We need to retrieve the indices of the objects
        # this is accomplished by adding the PAD masks
        # to each of the subj_idx and obj_idx, where we
        # end up with a non-zero tensor except at the index
        # of the subject and object respectively.
        masks = inputs[1]
        subj_idx = subj_idx + masks.long()
        subj_idx = (subj_idx==0).nonzero()[:,1]

        obj_idx = obj_idx + masks.long()
        obj_idx = (obj_idx==0).nonzero()[:,1]

        # Semantic loss makes use of probabilities
        # so we need to normalize our logits
        relation_probs = F.softmax(logits, -1)

        if self.opt['ner_labels']:
            ner_probs = ner_logits.view(batch_size, -1, len(constant.NER_TO_ID))
        else:
            ner_probs = F.softmax(ner_logits, -1).view(batch_size, -1, len(constant.NER_TO_ID))

        # Separate subject and object probabilities as they are
        # treated as seperate propositional variables by the constraint
        subj_probs = ner_probs[np.arange(batch_size), subj_idx].tolist()
        obj_probs = ner_probs[np.arange(batch_size), obj_idx].tolist()

        ########### Probs ############

        # We're not only interested in the subj/obj predictions
        # but also the carry over to named entity recognition in general
        ner_predictions = torch.argmax(ner_probs.detach().clone().view(-1, len(constant.NER_TO_ID)), axis=1)
        ner_predictions = ner_predictions[valid_indices].tolist()
        ner_labels = ner_labels.tolist()

        i = 0
        tmp_predictions = []
        tmp_labels = []
        for mask in masks:
            l = (mask == 0).sum().item()
            tmp_predictions += [ner_predictions[i: i + l]]
            tmp_labels += [ner_labels[i: i + l]]
            i += l

        ner_predictions = tmp_predictions
        ner_labels = tmp_labels

        subj_idx = subj_idx.tolist()
        obj_idx = obj_idx.tolist()

        # We restore data to its original order for evaluation
        if unsort:
            _, predictions, probs, subj_probs, obj_probs, ner_predictions, ner_labels, subj_idx, obj_idx =\
                    [list(t) for t in zip(*sorted(zip(orig_idx, predictions, probs, subj_probs, obj_probs,\
                    ner_predictions, ner_labels, subj_idx, obj_idx)))]

        if evaluate:
            return probs, subj_probs, obj_probs, predictions, ner_predictions, ner_labels, subj_idx, obj_idx
        else:
            return predictions, ner_predictions, ner_labels, subj_idx, obj_idx,\
                    [rel_loss.detach().clone().mean()], [semantic_loss.detach().clone().mean()], [ner_loss.detach().clone().mean()]

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def save(self, filename, epoch):
        params = {
                'model': self.model.state_dict(),
                'config': self.opt,
                'epoch': epoch
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
            self.model.load_state_dict(checkpoint['model'])

        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()

class PositionAwareRNN(nn.Module):
    """ A sequence model for relation extraction. """

    def __init__(self, opt, emb_matrix=None):
        super(PositionAwareRNN, self).__init__()

        # Dropout Layer
        self.drop = nn.Dropout(opt['dropout'])

        # An embedding layer for the sentence tokens
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)

        # An embedding layer for the POS tags
        if opt['pos_dim'] > 0:
            self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim'],
                    padding_idx=constant.PAD_ID)

        # An embedding layer for the ner tags
        if opt['ner_dim'] > 0:
            self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim'],
                    padding_idx=constant.PAD_ID)

        # Input dimensionalty, which changes depending on whether or not we
        # factor in the POS tags as well as the NER tags
        input_size = opt['emb_dim'] + opt['pos_dim'] + len(constant.NER_TO_ID)  #+ opt['ner_dim']

        # RNN
        self.rnn = nn.LSTM(input_size, opt['hidden_dim'], opt['num_layers'], batch_first=True,\
                dropout=opt['dropout'])

        # If we're using attention
        if opt['attn']:
            self.pe_emb = nn.Embedding(constant.MAX_LEN * 2 + 1, opt['pe_dim'])
            self.attn_layer = layers.PositionAwareAttention(opt['hidden_dim'],
                    opt['hidden_dim'], 2*opt['pe_dim'], opt['attn_dim'])

        # Output layer
        self.linear = nn.Linear(opt['hidden_dim'], opt['num_class'])

        # Set options
        self.opt = opt
        self.topn = self.opt.get('topn', 1e10)
        self.use_cuda = opt['cuda']
        self.emb_matrix = emb_matrix
        self.init_weights()

        # Named-Entity Recognition
        self.NER = NER_Net(opt)
    
    def init_weights(self):

        # Initialize embedding matrix
        if self.emb_matrix is None:
            self.emb.weight.data[1:,:].uniform_(-1.0, 1.0) # keep padding dimension to be 0
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)

        # Initialize POS embeddings
        if self.opt['pos_dim'] > 0:
            self.pos_emb.weight.data[1:,:].uniform_(-1.0, 1.0)

        # Initialize NER embeddings
        if self.opt['ner_dim'] > 0:
            self.ner_emb.weight.data[1:,:].uniform_(-1.0, 1.0)

        # Initialize fully connected layer
        self.linear.bias.data.fill_(0)
        init.xavier_uniform_(self.linear.weight, gain=1)

        # Initialize position embeddings for attention layer
        if self.opt['attn']:
            self.pe_emb.weight.data.uniform_(-1.0, 1.0)

        # Decide finetuning
        if self.topn <= 0:
            print("Do not finetune word embedding layer.")
            self.emb.weight.requires_grad = False

        elif self.topn < self.opt['vocab_size']:
            print("Finetune top {} word embeddings.".format(self.topn))
            self.emb.weight.register_hook(lambda x: \
                    torch_utils.keep_partial_grad(x, self.topn))

        else:
            print("Finetune all embeddings.")

    def zero_state(self, batch_size): 
        """ Zeros the state of the LSTM """ 

        state_shape = (self.opt['num_layers'], batch_size, self.opt['hidden_dim'])
        h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
        if self.use_cuda:
            return h0.cuda(), c0.cuda()
        else:
            return h0, c0
    
    def forward(self, inputs):
        words, masks, pos, ner, deprel, subj_pos, obj_pos = inputs # unpack
        seq_lens = masks.detach().clone().eq(constant.PAD_ID).long().sum(1)
        batch_size = words.size()[0]  

        if self.opt['ner_labels']:
            ner_shape = ner.shape
            predicted_ner = ner.view(ner_shape[0], ner_shape[1], 1)
            predicted_ner = torch.zeros(ner_shape[0], ner_shape[1], len(constant.NER_TO_ID)).cuda()\
                    .scatter_(2, predicted_ner, 1)
            predicted_ner *= 1000
        else:
            predicted_ner = self.NER(words)
            ner = torch.max(predicted_ner, -1)[1]

            
        word_inputs = words

        inputs = [word_inputs]
        if self.opt['pos_dim'] > 0:
            inputs += [self.pos_emb(pos)]
        if self.opt['ner_dim'] > 0:
            inputs += [predicted_ner]
        inputs = self.drop(torch.cat(inputs, dim=2)) # add dropout to input
        input_size = inputs.size(2)

        # rnn
        h0, c0 = self.zero_state(batch_size)
        inputs = nn.utils.rnn.pack_padded_sequence(inputs, seq_lens, batch_first=True)
        outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
        outputs, output_lens = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        hidden = self.drop(ht[-1,:,:]) # get the outmost layer h_n
        outputs = self.drop(outputs)
        
        # attention
        if True:
            # convert all negative PE numbers to positive indices
            # e.g., -2 -1 0 1 will be mapped to 98 99 100 101
            subj_pe_inputs = self.pe_emb(subj_pos + constant.MAX_LEN)
            obj_pe_inputs = self.pe_emb(obj_pos + constant.MAX_LEN)
            pe_features = torch.cat((subj_pe_inputs, obj_pe_inputs), dim=2)
            final_hidden = self.attn_layer(outputs, masks, hidden, pe_features)
        else:
            final_hidden = hidden

        logits = self.linear(final_hidden)

        return logits, final_hidden, predicted_ner, subj_pos, obj_pos

class NER_Net(nn.Module):

    def __init__(self, opt):
        super(NER_Net, self).__init__()
        self.LSTM_HIDDEN_SIZE = 50

        # maps each token to an embedding_dim vector
        self.embedding = nn.Embedding(opt['vocab_size'], opt['ner_dim'])

        # the LSTM takes as input the dim of its inputs, and the dim
        # of its hidden size
        self.lstm = nn.LSTM(768, self.LSTM_HIDDEN_SIZE, batch_first=True)

        # fc layer transforms the output to give the final output layer
        self.fc = nn.Linear(50, len(constant.NER_TO_ID))

        # Initialize fully connected layer
        #self.fc.bias.data.fill_(0)
        #init.xavier_uniform(self.fc.weight, gain=1)

    def forward(self, s):
        # apply the embedding layer that maps each token to its embedding
        #s = self.embedding(s)   # dim: batch_size x batch_max_len x embedding_dim
                    
        # run the LSTM along the sentences of length batch_max_len
        s, _ = self.lstm(s)     # dim: batch_size x batch_max_len x lstm_hidden_dim                
                            
        # apply the fully connected layer and obtain the output for each token
        s = self.fc(s)          # dim: batch_size*batch_max_len x num_tags

        return s
