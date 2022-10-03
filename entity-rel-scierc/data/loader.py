"""
Data loader for json files.
"""

import json
import random
import torch
import numpy as np

from utils.get_bert_feats import BERT_FEATURES, process_sents
from utils import constant, helper, vocab
from transformers import BertTokenizer

class CombinedDataLoader(object):
    def __init__(self, d1, d2):
        self.d1 = d1
        self.d2 = d2

    def __len__(self):
        """ Return the length of the data """

        return len(self.d1) + len(self.d2)

    def __getitem__(self, key):
        """ Get a batch with index. """

        if not isinstance(key, int):
            raise TypeError

        if key < 0 or key >= self.__len__():
            raise IndexError

        if key >= 0 and key < len(self.d1):
            return self.d1[key]
        else:
            return self.d2[key - len(self.d1)]

    def __iter__(self, key):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, vocab, evaluation=False, rel_preds=None, ner_preds=None):
        self.batch_size = batch_size
        self.opt = opt
        self.vocab = vocab
        self.eval = evaluation
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',\
                do_lower_case=self.opt['lower'], do_basic_tokenize=False)
        self.bert = BERT_FEATURES() 

        if self.opt['cuda']:
            self.bert.cuda()

        # Read json file creating the data
        with open(filename) as infile:
            data = json.load(infile)

        # Read unlabeled data to be used in ilp
        if rel_preds is not None and ner_preds is not None:
            with open(opt['data_dir'] + '/ilp.json') as infile:
                ilp_data = json.load(infile)

        # preprocess the data
        data = self.preprocess(data, vocab, opt)
        
        # shuffle the data for training
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]

        # Create a dict mapping ids to labels
        id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()])

        # The label is the last field of the datum
        self.labels = np.array([id2label[d[-1]] for d in data])
        self.num_examples = len(data)

        # chunk into batches
        self.data = [data[i: i + batch_size] for i in range(0, len(data), batch_size)]

        # Cache BERT features
        self.bert_feats = []
        for batch in self.data:
            batch_data = list(zip(*batch))
            words = batch_data[0]
            words, masks = get_bert_feats(self.tokenizer, self.bert, 
                    words, lower_case=self.opt['lower'], batch_size=self.opt['batch_size'], cuda=self.opt['cuda'])
            self.bert_feats += [(words, masks)]

        print("{} batches created for {}".format(len(data), filename))

    def preprocess(self, data, vocab, opt):
        """ Preprocess the data and convert to ids. """

        processed = []
        for d in data:
            tokens = d['token']
            
            # Set all tokens to lower-case
            if opt['lower']:
                tokens = [t.lower() for t in tokens]

            # subject/object start and end
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']

            # Anonymize tokens i.e. set subject tokens and object
            # tokens to "SUBJ" + subj_type, and similarly for objects
            #tokens[ss:se+1] = ['SUBJ-'+d['subj_type']] * (se-ss+1)
            #tokens[os:oe+1] = ['OBJ-'+d['obj_type']] * (oe-os+1)

            # Map to ids
            #tokens = map_to_ids(tokens, vocab.word2id)
            pos = map_to_ids(d['stanford_pos'], constant.POS_TO_ID)
            ner = map_to_ids(d['stanford_ner'], constant.NER_TO_ID)
            deprel = map_to_ids(d['stanford_deprel'], constant.DEPREL_TO_ID)
            relation = constant.LABEL_TO_ID[d['relation']]

            l = len(tokens)
            # Encode subject/object positions as zeros
            subj_positions = get_positions(d['subj_start'], d['subj_start'], l)
            obj_positions = get_positions(d['obj_start'], d['obj_start'], l)

            # Add current datum to the list of processed data 
            processed += [(tokens, pos, ner, deprel, subj_positions, obj_positions, relation)]

        return processed

    def gold(self):
        """ Return gold labels as a list. """

        return self.labels

    def __len__(self):
        """ Return the length of the data """

        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """

        if not isinstance(key, int):
            raise TypeError

        if key < 0 or key >= len(self.data):
            raise IndexError
        
        # We group the same field together i.e. before zipping,
        # batch[0] is the first datum, after zipping, batch[0]
        # refers to the tokens of samples in the batch
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 7

        # Sort the data by the length of their tokens
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)
        
        # If we're training, we dropout some tokens and replace them
        # with UNK tokens as a form of regularization
        #if not self.eval:
        #    words = [word_dropout(sent, self.opt['word_dropout']) for sent in batch[0]]
        #else:
        #    words = batch[0]
        
        words = batch[0]

        if self.opt['no_cached_feats']:
            words, masks = get_bert_feats(self.tokenizer, self.bert,
                words, lower_case=self.opt['lower'], batch_size=self.opt['batch_size'])

        else:
            words, masks = self.bert_feats[key]
            words = words[orig_idx]
            masks = masks[orig_idx]
            

        # Pad the tensors so that all examples in the
        # batch have the same size
        pos = get_long_tensor(batch[1], batch_size)
        ner = get_long_tensor(batch[2], batch_size)
        deprel = get_long_tensor(batch[3], batch_size)
        subj_positions = get_long_tensor(batch[4], batch_size)
        obj_positions = get_long_tensor(batch[5], batch_size)
        rels = torch.LongTensor(batch[6])

        # Return
        return (words, masks, pos, ner, deprel, subj_positions, obj_positions, rels, orig_idx)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def adjust(self, rel_predictions, ner_predictions):
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                tokens, pos, ner, deprel, subj_positions, obj_positions, relation = self.data[i][j]
                relation = rel_predictions[i * self.opt['batch_size'] + j]
                ner = ner_predictions[i * self.opt['batch_size'] + j]
                self.data[i][j] = (tokens, pos, ner, deprel, subj_positions, obj_positions, relation)
                

def map_to_ids(tokens, vocab):
    """ Maps a list of tokens to the corresponding ids given by the dictionary, vocab"""
    ids = [vocab[t] for t in tokens]
    return ids

def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence, where the position(s) of the subject/object are marked by 0s """
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + \
            list(range(1, length-end_idx))


def pad_tokens(tokens_list, batch_size):
    token_len = max(len(x) for x in tokens_list)

    tokens, masks = [], [] 
    for i, s in enumerate(tokens_list):
        required_pad = [constant.PAD_TOKEN] * (token_len - len(s))
        tokens.append(s + required_pad)
        masks.append([False] * len(s) + [True] * len(required_pad))
    
    return tokens, masks

def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    
    # Get the length of the longest sentence in the batch
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)

    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)

    return tokens

def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """

    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]

    return sorted_all[2:], sorted_all[1]

def word_dropout(tokens, dropout):
    """ Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
    return [constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout \
            else x for x in tokens]

def get_bert_feats(tokenizer, feat_extractor, sentences, lower_case=True, batch_size=256, cuda=True):

    token_ids, valid_positions, valid_tokens, masks = process_sents(tokenizer, sentences)
    input = torch.tensor(token_ids)
    valid_ids = torch.tensor(valid_positions)
    if cuda:
        input = input.cuda()
        valid_ids = valid_ids.cuda()
    # input = torch.tensor(token_ids).cuda()
    # valid_ids = torch.tensor(valid_positions).cuda()
    masks = torch.BoolTensor(masks)

    word_embeddings, batch_sent_embeddings = feat_extractor(input, valid_ids=valid_ids)

    return word_embeddings, masks
