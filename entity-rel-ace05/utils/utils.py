import argparse
import torch
import numpy as np

# Saving heatmaps to tensorboard
import matplotlib.pyplot as plt

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset/ace05')
    parser.add_argument('--vocab_dir', type=str, default='dataset/vocab')
    parser.add_argument('--no_cached_feats', default=False, action='store_true')

    # Dimensions used by the attention
    parser.add_argument('--emb_dim', type=int, default=768, help='Word embedding dimension.')
    parser.add_argument('--ner_dim', type=int, default=30, help='NER embedding dimension.')
    parser.add_argument('--pos_dim', type=int, default=30, help='POS embedding dimension.')

    parser.add_argument('--hidden_dim', type=int, default=200, help='RNN hidden state size.')
    parser.add_argument('--num_layers', type=int, default=2, help='Num of RNN layers.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Input and RNN dropout rate.')
    parser.add_argument('--word_dropout', type=float, default=0.04, help='The rate at which randomly set a word to UNK.')
    parser.add_argument('--topn', type=int, default=1e10, help='Only finetune top N embeddings.')
    parser.add_argument('--lower', dest='lower', action='store_true', help='Lowercase all words.')
    parser.add_argument('--no-lower', dest='lower', action='store_false')
    parser.set_defaults(lower=False)

    parser.add_argument('--attn', dest='attn', action='store_true', help='Use attention layer.')
    parser.add_argument('--no-attn', dest='attn', action='store_false')
    parser.set_defaults(attn=True)
    parser.add_argument('--attn_dim', type=int, default=200, help='Attention size.')
    parser.add_argument('--pe_dim', type=int, default=30, help='Position encoding dimension.')

    
    parser.add_argument('--lr', type=float, default=1.0, help='Applies to SGD and Adagrad.')
    parser.add_argument('--lr_decay', type=float, default=0.9)
    parser.add_argument('--optim', type=str, default='sgd', help='sgd, adagrad, adam or adamax.')
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
    parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
    parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
    parser.add_argument('--save_epoch', type=int, default=5, help='Save model checkpoints every k epochs.')
    parser.add_argument('--save_dir', type=str, default='./saved_models/', help='Root dir for saving models.')
    parser.add_argument('--id', type=str, default='00', help='Model ID under which to save models.')
    parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')

    parser.add_argument('--num_samples', type=int)
    parser.add_argument('--semantic_weight', type=float, default=0.0)
    parser.add_argument('--entropy_weight', type=float, default=0.0)
    parser.add_argument('--tnorm_weight', type=float, default=0.0)
    parser.add_argument('--rel_weight', type=float, default=1.0)
    parser.add_argument('--ner_weight', type=float, default=1.0)
    parser.add_argument('--ner_labels', default=False, action='store_true')
    parser.add_argument('--finetune', type=str, default=None)
    parser.add_argument('--ilp', default=False, action='store_true')
    parser.add_argument('--transductive', default=False, action='store_true')
    parser.add_argument('--THRESHOLD', type =int, default=None)


    parser.add_argument('--gpu_id', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
    args = parser.parse_args()

    return args
