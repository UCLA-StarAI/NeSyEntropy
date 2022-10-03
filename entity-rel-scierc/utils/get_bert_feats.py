import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel


class BERT_FEATURES(nn.Module):

    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased') 

        # Set model to evaluation
        for name, param in self.bert.named_parameters():
            param.requires_grad = False
        self.bert.eval()

    def forward(self, input_ids, mask=None, valid_ids=None):
        embeddings, sent_embedding = self.bert(input_ids,
                                               token_type_ids=None,
                                               attention_mask=mask)

        # getting rid of vectors of '[CLS]' and '[SEP]'
        embeddings = embeddings[:, 1:-1, :].contiguous()  

        bsz, _, nhid = embeddings.size()
        if valid_ids is not None:
            embeddings = embeddings[torch.arange(bsz).unsqueeze(-1), valid_ids]

        return embeddings, sent_embedding


def process_sents(tokenizer, batch):

    sentences = []
    valid_tokens = []
    valid_positions = []
    sent_lengths = []

    for i, tokens in enumerate(batch):

        end_idx = 0
        curr_sentence, val_pos, val_tokens = [], [], []
        for token in tokens:
            b_token  = tokenizer.tokenize(token)
            curr_sentence.extend(b_token)
            val_pos.append(end_idx)
            val_tokens.append(b_token[0])
            end_idx += len(b_token)
           
        curr_sentence = ['[CLS]'] + curr_sentence + ['[SEP]']

        sentences.append(curr_sentence)
        valid_tokens.append(val_tokens)
        valid_positions.append(val_pos)

    max_sent_length = max([len(sent) for sent in sentences])
    valid_max_sent_length = max([len(v) for v in valid_positions])

    assert max_sent_length >= valid_max_sent_length

    masks = []
    padded_sentences = []
    for i, sent in enumerate(sentences):
        required_pad_tokens = ['[PAD]'] * (max_sent_length - len(sent))
        padded_sentences.append(sent + required_pad_tokens)

        required_pad = valid_max_sent_length - len(valid_positions[i])
        masks.append([0] * len(valid_positions[i]) + [1] * required_pad)
        if required_pad > 0:
            low = valid_positions[i][-1]
            valid_positions[i].extend([low] * required_pad)


    token_ids = [tokenizer.convert_tokens_to_ids(sent) for sent in padded_sentences]
    return token_ids, valid_positions, valid_tokens, masks


def get_bert_feats(sentences, lower_case=True, batch_size=256):

    # Initialize tokenizer as well as model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=lower_case)
    feat_extractor = BERT_FEATURES()
    feat_extractor.cuda()

    token_ids, valid_positions, valid_tokens, masks = process_sents(tokenizer, sentences)
    input = torch.tensor(token_ids).cuda()
    valid_ids = torch.tensor(valid_positions).cuda()
    masks = torch.BoolTensor(masks)
    
    word_embeddings, batch_sent_embeddings = feat_extractor(input, valid_ids=valid_ids)

    return word_embeddings, masks
