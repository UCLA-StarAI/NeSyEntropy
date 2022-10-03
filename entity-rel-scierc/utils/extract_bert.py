import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

plt.switch_backend('agg')
import matplotlib.ticker as ticker

from pytorch_pretrained_bert import BertTokenizer, BertModel

lower_case = True  # set to False if cased model is used

# bert_model = 'bert-base-uncased'
# bert_model = 'bert-base-cased'


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)  # (0 for pad positions)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)
    # ax.tick_params(axis='x', pad=60)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center",
             rotation_mode="default")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


class BERT_FEATURES(nn.Module):
    def __init__(self):
        super().__init__()
        # initialize pretrained model, base models have output of dim 768
        self.bert = BertModel.from_pretrained('bert-base-uncased')  # may take time to download
        for name, param in self.bert.named_parameters():
            param.requires_grad = False
        self.bert.eval()  # unless you want to fine-tune model

    def forward(self, input_ids, mask=None, valid_ids=None):
        # token_type_ids used only in BERT model LM training task
        embeddings, sent_embedding = self.bert(input_ids,
                                               token_type_ids=None,
                                               attention_mask=mask,
                                               output_all_encoded_layers=False)

        embeddings = embeddings[:, 1:-1, :].contiguous()  # getting rid of vectors of '[CLS]' and '[SEP]'

        bsz, _, nhid = embeddings.size()
        if valid_ids is not None:
            embeddings = embeddings[torch.arange(bsz).unsqueeze(-1), valid_ids]

        return embeddings, sent_embedding


def process_sents(tokenizer, sample):
    sentences = []
    valid_tokens = []
    valid_positions = []
    sent_lengths = []
    for i, sent in enumerate(sample):
        tokens = tokenizer.tokenize(sent)  # we expect |token| = 1
        val_pos = []
        val_toks = []
        end_idx = 0
        for token in tokens:
            if not token.startswith('##'):
                val_toks.append(token)
                val_pos.append(end_idx)
            end_idx += 1

        tokens = ['[CLS]'] + tokens + ['[SEP]']
        sentences.append(tokens)
        valid_tokens.append(val_toks)
        valid_positions.append(val_pos)

    max_sent_length = max([len(sent) for sent in sentences])
    valid_max_sent_length = max([len(v) for v in valid_positions])

    assert max_sent_length >= valid_max_sent_length

    reshaped_sentences = []
    for i, sent in enumerate(sentences):
        required_pad_tokens = ['[PAD]'] * (max_sent_length - len(sent))
        reshaped_sentences.append(sent + required_pad_tokens)

        required_pad = valid_max_sent_length - len(valid_positions[i])
        if required_pad > 0:
            low = valid_positions[i][-1]
            valid_positions[i].extend([low] * required_pad)

    token_ids = [tokenizer.convert_tokens_to_ids(sent) for sent in reshaped_sentences]
    return token_ids, valid_positions, valid_tokens


def main():
    raw_sentences_1 = [
        "However, by now it is clear that ERPs following fixation of a target are different than ERPs following fixation of a nontarget and that these differences are associated with topdown stimulus processing.",
        "Ectopic expression of DREB2A in Arabidopsis increase endurance to drought, stress, and heat stresses.",
        "Transgenic plants with overexpressed SFAR4 exhibited tolerance to glucose stress and a higher germination rate.",
        "This sustained posterior contralateral negativitycomponent is assumed to reflect the encoding of target stimuli in visual working memory that follows their initial selection see refsandfor details.Figure 3Average N2pc for left and right targets in Experiment 1.",
        "Expression of VrDREB2A in mungbean seedlings was markedly induced by drought, highsalt stress and ABA treatment, but only slightly by cold stress."]

    raw_sentences_2 = [
        "MSCs derived from bone marrow, ATderived MSCof humans and rhesus macaque showed altered cell cycle progression at earlyand higher passagesin a longterm in vitro expansion.",
        "Foveal bulge is the site of maximum cone density and hence the site of maximum vision.",
        "Further analyses of HIV Env have revealed no consistent correlation between higher resistance to sCD4mediated inhibition and sCD4binding affinity.",
        "The neuromodulator dopamine has been linked particularly strongly to behavioural activation in the context of reward putatively by amplifying the perceived benefits of action over their costs.",
        "Foveal bulge is the site of maximum cone density and hence the site of maximum vision."]

    # raw_sentences_1 = [
    #     "Symptoms of influenza include fever and nasal congestion.",
    #     "Her life spanned years of incredible change for women as they gained more rights than ever before.",
    #     "Giraffes like Acacia leaves and hay, and they can consume 75 pounds of food a day."]
    #
    # raw_sentences_2 = [
    #     "A stuffy nose and elevated temperature are signs you may have the flu.",
    #     "She lived through the exciting era of women's liberation.",
    #     "A giraffe can eat up to 75 pounds of Acacia leaves and hay daily."]

    import pdb; pdb.set_trace()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=lower_case)
    feat_extractor = BERT_FEATURES()
    feat_extractor.cuda()

    token_ids, valid_positions, valid_tokens_1 = process_sents(tokenizer, raw_sentences_1)
    input = torch.tensor(token_ids).cuda()
    valid_ids = torch.tensor(valid_positions).cuda()

    embeddings_1, sent_embeddings_1 = feat_extractor(input, valid_ids=valid_ids)
    # embeddings_1, sent_embeddings_1 = feat_extractor(input, valid_ids=None)
    pooled_rep_1 = torch.mean(embeddings_1, dim=1)

    token_ids, valid_positions, valid_tokens_2 = process_sents(tokenizer, raw_sentences_2)
    input = torch.tensor(token_ids).cuda()
    valid_ids = torch.tensor(valid_positions).cuda()

    embeddings_2, sent_embeddings_2 = feat_extractor(input, valid_ids=valid_ids)
    # embeddings_2, sent_embeddings_2 = feat_extractor(input, valid_ids=None)
    pooled_rep_2 = torch.mean(embeddings_2, dim=1)

    cosine_sim = F.cosine_similarity(pooled_rep_1, pooled_rep_2, dim=1)
    print(cosine_sim.cpu().numpy().tolist())

    # cosine_sim = F.cosine_similarity(sent_embeddings_1, sent_embeddings_2, dim=1)
    # print(cosine_sim.cpu().numpy())

    len_tokens_2 = [len(valid_tokens_2[i]) for i in range(len(valid_tokens_2))]
    len_tokens_2 = torch.from_numpy(np.asarray(len_tokens_2)).cuda()
    mask = sequence_mask(len_tokens_2)
    mask = mask.unsqueeze(1)  # Make it broadcastable.

    attention = torch.bmm(embeddings_1, embeddings_2.transpose(1, 2))
    attention.masked_fill_(1 - mask, -float('inf'))
    attention = F.softmax(attention, dim=2)

    # create plots
    for i in range(len(raw_sentences_1)):
        fig, ax = plt.subplots()
        len_1, len_2 = len(valid_tokens_1[i]), len(valid_tokens_2[i])
        im, cbar = heatmap(attention[i].cpu().numpy()[:len_1, :len_2],
                           valid_tokens_1[i],
                           valid_tokens_2[i],
                           ax=ax,
                           cmap="YlGn",
                           cbarlabel="Softmax weights")

        fig.tight_layout()
        # plt.show()
        plt.savefig('ex_%d.png' % i)
        plt.clf()


if __name__ == '__main__':
    main()
