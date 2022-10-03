import json
import pickle
from collections import defaultdict
from stanfordcorenlp import StanfordCoreNLP

def split_by_setence_and_relation():
    '''
    Original training data contains paper abstracts split by sentence.
    Each sentence has NERs and relations.
    This splits the data so that each data point is 1 sentence with 1 relation.
    '''
    orig_dir = 'original/'
    files = ['dev.json','test.json','train.json']
    nlp = StanfordCoreNLP('../../../../../home/ericzxwang/stanford-corenlp-full-2018-10-05', memory='8g', timeout=990000)
    for f in files:
        data = json.load(open(orig_dir+f))
        split_data = []
        for d in data:
            sentences = d['sentences']
            ners = d['ner']
            rels = d['relations']
            len_sum = 0
            for i in range(len(sentences)):
                if len(sentences[i]) == 0 or len(ners[i]) == 0 or len(rels[i]) == 0:
                    len_sum += len(sentences[i])
                    continue
                # adjust idexes to start from beginning of sentence
                cur_ners = ners[i]
                ner_vec = ['O' for j in range(len(sentences[i]))]
                # pos_vec = ['O' for j in range(len(sentences[i]))]
                pos_vec = get_pos_vec(sentences[i], nlp)
                deprel_vec = ['O' for j in range(len(sentences[i]))]
                for ner in cur_ners:
                    ner[0] -= len_sum
                    ner[1] -= len_sum
                    for j in range(ner[0],ner[1]+1):
                        ner_vec[j] = ner[-1]
                for rel in rels[i]:
                    for j in range(len(rel)-1):
                        rel[j] -= len_sum
                    ss, se, os, oe, r = rel
                    stype = ner_vec[ss]
                    otype = ner_vec[os]
                    split_data.append({'token':sentences[i],'stanford_pos':pos_vec,\
                        'stanford_deprel':deprel_vec,'stanford_ner':ner_vec,'relation':r,\
                        'subj_start':ss,'subj_end':se,'obj_start':os,'obj_end':oe,\
                        'subj_type':stype,'obj_type':otype}) 
                len_sum += len(sentences[i])
        json.dump(split_data, open(f, 'w'), indent=2)

def get_pos_vec(tkns, nlp):
    s = ' '.join(tkns)
    pos = nlp.pos_tag(s)
    return [x[1] for x in pos]

def get_pos_tags():
    files = ['dev.json','test.json','train.json']
    pos_tags = []
    pos_ctr = defaultdict(lambda: 0)
    for f in files:
        data = json.load(open(f))
        for d in data:
            pos = d['stanford_pos']
            for p in pos:
                pos_ctr[p] += 1
    pos_tags = sorted(pos_ctr, key=lambda k:pos_ctr[k], reverse=True)
    pos_tag_map = {}
    for i in range(len(pos_tags)):
        pos_tag_map[pos_tags[i]] = i
    return pos_tag_map

        