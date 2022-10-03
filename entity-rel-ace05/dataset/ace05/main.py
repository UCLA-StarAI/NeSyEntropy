import os
import copy
import re
from parser import Parser
import json
from stanfordcorenlp import StanfordCoreNLP
import argparse
from tqdm import tqdm
import glob

# Return the constituents of the test, dev and train set
def get_data_paths(ace2005_path):

    train_files = [os.path.splitext(os.path.splitext(f)[0])[0] for f in glob.glob(ace2005_path + 'train/*.split.txt')]
    dev_files = [os.path.splitext(os.path.splitext(f)[0])[0] for f in glob.glob(ace2005_path + 'dev/*.split.txt')]
    test_files = [os.path.splitext(os.path.splitext(f)[0])[0] for f in glob.glob(ace2005_path + 'test/*.split.txt')]

    return test_files, dev_files, train_files

def find_token_index(tokens, start_pos, end_pos, phrase):
    start_idx, end_idx = -1, -1
    for idx, token in enumerate(tokens):
        if token['characterOffsetBegin'] <= start_pos:
            start_idx = idx

    assert start_idx != -1, "start_idx: {}, start_pos: {}, phrase: {}, tokens: {}".format(start_idx, start_pos, phrase, tokens)
    chars = ''

    def remove_punc(s):
        s = re.sub(r'[^\w]', '', s)
        return s

    for i in range(0, len(tokens) - start_idx):
        chars += remove_punc(tokens[start_idx + i]['originalText'])
        if remove_punc(phrase) in chars:
            end_idx = start_idx + i + 1
            break

    assert end_idx != -1, "end_idx: {}, end_pos: {}, phrase: {}, tokens: {}, chars:{}".format(end_idx, end_pos, phrase, tokens, chars)
    return start_idx, end_idx

def preprocessing(data_type, files):

    result = []

    print('=' * 20)
    print('[preprocessing] type: ', data_type)

    for file in tqdm(files):
        parser = Parser(path=file)

        for item in parser.get_data():
            data = dict()
            data['sentence'] = item['sentence']
            data['golden-entity-mentions'] = item['golden-entity-mentions']
            data['golden-relation-mentions'] = []

            try:
                nlp_res_raw = nlp.annotate(item['sentence'], properties={'annotators': 'tokenize, ssplit, pos, lemma, parse'})
                nlp_res = json.loads(nlp_res_raw)

            except Exception as e:
                print('[Warning] StanfordCore Exception: ', nlp_res_raw, 'This sentence will be ignored.')
                continue

            tokens = nlp_res['sentences'][0]['tokens']

            if len(nlp_res['sentences']) >= 2:
                # TODO: issue where the sentence segmentation of NTLK and StandfordCoreNLP do not match
                # This error occurred so little that it was temporarily ignored (< 20 sentences).
                continue

            data['stanford-colcc'] = []
            for dep in nlp_res['sentences'][0]['enhancedPlusPlusDependencies']:
                data['stanford-colcc'].append('{}/dep={}/gov={}'.format(dep['dep'], dep['dependent'] - 1, dep['governor'] - 1))

            data['words'] = list(map(lambda x: x['word'], tokens))
            data['pos-tags'] = list(map(lambda x: x['pos'], tokens))
            data['lemma'] = list(map(lambda x: x['lemma'], tokens))
            data['parse'] = nlp_res['sentences'][0]['parse']

            # Calculate the start/end index for every mention
            for entity_mention in item['golden-entity-mentions']:
                start_idx, end_idx = find_token_index(
                    tokens=tokens,
                    start_pos=entity_mention['position'][0] - item['position'][0],
                    end_pos=entity_mention['position'][1] - item['position'][0] + 1,
                    phrase=entity_mention['text'],
                )
                entity_mention['start'] = start_idx
                entity_mention['end'] = end_idx - 1

                del entity_mention['position']

            # Get a list of ners for the entire sentence
            data['ner'] = []
            for word in data['words']:
                found = False
                for entity in item['golden-entity-mentions']:
                    if word in entity['text']:
                        data['ner'].append(entity['entity-type'])
                        found = True
                        break

                if not found:
                    data['ner'].append('O')

            try:
                assert(len(data['ner']) == len(data['words']))
            except:
                import pdb; pdb.set_trace()

            # Every sentence should have one and only one relation
            for relation_mention in item['golden-relation-mentions']:
                tmp = copy.deepcopy(data)
                for entity in relation_mention['arguments']:
                    start_idx, end_idx = find_token_index(
                        tokens=tokens,
                        start_pos=entity['position'][0] - item['position'][0],
                        end_pos=entity['position'][1] - item['position'][0] + 1,
                        phrase=entity['text'],
                    )

                    entity['start'] = start_idx
                    entity['end'] = end_idx - 1
                    del entity['position']

                tmp['golden-relation-mentions'].append(relation_mention)
                result.append(tmp)


    with open('output/{}.json'.format(data_type), 'w') as f:
        json.dump(result, f, indent=2)

if __name__ == '__main__':
    
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help="Path of ACE2005 English data", default='/local/ahmedk/Data/LSTM-ER/data/ace2005/corpus/')
    args = parser.parse_args()

    # Get test, validation and training files
    test_files, dev_files, train_files = get_data_paths(args.data)

    # Preprocess the data
    with StanfordCoreNLP('./stanford-corenlp-full-2018-10-05', memory='8g', timeout=990000) as nlp:
        preprocessing('dev', dev_files)
        preprocessing('test', test_files)
        preprocessing('train', train_files)
