from xml.etree import ElementTree
from bs4 import BeautifulSoup
import nltk
import json
import re
import numpy as np

class Parser:
    def __init__(self, path):
        self.path = path
        self.entity_mentions = []
        self.relation_mentions = []

        self.entity_mentions, self.relation_mentions = self.parse_ann_file(path + '.split.ann')
        self.sents_with_pos = self.parse_txt_file(path + '.split.txt')

    @staticmethod
    def parse_entity(line):
        '''
        TAFP_ENG_20030327.0224-E112-139 PER 197 203 troops
        '''
        entity_mention = dict()
        split = line.split()

        entity_mention['entity-id'] = split[0] 
        entity_mention['entity-type'] = split[1]
        entity_mention['text'] = ' '.join(split[4:])
        entity_mention['position'] = [int(split[2]), int(split[3])]

        return entity_mention
    
    @staticmethod
    def parse_relation(line):
        '''
        RAFP_ENG_20030327.0224-R2-1 ORG-AFF Arg1:TAFP_ENG_20030327.0224-E112-139 Arg2:TAFP_ENG_20030327.0224-E22-190
        '''
        relation_mention = dict()
        split = line.split()

        relation_mention['relation-type'] = split[1] 
        relation_mention['arguments'] = [split[2].split(':')[1], split[3].split(':')[1]]

        return relation_mention

    def parse_ann_file(self, path):
        entity_mentions, relation_mentions = [], []
        with open(path) as f:
            for line in f:
                if line.split()[0].split('-')[-2][0] == 'E':
                    entity_mentions += [self.parse_entity(line)]
                else:
                    relation_mentions += [self.parse_relation(line)]
        return entity_mentions, relation_mentions

    @staticmethod
    def parse_txt_file(path):
        sentences_w_pos = []
        with open(path) as f:
            sentences = f.readlines()
            pos = 0
            for sentence in sentences:
                sentences_w_pos.append({
                    'text': sentence,
                    'position': [pos, pos + len(sentence)]
                    })
                pos += len(sentence)
        return sentences_w_pos

    def get_data(self):

        data = []
        for sent in self.sents_with_pos:
            item = dict()
            item['sentence'] = sent['text']
            item['position'] = sent['position']
            

            entity_map = dict()
            item['golden-entity-mentions'] = []
            item['golden-relation-mentions'] = []

            for entity_mention in self.entity_mentions:

                # Check that the entity is within the sentence bounds
                if item['position'][0] <= entity_mention['position'][0] and entity_mention['position'][1] <= item['position'][1]:

                    item['golden-entity-mentions'].append({
                        'text': entity_mention['text'],
                        'position': entity_mention['position'],
                        'entity-type': entity_mention['entity-type']
                    })

                    entity_map[entity_mention['entity-id']] = entity_mention

            # If there are no entities, we just skip this sentence
            if len(entity_map.keys()) < 2:
                continue

            for relation_mention in self.relation_mentions:
                if relation_mention['arguments'][0] in entity_map.keys()\
                        and relation_mention['arguments'][1] in entity_map.keys():

                    entity1 = entity_map[relation_mention['arguments'][0]]
                    entity2 = entity_map[relation_mention['arguments'][1]]

                    # Construct each argument of the relation
                    relation_arguments = []
                    for entity in [entity1, entity2]: 
                        relation_arguments.append({
                            'position': entity['position'],
                            'entity-type': entity['entity-type'],
                            'text': entity['text'],
                        })

                    # Add the current relation to the list of realtions
                    item['golden-relation-mentions'].append({
                        'arguments': relation_arguments,
                        'relation-type': relation_mention['relation-type'],
                    })

            # If none of the available relations match the current sentece,
            # we pick two entities at random and set their relation to be 'NA'
            if len(item['golden-relation-mentions']) == 0:
                choices = np.random.choice(len(item['golden-entity-mentions']), 2)

                relation_arguments = []
                for i in choices:
                    relation_arguments.append({
                        'position': item['golden-entity-mentions'][i]['position'],
                        'entity-type': item['golden-entity-mentions'][i]['entity-type'],
                        'text': item['golden-entity-mentions'][i]['text']
                        })

                item['golden-relation-mentions'].append({
                    'arguments': relation_arguments,
                    'relation-type': 'NA',
                })

            data.append(item)
        return data

if __name__ == '__main__':
    parser = Parser('./data/ace_2005_td_v7/data/English/un/timex2norm/alt.corel_20041228.0503')
    data = parser.get_data()

    with open('./output/debug.json', 'w') as f:
        json.dump(data, f, indent=2)
