import json

for split in ['test', 'train', 'dev']:
    with open(split + '.json') as f:
        split_data = json.load(f)

    data = []
    for datum in split_data:

        new_datum = dict()
        new_datum['sentence'] = datum['sentence']
        new_datum['token'] = datum['words'] 
        new_datum['stanford_pos'] = datum['pos-tags']
        new_datum['stanford_ner'] = datum['ner']

        new_datum['relation'] = datum['golden-relation-mentions'][0]['relation-type']

        new_datum['subj_start'] = datum['golden-relation-mentions'][0]['arguments'][0]['start'] 
        new_datum['subj_end'] = datum['golden-relation-mentions'][0]['arguments'][0]['end'] 

        new_datum['obj_start'] = datum['golden-relation-mentions'][0]['arguments'][1]['start'] 
        new_datum['obj_end'] = datum['golden-relation-mentions'][0]['arguments'][1]['end'] 

        new_datum['subj_type'] = datum['golden-relation-mentions'][0]['arguments'][0]['entity-type'] 
        new_datum['obj_type'] = datum['golden-relation-mentions'][0]['arguments'][1]['entity-type'] 

        new_datum['stanford_deprel'] = [] 
        for tag in datum['stanford-colcc']:
            new_datum['stanford_deprel'].append(tag.split('/')[0])

        data.append(new_datum)
        
    with open(split + '.tacred.json', 'w') as f:
        json.dump(data, f, indent=2)
