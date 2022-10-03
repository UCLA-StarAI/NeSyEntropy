"""
Define common constants.
"""
TRAIN_JSON = 'train.json'
DEV_JSON = 'dev.json'
TEST_JSON = 'test.json'

GLOVE_DIR = 'dataset/glove'

EMB_INIT_RANGE = 1.0
MAX_LEN = 400

# vocab
PAD_TOKEN = '<PAD>'
PAD_ID = 0 
UNK_TOKEN = '<UNK>'
UNK_ID = 1

VOCAB_PREFIX = [PAD_TOKEN, UNK_TOKEN]

POS_TO_ID = {'ADV': 0, 'ADP': 1, 'DET': 2, 'NOUN': 3, 'AUX': 4, 'VERB': 5, 'PRON': 6, 'ADJ': 7, 'PUNCT': 8, 'NUM': 9, 'PROPN': 10, 'CCONJ': 11, 'PART': 12, 'SCONJ': 13, 'SYM': 14, 'INTJ': 15, 'X': 16}

DEPREL_TO_ID = {'advmod': 0, 'case': 1, 'det': 2, 'nsubj': 3, 'compound': 4, 'nmod': 5, 'aux': 6, 'root': 7, 'nmod:poss': 8, 'obj': 9, 'amod': 10, 'obl': 11, 'obl:tmod': 12, 'punct': 13, 'nummod': 14, 'conj': 15, 'xcomp': 16, 'parataxis': 17, 'fixed': 18, 'cc': 19, 'flat': 20, 'cop': 21, 'appos': 22, 'expl': 23, 'acl:relcl': 24, 'mark': 25, 'advcl': 26, 'ccomp': 27, 'compound:prt': 28, 'acl': 29, 'iobj': 30, 'discourse': 31, 'det:predet': 32, 'nsubj:pass': 33, 'aux:pass': 34, 'vocative': 35, 'obl:npmod': 36, 'nmod:tmod': 37, 'list': 38, 'csubj': 39, 'nmod:npmod': 40, 'cc:preconj': 41, 'goeswith': 42, 'reparandum': 43}

LABEL_TO_ID = {'NA': 0, 'PART-WHOLE:Geographical': 1, 'ART:User-Owner-Inventor-Manufacturer': 2, 'ORG-AFF:Membership': 3, 'PER-SOC:Business': 4, 'ORG-AFF:Employment': 5, 'ORG-AFF:Student-Alum': 6, 'PHYS:Located': 7, 'PART-WHOLE:Subsidiary': 8, 'GEN-AFF:Org-Location': 9, 'GEN-AFF:Citizen-Resident-Religion-Ethnicity': 10, 'PER-SOC:Family': 11, 'ORG-AFF:Founder': 12, 'ORG-AFF:Ownership': 13, 'PHYS:Near': 14, 'PART-WHOLE:Artifact': 15, 'PER-SOC:Lasting-Personal': 16, 'ORG-AFF:Investor-Shareholder': 17, 'ORG-AFF:Sports-Affiliation': 18}

LABEL_TO_ID_04 = {'NA': 0, 'EMP-ORG:Employ-Staff':5, 'PHYS:Located':7, 'PHYS:Part-Whole':1, 'GPE-AFF:Citizen-or-Resident':10, 'EMP-ORG:Employ-Executive':5, 'DISC:DISC':0, 'EMP-ORG:Member-of-Group':3, 'GPE-AFF:Based-In':7, 'ART:User-or-Owner':2, 'EMP-ORG:Subsidiary':8, 'PHYS:Near':14, 'OTHER-AFF:Ethnic':10, 'OTHER-AFF:Other':10, 'EMP-ORG:Other':5, 'OTHER-AFF:Ideology':10, 'PER-SOC:Business':4, 'EMP-ORG:Employ-Undetermined':5, 'GPE-AFF:Other':10, 'PER-SOC:Other':4, 'EMP-ORG:Partner':17, 'PER-SOC:Family':11, 'ART:Inventor-or-Manufacturer':2, 'ART:Other':2}

#NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'O': 2, 'FAC': 3, 'GPE': 4, 'LOC': 5, 'ORG': 6, 'PER': 7, 'VEH': 8, 'WEA': 9}
NER_TO_ID = {'O': 0, 'FAC': 1, 'GPE': 2, 'LOC': 3, 'ORG': 4, 'PER': 5, 'VEH': 6, 'WEA': 7}
#NER_TO_ID = {'O': 1, 'FAC': 2, 'GPE': 3, 'LOC': 4, 'ORG': 5, 'PER': 6, 'VEH': 7, 'WEA': 9}

INFINITY_NUMBER = 1e12
