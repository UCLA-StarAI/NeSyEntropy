import re
import numpy as np
from utils import constant
from tnorm.tree_node import *
import torch.nn.functional as F

e2id = constant.NER_TO_ID
r2id = constant.LABEL_TO_ID


def ACE05Constraint(relations, subj, obj):
    kb = np.load('kb.npy', allow_pickle=True).item()

    all_rules = []
    for k, v in kb.items():

        poss = []
        for entity_pair in v.split('|'):

            s, o = re.sub('[\(\)12 ]', '', entity_pair).split('&')

            arg1 = IsEq(subj, e2id[s])
            arg2 = IsEq(obj, e2id[o])
            
            poss.append(And(arg1, arg2))
        
        curr_rule = poss[0]
        for i in range(1, len(poss)):
            curr_rule = curr_rule.Or(poss[i])

        all_rules.append(Implication(IsEq(relations, r2id[k]), curr_rule))
    
    big_rule = all_rules[0]
    for i in range(1, len(all_rules)):
        big_rule = big_rule.And(all_rules[i])
    prob = ProductTNormVisitor().visit(big_rule).clamp(0, 1)
    return (1 - prob)


class Net(torch.nn.Module):
    '''Neural network with a single input (fixed) and two categorical outputs.'''

    def __init__(self, num_labels, w=None):
        super().__init__()
        self.num_labels = num_labels
        if w is not None:
            self.w = torch.nn.Parameter(
                torch.tensor(w).float().view(
                    self.num_labels*2, 1))
        else:
            self.w = torch.nn.Parameter(
                torch.rand(self.num_labels*2, 1))

    def forward(self, x):
        return torch.matmul(self.w, x).view(2, self.num_labels)


def train(net, constraint=None, epoch=100):
    x = torch.tensor([1.0])
    y = torch.tensor([0, 1])
    y0 = F.softmax(net(x), dim=-1)
    opt = torch.optim.SGD(net.parameters(), lr=0.1)

    for _ in range(100):
        opt.zero_grad()
        y_logit = net(x)
        loss = F.cross_entropy(y_logit[1:], y[1:])
        if constraint is not None:
            loss += constraint(y_logit)
        loss.backward()
        opt.step()

    return net, y0


def XorConstraint(y, probs):
    cond = Or(And(y[0], Not(y[1])), And(Not(y[0]), y[1]))
    return -ProductTNormVisitor().visit(cond, probs).log()


def test_xor_binary():

    x = torch.tensor([1.0])
    y = torch.tensor([0, 1])
    net = Net(num_labels=2)

    y0 = F.softmax(net(x), dim=-1)
    opt = torch.optim.SGD(net.parameters(), lr=0.1)

    for i in range(500):
        opt.zero_grad()
        y_logit = net(x)
        loss = F.cross_entropy(y_logit[1:], y[1:])
        loss +=  XorConstraint(y_logit.softmax(dim=-1), y_logit.softmax(dim=-1)).sum()
        loss.backward()
        opt.step()

    y = F.softmax(net(x), dim=-1)
    assert y[0, 1] < 0.25 and y[0, 0] > 0.75

# Entity-Relation Extraction test

ENTITY_TO_ID = {"O": 0, "Loc": 1, "Org": 2, "Peop": 3, "Other": 4}
REL_TO_ID = {"*": 0, "Work_For_arg1": 1, "Kill_arg1": 2, "OrgBased_In_arg1": 3, "Live_In_arg1": 4,
             "Located_In_arg1": 5, "Work_For_arg2": 6, "Kill_arg2": 7, "OrgBased_In_arg2": 8,
             "Live_In_arg2": 9, "Located_In_arg2": 10}

def OrgBasedIn_Org_Loc(ne, re):

    import pdb; pdb.set_trace()
    arg1 = (re==3).nonzero(as_tuple=False)
    arg2 = (re==8).nonzero(as_tuple=False)
    And(ForAll(Eq(ne[Eq(re,3)],2)),ForAll(Eq(ne[Eq(re,8)],1)))

    cond = And(IsEq(ne[arg1], 2), IsEq(ne[arg2], 1)) 
    return -ProductTNormVisitor().visit(cond).log()

class NER_Net(torch.nn.Module):
    '''Simple Named Entity Recognition model'''

    def __init__(self, vocab_size, num_classes, hidden_dim=50, embedding_dim=100):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # layers
        self.embedding = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
        #self.embedding.weight = torch.nn.Parameter(vocab.vectors)
        self.embedding.weight.data.uniform_(-1.0, 1.0)

        self.lstm = torch.nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(self.hidden_dim, num_classes)

        # Initialize fully connected layer
        self.fc.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.fc.weight, gain=1)

    def forward(self, s):
        s = self.embedding(s)   # dim: batch_size x batch_max_len x embedding_dim
        s, _ = self.lstm(s)     # dim: batch_size x batch_max_len x lstm_hidden_dim
        s = self.fc(s)          # dim: batch_size*batch_max_len x num_tags

        return s

class RE_Net(torch.nn.Module):
    '''Simple Relation extraction model'''

    def __init__(self, vocab_size, num_classes, hidden_dim=50, embedding_dim=100):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # layers
        self.embedding = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
        #self.embedding.weight = torch.nn.Parameter(vocab.vectors)
        self.embedding.weight.data.uniform_(-1.0, 1.0)

        self.lstm = torch.nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(self.hidden_dim, num_classes)

        # Initialize fully connected layer
        self.fc.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.fc.weight, gain=1)

    def forward(self, s):
        s = self.embedding(s)   # dim: batch_size x batch_max_len x embedding_dim
        s, _ = self.lstm(s)     # dim: batch_size x batch_max_len x lstm_hidden_dim
        s = self.fc(s)          # dim: batch_size*batch_max_len x num_tags

        return s

def test_entity_relation():

    ner = NER_Net(vocab_size=3027, num_classes=len(ENTITY_TO_ID))
    re = RE_Net(vocab_size=3027, num_classes=len(REL_TO_ID))

    opt = torch.optim.SGD(list(ner.parameters()) + list(re.parameters()), lr=1.0)

    tokens, entities, relations = get_data()

    for i in range(100):
        opt.zero_grad()

        ner_logits = ner(tokens)
        ner_logits = ner_logits.view(-1, ner_logits.shape[2])

        re_logits = re(tokens)
        re_logits = re_logits.view(-1, re_logits.shape[2])

        re_loss = F.cross_entropy(re_logits, relations.view(-1))
        closs =  OrgBasedIn_Org_Loc(ner_logits.softmax(dim=-1), re_logits.softmax(dim=-1)).sum()
        print(closs)
        loss = 0.05 * closs + 10 * re_loss

        loss.backward()
        opt.step()

    re = torch.argmax(torch.softmax(re(tokens).view(-1, 11), dim=-1), dim=-1)
    ner = torch.argmax(torch.softmax(ner(tokens).view(-1, 5), dim=-1), dim=-1)

    assert (ner[re == 3] == 2).all() and (ner[re == 8] == 1).all()
    return ner, re

def get_data():

    tokens = torch.tensor([[32, 1973, 2272,   15,    3,    0,    0,    5,    0,  389,    0,   12,
                            7,  823,    4, 2636,    4,    0,  114,    5,    3, 2701,    6]])
    entities = torch.LongTensor([0, 0, 0, 0, 0, 0, 2, 0, 4, 0, 1, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0])
    relations = torch.LongTensor([0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    return tokens, entities, relations

if __name__ == "__main__":
    test_entity_relation()
