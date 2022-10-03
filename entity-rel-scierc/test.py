from eval_script import *
import torch.nn.functional as F
from model.compute_mpe import CircuitMPE


# model_dir = 'sl_entreg_sat_search'
# model_dir = 'sl_entreg_sat_search/75_0.05_1234'
model_name='best_model.pt'
data_dir='dataset/sciERC'
dataset='test'

# model_file = model_dir + '/' + model

# opt = torch_utils.load_config(model_file)
# model = RelationModel(opt)
# model.load(model_file)

cmpe = CircuitMPE('dataset/sciERC/no_me_constraint.vtree', 'dataset/sciERC/no_me_constraint.sdd')
# model.model.eval()
# logits, _, ner_logits, subj_idx, obj_idx = model.model(inputs)

def get_sat_joint_dist(model, logits, ner_logits, masks, subj_idx, obj_idx, batch_size):
    subj_idx = subj_idx + masks.long()
    subj_idx = (subj_idx==0).nonzero()[:,1]

    obj_idx = obj_idx + masks.long()
    obj_idx = (obj_idx==0).nonzero()[:,1]

    relation_probs = F.softmax(logits, -1)
    num_ners = len(constant.NER_TO_ID)
    ner_probs = F.softmax(ner_logits, -1).view(batch_size, -1, num_ners)

    subj_probs = ner_probs[np.arange(batch_size), subj_idx]
    obj_probs = ner_probs[np.arange(batch_size), obj_idx]

    joint_dists = []

    for i in range(batch_size):
        # index goes rel, subj, obj
        joint_dist = model.all_prod(relation_probs[i,:],model.all_prod(subj_probs[i,:],obj_probs[i,:]))
        # ent_reg_loss -= torch.sum(joint_dist * torch.log(joint_dist))
        sat_joint_dist = joint_dist[model.sat_idxs]
        sat_joint_dist /= torch.sum(sat_joint_dist)
        entropy = -torch.sum(sat_joint_dist * torch.log(sat_joint_dist))
        torch_entropy = torch.distributions.Categorical(probs=sat_joint_dist).entropy()
        # print(sat_joint_dist)
        # print("entropy: %f\n" %(entropy))

        #psdd entropy
        probs = torch.cat((relation_probs[i, 1:], subj_probs[i,:], obj_probs[i,:]))
        # probs = torch.unbind(probs, dim=1)
        wmc = cmpe.get_torch_ac([[1.0 - p,p] for p in probs])
        circuit_entropy = cmpe.Shannon_entropy()

        joint_dists.append((sat_joint_dist,entropy,torch_entropy,circuit_entropy))
    return joint_dists

def get_avg_entropy(model_dir):
    model_file = model_dir + '/' + model_name

    opt = torch_utils.load_config(model_file)
    model = RelationModel(opt)
    model.load(model_file)

    vocab_file = model_dir + '/vocab.pkl'
    vocab = Vocab(vocab_file, load=True)
    assert opt['vocab_size'] == vocab.size, "Vocab size must match that in the saved model."

    data_file = data_dir + '/{}.json'.format(dataset)
    batch = DataLoader(data_file, opt['batch_size'], opt, vocab, evaluation=True)

    entropies = []
    torch_entropies = []
    circuit_entropies = []

    for cur_batch in batch:
        inputs = [b.cuda() for b in cur_batch[:7]]
        labels = cur_batch[7].cuda()
        orig_idx = cur_batch[8]
        
        model.model.eval()
        logits, _, ner_logits, subj_idx, obj_idx = model.model(inputs)
        joint_dists = get_sat_joint_dist(model, logits, ner_logits, inputs[1], subj_idx, obj_idx, len(labels))

        for x in joint_dists:
            ent = x[1].item()
            entropies.append(ent)
            torch_entropies.append(x[2].item())
            circuit_entropies.append(x[3].item())
            
    avg_entropy = sum(torch_entropies)/len(torch_entropies)
    return avg_entropy
    # print("avg entropy: %f\n"%(sum(torch_entropies)/len(torch_entropies)))
    # return entropies, torch_entropies, circuit_entropies

eval_out_file = 'sciERC_experiments/eval_sl_no_me_sdd_entreg.txt'
f = open(eval_out_file).readlines()
avg_entropies = []
for l in f:
    if 'search' in l:
        avg_entropies.append((l[:-1],get_avg_entropy(l[:-1])))
# e, te, ce = get_avg_entropy('sciERC_experiments/sl_no_me_bf_entreg_search/5_0.1_2345')