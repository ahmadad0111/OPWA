""" Meta Learner """
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.feature_extractor import FeatureExtractor, FeatureExtractorCNN
from torchvision.transforms import v2
from InfoNCE.utils import normalize
from InfoNCE.contrastive_learning import get_similarity_matrix,Supervised_NT_xent_pre,Supervised_NT_xent_n,Supervised_NT_xent_uni
mseed = 48


if mseed == 0:
    torch.backends.cudnn.benchmark = True
else:
    torch.manual_seed(mseed)
    torch.cuda.manual_seed(mseed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
class PredHead(nn.Module):
    """The class for inner loop."""
    def __init__(self, args, z_dim):
        super().__init__()
        self.args = args
        self.z_dim = z_dim
        self.vars = nn.ParameterList()
        self.fc1_w = nn.Parameter(torch.ones([self.args.way, self.z_dim]))
        torch.nn.init.kaiming_normal_(self.fc1_w)
        self.vars.append(self.fc1_w)
        self.fc1_b = nn.Parameter(torch.zeros(self.args.way))
        self.vars.append(self.fc1_b)

    def forward(self, input_x, the_vars=None):
        if the_vars is None:
            the_vars = self.vars
        fc1_w = the_vars[0]
        fc1_b = the_vars[1]
        net = F.softmax(F.linear(input_x, fc1_w, fc1_b), dim=1)
        return net, F.linear(input_x, fc1_w, fc1_b)

    def parameters(self):
        return self.vars

class MtlLearner(nn.Module):
    """The class for outer loop."""
    def __init__(self, args, mode='meta', num_cls=4):
        super().__init__()
        self.args = args
        self.mode = mode
        self.update_lr = args.base_lr
        self.update_step = args.update_step
        #z_dim = 4*2*48
        if args.modality == 'EEG':
            z_dim =  4*2*56 #4*2*160#
        else:
            z_dim = 400
        
        #z_dim = 15360
        z_pred = args.z_pred
        self.pred_head = PredHead(args, z_pred)
        self.pred_embed = nn.Linear(z_dim, z_pred)
        if args.modality == 'EEG':
            self.extractor = FeatureExtractor()
        else:
            self.extractor = FeatureExtractorCNN()

        if self.mode == 'meta':
            self.encoder = nn.Sequential(self.extractor, self.pred_embed)
        else:
            self.encoder = nn.Sequential(self.extractor, self.pred_embed)
            #self.pre_fc = nn.Sequential(nn.Linear(z_pred, num_cls))
        with torch.no_grad():
            self.sim_aug = v2.Compose([v2.RandomAffine(5, (0.01, 0.01)), v2.RandomRotation(5), ])

    def forward(self, inp):
        if self.mode=='meta':
            data_shot, label_shot, data_query = inp
            return self.meta_forward2(data_shot, label_shot, data_query)
        elif self.mode=='preval':
            data_shot, label_shot, data_query = inp
            return self.preval_forward(data_shot, label_shot, data_query)
        else:
            raise ValueError('Please set the correct mode.')
  

    def preval_forward(self, data_shot, label_shot, data_query):
        embedding_query = self.encoder(data_query)
        embedding_shot = self.encoder(data_shot)
        logits = self.pred_head(embedding_shot)
        loss = F.cross_entropy(logits, label_shot)
        grad = torch.autograd.grad(loss, self.pred_head.parameters())
        fast_weights = list(map(lambda p: p[1] - 0.005 * p[0], zip(grad, self.pred_head.parameters())))
        logits_q = self.pred_head(embedding_query, fast_weights)

        for _ in range(2):
            logits = self.pred_head(embedding_shot, fast_weights)
            loss = F.cross_entropy(logits, label_shot)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - 0.005 * p[0], zip(grad, fast_weights)))
            logits_q = self.pred_head(embedding_query, fast_weights)
        return logits_q

    def meta_forward(self, data_shot, label_shot, data_query):
        embedding_query = self.encoder(data_query)
        embedding_shot = self.encoder(data_shot)
        logits = self.pred_head(embedding_shot)
        loss = F.cross_entropy(logits, label_shot)
        grad = torch.autograd.grad(loss, self.pred_head.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.pred_head.parameters())))
        logits_q = self.pred_head(embedding_query, fast_weights)

        for _ in range(1, self.update_step):
            logits = self.pred_head(embedding_shot, fast_weights)
            loss = F.cross_entropy(logits, label_shot)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
            logits_q = self.pred_head(embedding_query, fast_weights)        
        return logits_q

    def meta_forward2(self, data_shot, label_shot, data_query):

        embedding_shot=self.encoder(data_shot)
        embedding_query = self.encoder(data_query)
        
        for _ in range(self.update_step):
            # data_shot_aug = self.sim_aug(data_shot)
            # embedding_aug = self.encoder(data_shot_aug)
            # rot_sim_labels = torch.cat([label_shot],dim=0)
            # simclr = normalize(torch.cat([embedding_shot, embedding_aug], dim=0))
            
            # sim_matrix = get_similarity_matrix(simclr)
            # loss_sim = Supervised_NT_xent_n(sim_matrix, labels=rot_sim_labels, temperature=0.05)


            
            params=self.pred_head.parameters()
            optimizer=optim.Adam(params,lr=self.update_lr)
            logits,_ = self.pred_head(embedding_shot)
            loss = F.cross_entropy(logits, label_shot)#+loss_sim
            loss.backward(retain_graph=True)
            optimizer.step()
        logits_q,activ = self.pred_head(embedding_query)

        return logits_q,activ



