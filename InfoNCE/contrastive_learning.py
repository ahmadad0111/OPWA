import pdb

import torch
import torch.distributed as dist
import torch.nn as nn
import  torch.nn.functional as F
def JS_MI(sim_matrix, labels, temperature=0.5,eps=1e-8):
    device = sim_matrix.device
    label=labels.repeat(2)
    label = label.contiguous().view(-1, 1)

    #import pdb
    #pdb.set_trace()
    Mask2 = torch.eq(label, label.t()).float().to(device)
    Mask1 = torch.ne(label, label.t()).float().to(device)

    Mask1 = Mask1 / (Mask1.sum(dim=1, keepdim=True) + eps)
    Mask2 = Mask2 / (Mask2.sum(dim=1, keepdim=True) + eps)

    B = sim_matrix.size(0) // 2
    import pdb
    loss1 = 1*torch.sum(torch.log(torch.exp(Mask1*sim_matrix+eps))) / (1*B) +torch.sum(torch.log(torch.exp(Mask2*(1-sim_matrix)+eps)))/(2*B)
    return loss1


def JS_MI_pre(sim_matrix, labels, temperature=0.5, eps=1e-8):
    device = sim_matrix.device
    label = labels#.repeat(2)
    label = label.contiguous().view(-1, 1)

    # import pdb
    # pdb.set_trace()
    Mask1 = torch.eq(label, label.t()).float().to(device)
   # Mask2 = torch.ne(label, label.t()).float().to(device)

    Mask1 = Mask1 / (Mask1.sum(dim=1, keepdim=True) + eps)
    #Mask2 = Mask2 / (Mask2.sum(dim=1, keepdim=True) + eps)

    B = sim_matrix.size(0)  # // chunk
    eye = torch.eye(B).to(device)  # (B', B')
    sim_matrix = sim_matrix * (1 - eye)
    # sim_matrix = -torch.log(sim_matrix + eps)
    loss1 = torch.sum(torch.log(torch.exp(Mask1*sim_matrix+eps))) / B
    return loss1


def Supervised_NT_xent_n(sim_matrix, labels, embedding=None,temperature=0.5, chunk=2, eps=1e-8, multi_gpu=False):
    '''
        Compute NT_xent loss
        - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
    '''

    device = sim_matrix.device
  #  labels1 = labels
    labels1 = labels.repeat(2)


    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)

    sim_matrix = sim_matrix - logits_max.detach()


    B = sim_matrix.size(0) // chunk  # B = B' / chunk

    eye = torch.eye(B * chunk).to(device)  # (B', B')
    sim_matrix = torch.exp(sim_matrix / temperature) * (1 - eye)  # remove diagonal

    denom = torch.sum(sim_matrix, dim=1, keepdim=True)
 #   denom2 = torch.sum(mask*embedding,dim=1,keepdim=True)

    sim_matrix = -torch.log(sim_matrix/(denom+eps)+eps)  # loss matrix
  #  embedding = -torch.log(embedding/(denom2+eps)+eps)

#    sim_label = -torch.log(embedding/(denom2+eps)+eps)

    labels1 = labels1.contiguous().view(-1, 1)

    Mask1 = torch.eq(labels1, labels1.t()).float().to(device)

    Mask1 = Mask1 / (Mask1.sum(dim=1, keepdim=True) + eps)
    
    
    loss1 = 2*torch.sum(Mask1 * sim_matrix) / (2 * B)

    return (torch.sum(sim_matrix[:B, B:].diag() + sim_matrix[B:, :B].diag()) / (2 * B)) +  loss1#+1*loss2




def Supervised_NT_xent_uni(sim_matrix, labels, temperature=0.5, chunk=2, eps=1e-8, multi_gpu=False):
    '''
        Compute NT_xent loss
        - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
    '''

    device = sim_matrix.device
   # labels1 = labels
    labels1 = labels.repeat(2)


    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)

    sim_matrix = sim_matrix - logits_max.detach()

    B = sim_matrix.size(0) // chunk  # B = B' / chunk

    sim_matrix = torch.exp(sim_matrix / temperature)# * (1 - eye)  # remove diagonal

    denom = torch.sum(sim_matrix, dim=1, keepdim=True)

    sim_matrix = -torch.log(sim_matrix/(denom+eps)+eps)  # loss matrix

    labels1 = labels1.contiguous().view(-1, 1)

    Mask1 = torch.eq(labels1, labels1.t()).float().to(device)

    Mask1 = Mask1 / (Mask1.sum(dim=1, keepdim=True) + eps)

    a = 1

    return torch.sum(Mask1 * sim_matrix) / (2 * B)



   # return Loss
def Supervised_NT_xent_pre(sim_matrix, labels, temperature=0.5, chunk=2, eps=1e-8, multi_gpu=False):
    '''
        Compute NT_xent loss
        - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
    '''

    device = sim_matrix.device
  #  labels1 = labels
    labels1 = labels#.repeat(2)

    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)

    sim_matrix = sim_matrix - logits_max.detach()

    B = sim_matrix.size(0) // chunk  # B = B' / chunk

    sim_matrix = torch.exp(sim_matrix / temperature) #* (1 - eye)  # remove diagonal

    denom = torch.sum(sim_matrix, dim=1, keepdim=True)

    sim_matrix = -torch.log(sim_matrix/(denom+eps)+eps)  # loss matrix

    labels1 = labels1.contiguous().view(-1, 1)

    Mask1 = torch.eq(labels1, labels1.t()).float().to(device)

    Mask1 = Mask1 / (Mask1.sum(dim=1, keepdim=True) + eps)

    return torch.sum(Mask1 * sim_matrix) / (2 * B)




   # return Loss
def get_similarity_matrix(outputs, chunk=2, multi_gpu=False):
    '''
        Compute similarity matrix
        - outputs: (B', d) tensor for B' = B * chunk
        - sim_matrix: (B', B') tensor
    '''
    #sim_matrix = F.cosine_similarity(outputs.unsqueeze(1), outputs.unsqueeze(0), dim=-1)
    sim_matrix = torch.mm(outputs, outputs.t())  

    return sim_matrix

def Sup(sim_matrix, labels, temperature=0.5, chunk=2, eps=1e-8, multi_gpu=False):
    '''
        Compute NT_xent loss
        - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
    '''

    device = sim_matrix.device
    labels1 = labels
    labels1 = labels1.repeat(2)
    #labels2 = labels1.repeat(2)
    print("0",sim_matrix)

    B = sim_matrix.size(0) // chunk  # B = B' / chunk
    #print("BBB",B,chunk)
    eye = torch.eye(B * chunk).to(device)  # (B', B')

    sim_matrix = -torch.log(torch.max(sim_matrix,eps)[0])*(1-eye)  # loss matrix
    #print("ee3", sim_matrix.shape)
    labels1 = labels1.contiguous().view(-1, 1)

    Mask1 = torch.eq(labels1, labels1.t()).float().to(device)

    Mask1 = Mask1 / (Mask1.sum(dim=1, keepdim=True) + eps)

    a = 1
    #b = 1  # all is 1 means 2:1,-0.5&1 is1:2 no，all 1 is 1+1/n:n-1/n
   # print(a,b)
    #print("Ma",Mask.shape,sim_matrix.shape)
    loss1 = torch.sum(Mask1 * sim_matrix) / (2 * B)

    Loss = a* loss1#+1*loss2


    return Loss

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
           # labels = labels.repeat(2)

            labels = labels.contiguous().view(-1, 1)

            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
         #   print("mask",mask)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]#2
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        #print("log",logits_mask.shape)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        #print(mean_log_prob_pos.shape)
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()#单个view

        return loss
