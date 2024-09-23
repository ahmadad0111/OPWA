import torch
from torch.nn import functional as F
from torch.nn.modules import Module

class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''
    def __init__(self,embedding_dim = 128, n_classes=4, momentum=0.9):
        super(PrototypicalLoss, self).__init__()
        self.alpha = momentum
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = n_classes
        #self.prototypes = None  # Initialize prototypes as None
        self.prototypes = torch.zeros((self.num_classes, embedding_dim), device=self.device) # Initialize prototypes as global prototypes

        self.current_prototypes = torch.zeros((self.num_classes, embedding_dim), device=self.device) # Initialize prototypes as global prototypes
        

    def forward(self, input, target, update_prototypes = True):
        with torch.no_grad():  # Ensure that prototype updates are not tracked by autograd
            if update_prototypes:
                self.update_prototypes(input, target)
        return self.prototypical_loss(input, target), self.compute_accuracy(input, target)

    def euclidean_dist(self, x, y):
        '''
        Compute euclidean distance between two tensors
        '''
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        if d != y.size(1):
            raise Exception

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        return torch.pow(x - y, 2).sum(2)

    def update_prototypes(self, embeddings, labels):
        '''
        Update prototypes using momentum-based moving average
        '''
        prototypes = torch.zeros(self.num_classes, embeddings.size(1)).to(embeddings.device)
        counts = torch.zeros(self.num_classes).to(embeddings.device)


        for i in range(self.num_classes):
            mask = (labels == i)
            if mask.sum() > 0:
                prototypes[i] = embeddings[mask].mean(0)
                counts[i] = mask.sum()

        self.current_prototypes = prototypes

        # Update moving average of prototypes
        if self.prototypes.sum() == 0:
            self.prototypes = prototypes
        else:
            self.prototypes = self.alpha * self.prototypes + (1 - self.alpha) * prototypes

        
        # normalize the prototypes
        self.prototypes = F.normalize(self.prototypes, p=2, dim=1)

    def prototypical_loss(self, embeddings, labels):
        '''
        Compute the prototypical loss
        '''
        
        #  normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        

        distances = torch.cdist(embeddings, self.prototypes,p=2)  # Euclidean distance

        # distance with current prototypes  
        #distances = torch.cdist(embeddings, self.current_prototypes,p=2)  # Euclidean distance

        # dot product between them
        #distances = torch.mm(embeddings, self.prototypes.t())

        #print(distances.size())
        log_p_y = -distances
        log_p_y = F.log_softmax(log_p_y, dim=1)
        loss = F.nll_loss(log_p_y, labels)


        return loss#/ len(labels)
    
    def compute_accuracy(self, embeddings, labels):

        #  normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        # Compute pairwise distances
        distances = torch.cdist(embeddings, self.prototypes, p=2)  # Euclidean distance
        
        # Predict classes by finding the minimum distance prototype
        _, predicted_classes = torch.min(distances, dim=1)
        
        # Compute accuracy
        correct = (predicted_classes == labels).sum().item()
        accuracy = correct / labels.size(0)
        return accuracy
