
""" Sampler for dataloader. """
import torch
import numpy as np
from torch.utils.data import Sampler
import torch
import numpy as np
import random
mseed = 48


random.seed(mseed)
np.random.seed(mseed)
torch.manual_seed(mseed)
torch.cuda.manual_seed(mseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# class CategoriesSampler():
#     """The class to generate episodic data"""
#     def __init__(self, label, n_batch, n_cls, n_per):
#         self.n_batch = n_batch
#         self.n_cls = n_cls
#         self.n_per = n_per

#         label = np.array(label)
       
#         self.m_ind = []
#         for i in range(n_cls):
#             ind = np.argwhere(label == i).reshape(-1)
#             ind = torch.from_numpy(ind)
#             self.m_ind.append(ind)

#     def __len__(self):
#         return self.n_batch
#     def __iter__(self):
#         for i_batch in range(self.n_batch):
#             batch = []
#             for c in range(self.n_cls):
#                 l = self.m_ind[c]
#                 if len(l) >= self.n_per:
#                     pos = torch.randperm(len(l))[:self.n_per]
#                 elif len(l) == 0:
#                     #pos = torch.tensor([])
#                     continue
#                 else:
#                     pos = torch.arange(len(l)).repeat(self.n_per // len(l) + 1)
#                     pos = pos[torch.randperm(len(pos))[:self.n_per]]

#                 if len(pos) != 0:
#                     batch.append(l[pos])
            
#             batch = torch.stack(batch).t().reshape(-1)
#             yield batch


class CategoriesSampler(Sampler):
    """The class to generate episodic data with a specific sequence of classes"""
    def __init__(self, labels, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        labels = np.array(labels)
        self.m_ind = []
        for i in range(n_cls):
            ind = np.argwhere(labels == i).reshape(-1)
            self.m_ind.append(torch.from_numpy(ind))

        # Create a sequence of classes
        self.class_sequence = list(range(n_cls))

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for _ in range(self.n_batch):
            batch = []
            for c in self.class_sequence:
                l = self.m_ind[c]
                if len(l) == 0:
                    # Skip if there are no samples for this class
                    continue
                if len(l) >= self.n_per:
                    pos = torch.randperm(len(l))[:self.n_per]
                else:
                    # Handle cases where we have fewer samples than needed
                    pos = torch.arange(len(l)).repeat(self.n_per // len(l) + 1)
                    pos = pos[torch.randperm(len(pos))[:self.n_per]]

                batch.append(l[pos])

            # Ensure there are no empty batches
            if len(batch) == 0:
                continue

            # Stack batches and ensure correct shape
            batch = torch.stack(batch).t().reshape(-1)
            yield batch
