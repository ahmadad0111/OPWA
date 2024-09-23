""" Feature Extractor """
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
mseed = 48


if mseed == 0:
    torch.backends.cudnn.benchmark = True
else:
    torch.manual_seed(mseed)
    torch.cuda.manual_seed(mseed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class FeatureExtractor(nn.Module):

    def __init__(self, mtl=True):
        super(FeatureExtractor, self).__init__()
        self.Conv2d = nn.Conv2d
        self.conv1 = nn.Conv2d(1, 16, (1, 14), padding = 0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)
        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)
        
        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))

    def forward(self, x):
        
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = x.permute(0, 3, 1, 2)
        # Layer 2
        
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = self.pooling2(x)
        # Layer 3
        
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = self.pooling3(x)
        # x = x.reshape(-1, 4 *2 * 48)
        #x = x.reshape(-1, 15360)
        #x = x.reshape(-1, 4*2*25)
        x = x.reshape(-1, 4 *2 * 56) # 128 rate and 7 sec
        #x = x.reshape(-1, 4 *2 * 160) # 256 rate and 10 sec
        
        

        return x


class FeatureExtractorCNN(nn.Module):

    def __init__(self, mtl=True):
        super(FeatureExtractorCNN, self).__init__()
        # ECG Part
        self.S_ECGc1 = nn.Conv1d(in_channels=2, out_channels=26, kernel_size=1, stride=1, padding=0)
        self.S_ECGBN1 = nn.BatchNorm1d(26)
        self.S_ECGMP1 = nn.MaxPool1d(kernel_size=2, padding=0)

        self.S_ECGc2 = nn.Conv1d(in_channels=26, out_channels=26, kernel_size=1, stride=2)
        self.S_ECGBN2 = nn.BatchNorm1d(26)
        self.S_ECGMP2 = nn.MaxPool1d(kernel_size=2, padding=0)
        self.S_ECGGAP1 = nn.AdaptiveAvgPool1d(1)

        self.S_ECG_d1 = nn.Linear(26, 2100)
        self.S_ECG_drop = nn.Dropout(0.4)
        self.S_ECG_d2 = nn.Linear(2100, 1600)
        self.S_ECG_d3 = nn.Linear(1600, 800)
        self.S_ECG_d4 = nn.Linear(800, 400)
        #self.S_ECG_out = nn.Linear(400, nb_classes)

    def forward(self, x):
        
        # ECG Part
        x = F.relu(self.S_ECGc1(x))
        x = self.S_ECGBN1(x)
        x = self.S_ECGMP1(x)

        x = F.relu(self.S_ECGc2(x))
        x = self.S_ECGBN2(x)
        x = self.S_ECGMP2(x)
        x = self.S_ECGGAP1(x)

        x = F.relu(self.S_ECG_d1(x.squeeze()))
        x = self.S_ECG_drop(x)
        x = F.relu(self.S_ECG_d2(x))
        x = F.relu(self.S_ECG_d3(x))
        x = F.relu(self.S_ECG_d4(x))
        #S_ECG_out = torch.sigmoid(self.S_ECG_out(x))
        # use softmax
        #S_ECG_out = F.softmax(x, dim=1)
        
        return x



