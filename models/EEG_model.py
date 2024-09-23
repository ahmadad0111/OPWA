import torch
import torch.nn as nn
import torch.nn.functional as F


embedding_dim = 128

class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.W = nn.Linear(input_dim, input_dim)
        self.v = nn.Linear(input_dim, 1, bias=False)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        scores = self.W(x)  # (batch_size, seq_len, input_dim)
        scores = torch.tanh(scores)  # Apply non-linearity
        scores = self.v(scores).squeeze(-1)  # (batch_size, seq_len)
        weights = F.softmax(scores, dim=1)  # (batch_size, seq_len)
        weighted_sum = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (batch_size, input_dim)
        return weighted_sum


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_sizes[0], stride=stride, padding=(kernel_sizes[0] - 1) // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_sizes[1], stride=stride, padding=(kernel_sizes[1] - 1) // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        residual = self.shortcut(residual)
        out += residual
        out = self.relu(out)
        return out
    
class EEGNet1D(nn.Module):
    def __init__(self,args, num_classes=4, embedding_dim=128, num_heads=8):
        super(EEGNet1D, self).__init__()


        if args.dataset == 'AMIGOS':
            #modality
            if args.modality == 'EEG':
                self.in_channels = 14
            elif args.modality == 'ECG':
                self.in_channels = 2
        elif args.dataset == 'DEAP':
            self.in_channels = 32
        elif args.dataset == 'BCI_IV_2a':
            self.in_channels = 22
        elif args.dataset == 'PPB_EMO':
            self.in_channels = 32
        # Initial Conv1D Layer
        self.conv1 = nn.Conv1d(in_channels=self.in_channels, out_channels=32, kernel_size=7, stride=2, padding=0)
        self.bn1 = nn.BatchNorm1d(32)
        
        # Residual Blocks with different kernel sizes
        self.res_block1 = ResidualBlock(in_channels=32, out_channels=64, kernel_sizes=[15, 15])
        self.res_block2 = ResidualBlock(in_channels=64, out_channels=128, kernel_sizes=[21, 21])
        self.res_block3 = ResidualBlock(in_channels=128, out_channels=128, kernel_sizes=[43, 43])

        # Pooling
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)
        
        # Single Head Attention Layer
        self.attention = AttentionLayer(input_dim=128)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(128, 1024)
        #self.dropout1 = nn.Dropout(0.6)
        self.fc2 = nn.Linear(1024, 512)
        #self.dropout2 = nn.Dropout(0.6)
        self.fc3 = nn.Linear(512, 256)
        self.embedding = nn.Linear(256, embedding_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        
        x = self.pool(x)
        x = x.transpose(1, 2)  # Transpose to (batch_size, seq_len, input_dim) for Attention Layer
        x = self.attention(x)  # Apply attention
        
        x = F.relu(self.fc1(x))
        #x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        #x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.embedding(x)
        
        return x
    
# class EEGNet1D(nn.Module):
#     def __init__(self,args, num_classes=4,embedding_dim=128):
#         super(EEGNet1D, self).__init__()

#         if args.dataset == 'AMIGOS':
#             self.flat_dim = 128*41
#             #modality
#             if args.modality == 'EEG':
#                 self.in_channels = 14
#             elif args.modality == 'ECG':
#                 self.in_channels = 2
#         elif args.dataset == 'DEAP':
#             self.in_channels = 32
#             self.flat_dim = 128*89
#         elif args.dataset == 'BCI_IV_2a':
#             self.in_channels = 22
#             self.flat_dim = 128*43
#         elif args.dataset == 'PPB_EMO':
#             self.in_channels = 32
#             self.flat_dim = 128*87
#         # Initial Conv1D Layer
#         self.conv1 = nn.Conv1d(in_channels=self.in_channels, out_channels=32, kernel_size=7, stride=2, padding=0)
#         self.bn1 = nn.BatchNorm1d(32)
#         # Depthwise Separable Convolutions
#         self.ds_conv1 = nn.Conv1d(32, 64, kernel_size=15, stride=1, padding=8, bias=True)
#         self.bn2 = nn.BatchNorm1d(64)
#         self.ds_conv2 = nn.Conv1d(64, 128, kernel_size=43, stride=1, padding=8, bias=True)
#         self.bn3 = nn.BatchNorm1d(128)
        
#         # Pooling
#         self.pool = nn.MaxPool1d(kernel_size=4, stride=4)
        
#         # Fully Connected Layers
#         self.fc1 = nn.Linear(self.flat_dim, 512)  # Adjust input size based on output size of conv layers
#         # drop out
#         # self.dropout = nn.Dropout(0.5)
#         # self.fc2 = nn.Linear(512, 128)
#         # self.fc3 = nn.Linear(128, num_classes)
#         self.embedding = nn.Linear(512, embedding_dim)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(self.bn1(x))
        
#         x = self.ds_conv1(x)
#         x = F.relu(self.bn2(x))
#         x = self.ds_conv2(x)
#         x = F.relu(self.bn3(x))
        
#         x = self.pool(x)
#         x = x.view(x.size(0), -1)  # Flatten
        
#         x = F.relu(self.fc1(x))
#         #x = self.dropout(x)
#         #x = F.relu(self.fc2(x))
#         #x = self.fc3(x)
#         # probability
#         #x = F.softmax(x, dim=1)
#         x = self.embedding(x)
        
#         return x

class ClassifierEEGNet1D(nn.Module):
    def __init__(self,args, num_classes=4,embedding_dim=128):
        super(ClassifierEEGNet1D, self).__init__()
        self.eegnet = EEGNet1D(args=args,embedding_dim=embedding_dim)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.eegnet(x)
        x = self.fc(x)
        # softmax
        #x = F.softmax(x, dim=1)
        return x


class MetaClassifierEEGNet1D(nn.Module):
    def __init__(self,args, num_classes=4,embedding_dim=128):
        super(MetaClassifierEEGNet1D, self).__init__()
        self.eegnet = EEGNet1D(args=args,embedding_dim=embedding_dim)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, data,label,mode = 'meta'):
        
        if mode == 'meta':
            return self.meta_forward(data,label)
        else:
            x = self.eegnet(x)
            x = self.fc(x)
            # softmax
            #x = F.softmax(x, dim=1)
            return x
    def meta_forward(self,data,label):
        return data #############