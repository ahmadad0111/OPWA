import numpy as np
import time
import scipy
import scipy.signal
import scipy.io
# import self defined functions 
from torch.utils.data import Dataset
import random
import scipy.io as sio
import math
from scipy import interp
import pandas as pd
import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
mseed = 48


random.seed(mseed)
np.random.seed(mseed)
class DatasetLoader_subjectss(Dataset):

    def __init__(self, setname, args, X_train, y_train, X_val, y_val, X_test, y_test):


        if setname == 'train':
            self.data =  X_train
            self.label = y_train
        elif setname == 'val':
            self.data = X_val
            self.label = y_val
        elif setname == 'test':
            self.data = X_test
            self.label = y_test

        self.num_class=4

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # print("i:", i)
        # print("len(self.data):", len(self.data))
        data, label=self.data[i], self.label[i]
        return data, label

