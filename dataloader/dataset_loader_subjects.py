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
class DatasetLoader_subjects(Dataset):

    def __init__(self, setname, args, subject_id, train_aug=False):
     
        if args.dataset == 'AMIGOS':
            data,label = self.load_amigos_data(setname, args, subject_id)
        elif args.dataset == 'DEAP':
            data,label = self.load_deap_data(setname, args, subject_id)
        elif args.dataset == 'BCI_IV_2a':
            data,label = self.load_BCI_IV_2a_data(setname, args, subject_id)
        elif args.dataset == 'PPB_EMO':
            data,label = self.load_PPB_EMO_data(setname, args, subject_id)

        self.data = data
        self.label = label
        self.num_class=4

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # print("i:", i)
        # print("len(self.data):", len(self.data))
        data, label=self.data[i], self.label[i]
        return data, label


    def load_PPB_EMO_data(self, setname, args, subject_id):
        video_id = ['AD', 'DD', 'FD', 'HD', 'ND', 'SAD','SD']

        sampling_rate = 250  # Hz
        segment_length_sec = 3  # seconds
        stride_rate =  0.50 # 50% overlap

        # get data and labels for all video trial
        segmented_data_all = []
        categorized_labels_all = []

        # load labels
        all_labels_csv = self.load_labels_emo()


        for video_id_s in video_id:
            data, video_id_s = self.load_data_mat_emo(subject_id, video_id_s)

            if data is None:
                continue

            start_index = max(0, data.shape[1] - 32)
            data = data.iloc[:, start_index:]

            # fetch labels for the video and categorize
            label = all_labels_csv[all_labels_csv['PPB_Emo_dataset@Physiological_data'] == video_id_s]
            label_catg = self.categorize_val_arousal_self_emo(label['valence'].values[0], label['arousal'].values[0])

            # data segmentation
            segmented_data = self.segment_data(data, segment_length_sec, stride_rate, sampling_rate)
            segmented_labels = np.ones(segmented_data.shape[0]) * label_catg

            segmented_data_all.extend(segmented_data)
            categorized_labels_all.extend(segmented_labels)        



        # data numpy
        segmented_data_all = np.array(segmented_data_all)
        # replace nan with 0
        segmented_data_all = np.nan_to_num(segmented_data_all)

        # normalize data
        scaler = StandardScaler()
        #segmented_data_all = scaler.fit_transform(segmented_data_all)

        # swap axes 1,2
        segmented_data_all = np.swapaxes(segmented_data_all, 1, 2)

        # convert to float32
        segmented_data_all = segmented_data_all.astype(np.float32)
        # convert label to float32
        categorized_labels_all = np.array(categorized_labels_all).astype(np.float32)

        # split data
        X_train, X_test, y_train, y_test = train_test_split(segmented_data_all, categorized_labels_all, test_size=0.2, random_state=42) 

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        y_train = y_train.astype('float32')
        y_test = y_test.astype('float32')

        if setname == 'train':
            return X_train, y_train
        elif setname == 'val':
            return X_test, y_test
        elif setname == 'test':
            return X_test, y_test

    def load_BCI_IV_2a_data(self, setname, args, subject_id):

        data_folder='./data'
        data = sio.loadmat(data_folder+"/cross_sub/BCI_IV_2a/data_cross_subject"+".mat")
        # test_X	= data["test_x"][:,:,750:1500] 
        # val_X	= data["val_x"][:,:,750:1500]
        # train_X	= data["train_x"][:,:,750:1500]
        
        test_X	= data["test_x"]
        val_X	= data["val_x"]
        train_X	= data["train_x"]


        test_y	= data["test_y"].ravel()
        val_y = data["val_y"].ravel()
        train_y = data["train_y"].ravel()
        
        subject_id_train=data["subject_id_train"].ravel()
        subject_id_val=data["subject_id_val"].ravel()
        subject_id_test=data["subject_id_test"].ravel()

        # total number of classes in the train_X
        n_classes = len(np.unique(train_y))
        #print('n_classes', n_classes)



        # train_y-=1
        # val_y-=1
        # test_y-=1

        window_size = 400
        step = 360
        n_channel = 22  
        
        def windows(data, size, step):
            start = 0
            while ((start+size) < data.shape[0]):
                yield int(start), int(start + size)
                start += step

        def segment_signal_without_transition(data, window_size, step):
            #print('data', data.shape)
            segments = []
            for (start, end) in windows(data, window_size, step):
                if(len(data[start:end]) == window_size):
                    segments = segments + [data[start:end]]
            return np.array(segments)


        def segment_dataset(X, window_size, step):
            win_x = []
            for i in range(X.shape[0]):
                win_x = win_x + [segment_signal_without_transition(X[i], window_size, step)]
            win_x = np.array(win_x)
            return win_x

        train_raw_x = np.transpose(train_X, [0, 2, 1])
        val_raw_x = np.transpose(val_X, [0, 2, 1])
        test_raw_x = np.transpose(test_X, [0, 2, 1])
        #print('train_raw_x', train_raw_x.shape)

        train_win_x = segment_dataset(train_raw_x, window_size, step)
        val_win_x = segment_dataset(val_raw_x, window_size, step)
        test_win_x = segment_dataset(test_raw_x, window_size, step)
        #print('train_win_x', train_win_x.shape)


        expand_factor=train_win_x.shape[1]

        train_x=np.reshape(train_win_x,(-1,train_win_x.shape[2], train_win_x.shape[3]))  
        val_x=np.reshape(val_win_x,(-1,val_win_x.shape[2], val_win_x.shape[3]))  
        test_x=np.reshape(test_win_x, (-1, test_win_x.shape[2], test_win_x.shape[3]))
        
        train_y=np.repeat(train_y, expand_factor)
        val_y=np.repeat(val_y, expand_factor)
        test_y=np.repeat(test_y, expand_factor)

        train_win_y=train_y
        val_win_y=val_y
        test_win_y=test_y

        # train_x=np.reshape(train_x, [train_x.shape[0], 1, train_x.shape[1], train_x.shape[2]]).astype('float32')
        # train_y=np.reshape(train_y, [train_y.shape[0]]).astype('float32')
        # #print('train_x', train_x.shape)
        
        # val_x=np.reshape(val_x, [val_x.shape[0], 1, val_x.shape[1], val_x.shape[2]]).astype('float32')
        # val_y=np.reshape(val_y, [val_y.shape[0]]).astype('float32')
        
        # test_x=np.reshape(test_x, [test_x.shape[0], 1, test_x.shape[1], test_x.shape[2]]).astype('float32')
        # test_y=np.reshape(test_y, [test_y.shape[0]]).astype('float32')

        #swap axes to match the shape of the input to the model
        train_x = np.swapaxes(train_x, 1, 2)
        val_x = np.swapaxes(val_x, 1, 2)
        test_x = np.swapaxes(test_x, 1, 2)

        train_win_x=train_x.astype('float32')
        val_win_x=val_x.astype('float32')
        test_win_x=test_x.astype('float32')
    

        train_win_x=train_win_x.astype('float32') 
        val_win_x=val_win_x.astype('float32') 
        test_win_x=test_win_x.astype('float32')  

        list_subject_train=[i for i, e in enumerate(subject_id_train) if e == subject_id]
        list_subject_val=[i for i, e in enumerate(subject_id_val) if e == subject_id]
        list_subject_test=[i for i, e in enumerate(subject_id_test) if e == subject_id]

        X_train_s=train_win_x[list_subject_train]
        y_train_s=train_win_y[list_subject_train]
        X_val_s=val_win_x[list_subject_val]
        y_val_s=val_win_y[list_subject_val]
        X_test_s=test_win_x[list_subject_test]
        y_test_s=test_win_y[list_subject_test]
        #print('X_train_s', X_train_s.shape)

        if setname == 'train':
            return X_train_s, y_train_s
        elif setname == 'val':
            return X_val_s, y_val_s
        elif setname == 'test':
            return X_test_s, y_test_s


    def load_deap_data(self, setname, args, subject_id):

        data_folder='./data/cross_sub/DEAP/s'+str(subject_id).zfill(2)+'.dat'
        
        # last 50 seconds of the data 
        remove_few_sec = 3 + 10
        sampling_rate = 128
        segment_size = 768
        stride =  segment_size // 2 # 50% overlap
        num_trials = 40
        num_segments_per_trial = 15# 45
        num_channels = 32



        with open(data_folder, 'rb') as f:
            dataset = pickle.load(f, encoding='latin1')

        labels = dataset['labels']

        eeg_data = dataset['data'][:,:32,remove_few_sec* sampling_rate:]

        # Calculate the number of segments per trial
        num_segments = (eeg_data.shape[2] - segment_size) // stride + 1

        # Initialize lists to store segmented data and corresponding labels
        segmented_data = []
        segmented_labels = []

        # Iterate over each trial
        for i in range(eeg_data.shape[0]):
            trial_data = eeg_data[i]  # EEG data for current trial
            trial_labels = labels[i]  # Labels for current trial
            trial_labels = self.categorize_sample(trial_labels[0], trial_labels[1])

            # Initialize arrays to store segments and corresponding labels for current trial
            trial_segmented_data = []
            trial_segmented_labels = []

            # Iterate over segments within the current trial
            for j in range(num_segments):
                start = j * stride
                end = start + segment_size
                
                # Extract segment from EEG data
                segment = trial_data[:, start:end]
                trial_segmented_data.append(segment)
                
                # Use the same segment index for labels (assuming labels are trial-level)
                trial_segmented_labels.append(trial_labels)
            
            # Append segmented data and labels for current trial to the main lists
            segmented_data.append(trial_segmented_data)
            segmented_labels.append(trial_segmented_labels)

        # Convert lists to numpy arrays for easier manipulation (if needed)
        segmented_data = np.array(segmented_data)
        segmented_labels = np.array(segmented_labels)
        num_trials = segmented_data.shape[0]

        segmented_data = segmented_data.reshape(num_trials * num_segments_per_trial,num_channels, segment_size)
        segmented_labels = segmented_labels.reshape(num_trials * num_segments_per_trial)

        #print(segmented_data.shape)

        # replace nan with 0
        segmented_data[np.isnan(segmented_data)] = 0

        # swap axes to match the shape of the input to the model
        #segmented_data = np.swapaxes(segmented_data, 1, 2)

        # float 32
        segmented_data = segmented_data.astype(np.float32)
        # label to float32
        segmented_labels = segmented_labels.astype(np.float32)

        # # Check the shape of segmented data and labels
        # 
        # reshaped_data = np.transpose(reshaped_data, (0, 1,3, 2))
        # # reshape labels
        # 



        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(segmented_data, segmented_labels, test_size=0.2, random_state=42, stratify=segmented_labels)

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        y_train = y_train.astype('float32')

        if setname == 'train':
            return X_train, y_train
        elif setname == 'val':
            return X_test, y_test
        elif setname == 'test':
            return X_test, y_test



    def load_amigos_data(self, setname, args, subject_id):
        scaler = StandardScaler()
        sampling_rate = 128# 256# 128# 256 # Hz
        segment_length_sec = 5# 10# 7 #10  # seconds
        stride_rate =  0.50 # 50% overlap`

        # load subject 1 data
        combined_data = self.load_data_mat(subject_id)
        # get data and labels for all video trial
        segmented_data_all = []
        categorized_labels_all = []
        for video_idx in range(combined_data['joined_data'].shape[1] - 4):
            data = combined_data['joined_data'][0,video_idx]
            labels_self = combined_data['labels_selfassessment'][0,video_idx][:,0:2]
            labels_self = self.categorize_val_arousal_self(labels_self)
            #labels_ext_annot = combined_data['labels_ext_annotation'][0,video_idx]

            # Segment the data with zero-padding
            segmented_data = self.segment_data(data, segment_length_sec, stride_rate, sampling_rate)
            segmented_labels = np.repeat(np.array(labels_self),len(segmented_data),axis=0)
            
            segmented_data_all.extend(segmented_data)
            categorized_labels_all.extend(segmented_labels)
        
        # remove i from segmented_data_all and categorized_labels_all if it has label is 4
        segmented_data_all = [segmented_data_all[i] for i in range(len(segmented_data_all)) if categorized_labels_all[i] != 4]
        categorized_labels_all = [categorized_labels_all[i] for i in range(len(categorized_labels_all)) if categorized_labels_all[i] != 4]       
        # now separate EEG, ECG and GSR data
        frames_EEG = []
        frames_ECG = []
        frames_GSR = []

        for segment in segmented_data_all:
            frames_EEG.append(segment[:,0:14])
            frames_ECG.append(segment[:,14:16])
            frames_GSR.append(segment[:,-1])

        # convert into arrays and swap last two dimensions
        frames_EEG = np.array(frames_EEG)
        frames_ECG = np.array(frames_ECG)
        frames_GSR = np.array(frames_GSR)
        #print(frames_EEG.shape)
        # replace nan with zeros
        frames_EEG[np.isnan(frames_EEG)] = 0
        frames_ECG[np.isnan(frames_ECG)] = 0
        frames_GSR[np.isnan(frames_GSR)] = 0

        frames_EEG = np.swapaxes(frames_EEG, 1, 2)
        frames_ECG = np.swapaxes(frames_ECG, 1, 2)

        # convert to float32
        frames_EEG = frames_EEG.astype(np.float32)
        frames_ECG = frames_ECG.astype(np.float32)
        frames_GSR = frames_GSR.astype(np.float32)

        # convert labels to float32
        categorized_labels_all = np.array(categorized_labels_all)
        categorized_labels_all = categorized_labels_all.astype(np.float32)

        # Split into training and validation sets EEG

        X_train_EEG, X_val_EEG, y_train_EEG, y_val_EEG = train_test_split(frames_EEG, categorized_labels_all, test_size=0.20, random_state=42,stratify=categorized_labels_all)

        X_train_ECG, X_val_ECG, y_train_ECG, y_val_ECG = train_test_split(frames_ECG, categorized_labels_all, test_size=0.20, random_state=42,stratify=categorized_labels_all)

        if args.modality == 'EEG':
            if setname == 'train':
                return X_train_EEG, y_train_EEG
            elif setname == 'val':
                return X_val_EEG, y_val_EEG
            elif setname == 'test':
                return X_val_EEG, y_val_EEG
        elif args.modality == 'ECG':
            if setname == 'train':
                return X_train_ECG, y_train_ECG
            elif setname == 'val':
                return X_val_ECG, y_val_ECG
            elif setname == 'test':
                return X_val_ECG, y_val_ECG

    #load mat file in pandas dataframe and display

    def load_data_mat(self, subject_id):
        data_folder='./data/cross_sub/AMIGOS/Data_Preprocessed_P' + str(subject_id).zfill(2) + '/Data_Preprocessed_P' + str(subject_id).zfill(2) + '.mat'
        data = sio.loadmat(data_folder)
        return data

    
    def categorize_val_arousal_self(self, label):
        val = label[0][0]
        arousal = label[0][1]

        if val >= 5 and arousal >= 5:
            return 0
        elif val >= 5 and arousal < 5:
            return 1
        elif val < 5 and arousal < 5:
            return 2
        elif val < 5 and arousal >= 5:
            return 3
        else:
            return 4 

    def categorize_sample(self,valence, arousal):
        if valence > 5 and arousal > 5:
            return 0
        elif valence > 5 and arousal <= 5:
            return 1
        elif valence <= 5 and arousal > 5:
            return 2
        elif valence <= 5 and arousal <= 5:
            return 3
        else:
            raise ValueError("Invalid valence or arousal values")
    # def categorize_val_arousal_self(self, label):
    #     val = label[0][0]
    #     arousal = label[0][1]

    #     if val > 6 and arousal > 6:
    #         return 0
    #     elif val > 6 and arousal < 4:
    #         return 1
    #     elif val < 4 and arousal < 4:
    #         return 2
    #     elif val < 4 and arousal > 6:
    #         return 3
    #     else:
    #         return 4 

    def segment_data(self,data, segment_length_sec, stride_rate=0.5, sampling_rate=128):
        segment_length = segment_length_sec * sampling_rate  # Convert segment length from seconds to samples
        stride = int(segment_length * stride_rate)  # Calculate stride length in samples

        # Calculate how much zero-padding is needed
        pad_length = segment_length - (len(data) % segment_length)
        
        # Pad data with zeros to ensure it can be evenly divided into segments
        if pad_length > 0:
            data = np.pad(data, ((0, pad_length), (0, 0)), mode='constant')

        segments = []
        for start in range(0, len(data) - segment_length + 1, stride):
            segment = data[start:start + segment_length]
            segments.append(segment)
        
        return np.array(segments)
    
    def load_data_mat_emo(self,subject_id, video_id):

        data_folder='./data/cross_sub/ppb_emo/Physiological_data/P' + str(subject_id).zfill(2) + '/PPB_Emo_dataset@EEG-30s-P' + str(subject_id).zfill(2)+'-' +video_id + '.csv'

        # Check if the file exists
        if os.path.exists(data_folder):

            df = pd.read_csv(data_folder)
            video_id_ = 'PPB_Emo_dataset@EEG-30s-P' + str(subject_id).zfill(2)+'-' +video_id
            return df, video_id_
        else:
            return None, None

    def load_labels_emo(self):
        data_folder='./data/cross_sub/ppb_emo/EE Data/Emotion_label.xlsx'
        return pd.read_excel(data_folder, engine='openpyxl')


    def categorize_val_arousal_self_emo(self, val, arousal):

        if val >= 5 and arousal >= 5:
            return 0
        elif val >= 5 and arousal < 5:
            return 1
        elif val < 5 and arousal < 5:
            return 2
        else:
            return 3 
    
    