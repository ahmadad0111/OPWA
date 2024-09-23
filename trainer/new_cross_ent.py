""" Trainer for meta-train phase. """
import os.path as osp
import os
import tqdm
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from InfoNCE import tao as TL
from InfoNCE.utils import normalize
from InfoNCE.contrastive_learning import get_similarity_matrix,Supervised_NT_xent_pre,Supervised_NT_xent_n,Supervised_NT_xent_uni
from buffer import Buffer
from Meta_optimizer import Meta_Optimizer
from torch.utils.data import DataLoader
from dataloader.samplers import CategoriesSampler#,VariableCategoriesSampler
from models.mtl import MtlLearner
from models.EEG_model import EEGNet1D, ClassifierEEGNet1D
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score #f1_score
from sklearn.preprocessing import LabelBinarizer
from utils.misc import Averager, Timer, count_acc, compute_confidence_interval, ensure_path
from tensorboardX import SummaryWriter
from utils.early_stopping import EarlyStopping
#from dataloader.dataset_loader_subjects import DatasetLoader_subjects as Dataset
from dataloader.data_preprocess import DataPreprocess
from dataloader.data_loader import DatasetLoader_subjectss as Dataset
import csv
from itertools import zip_longest
from sklearn.metrics import f1_score, roc_auc_score
from buffer_class_balanced import Buffer_class_balanced
from torchvision.transforms import v2
import time
from copy import deepcopy
import copy
import random
from prototypical.prototypical import PrototypicalLoss

from utils.aggregation_metrics import aggregate_subj_embeddings

class NewCrossTrainer(object):
    """The class that contains the code for the meta-train phase and meta-eval phase."""
    def __init__(self, args):

        # Set manual seed for PyTorch
        if args.seed == 0:
            print('Using random seed.')
            torch.backends.cudnn.benchmark = True
        else:
            print('Using manual seed:', args.seed)
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        # Set the folder to save the records and checkpoints
        log_base_dir = './logs/'
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)
        meta_base_dir = osp.join(log_base_dir, 'meta')
        if not osp.exists(meta_base_dir):
            os.mkdir(meta_base_dir)
        save_path1 = '_'.join([args.dataset, args.model_type, 'MTL'])
        save_path2 = 'shot' + str(args.shot) + '_way' + str(args.way) + '_query' + str(args.train_query) + \
            '_step' + str(args.step_size) + '_gamma' + str(args.gamma) + '_lr1' + str(args.meta_lr1) + '_lr2' + str(args.meta_lr2) + \
            '_batch' + str(args.num_batch) + '_maxepisode' + str(args.max_episode) + \
            '_baselr' + str(args.base_lr) + '_updatestep' + str(args.update_step) + \
            '_stepsize' + str(args.step_size) + '_' + args.meta_label
        args.save_path = meta_base_dir + '/' + save_path1 + '_' + save_path2
        ensure_path(args.save_path)

        self.model_path = './saved_models/model.pth'

        # Set args to be shareable in the class
        self.args = args
        # Set the device
        if args.gpu is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:    
            self.device = torch.device('cpu')

        print('Using GPU:', self.device)

        #self.model = MtlLearner(self.args)

        #self.model = EEGNet1D(args = self.args,num_classes=self.args.way,embedding_dim=self.args.z_pred)
        self.model =ClassifierEEGNet1D(args = self.args,num_classes=self.args.way,embedding_dim=self.args.z_pred)


        self.model.to(self.device)
        # Set the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.base_lr)

        # lr scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_size, gamma=self.args.gamma)

        self.buffer_size = 200
        self.buffer_batch_size = 32
        self.buffer = Buffer(self.buffer_size, device=self.device)

        self.buffer_class_balanced = Buffer_class_balanced(self.buffer_size, self.args.way, device=self.device,model = self.model, z_pred = self.args.z_pred)


        with torch.no_grad():
            self.sim_aug = v2.Compose([v2.RandomAffine(10, (0.01, 0.01)) ])


        self.dataprocess = DataPreprocess(self.args)
        
        self.x_train_s = {}
        self.y_train_s = {}
        self.x_val_s = {}
        self.y_val_s = {}
        self.x_test_s = {}
        self.y_test_s = {}

        for subject_id in range(self.args.start_range, self.args.subject_range):
            x_train, y_train, x_val, y_val, x_test, y_test = self.dataprocess.data_fetch(subject_id)
            self.x_train_s[subject_id] = x_train
            self.y_train_s[subject_id] = y_train
            self.x_val_s[subject_id] = x_val
            self.y_val_s[subject_id] = y_val
            self.x_test_s[subject_id] = x_test
            self.y_test_s[subject_id] = y_test
    def save_model(self, name):
        """The function to save checkpoints.
        Args:
          name: the name for saved checkpoint
        """  
        torch.save(dict(params=self.model.state_dict()), osp.join(self.args.save_path, name + '.pth'))           

    def train(self):
        """The function for the meta-train phase."""
        # Set the meta-train log
        trlog = {}
        trlog['args'] = vars(self.args)
        trlog['train_loss'] = []
        trlog['val_loss'] = []
        trlog['train_acc'] = []
        trlog['val_acc'] = []
        trlog['max_acc'] = 0.0
        trlog['max_acc_episode'] = 0
        self.prev_embed_heads = []

        # Set the timer
        timer = Timer()
        # Set global count to zero
        global_count = 0
        # Set tensorboardX
        #writer = SummaryWriter(comment=self.args.save_path)
        writer = SummaryWriter()
        

        # # Generate the labels for train set of the episodes
        # label_shot = torch.arange(self.args.way).repeat(self.args.shot)
        # label_shot = label_shot.type(torch.LongTensor)
        # label_shot = label_shot.to(self.device)
        self.adapt_acc= []
        adapt_f1 = []
        adapt_auc = []
        subject_ids = []

        # initial prototype
        prototypes = torch.zeros((self.args.way, self.args.z_pred), device=self.device) 

        self.proto = PrototypicalLoss(embedding_dim = self.args.z_pred,prototypes=prototypes)

        # subject specific proto
        #self.subject_proto = [PrototypicalLoss(embedding_dim = self.args.z_pred, n_classes = self.args.way,prototypes=prototypes) for i in range(40)]

        self.subject_proto = []


    
        for subject_id in range(self.args.start_range, self.args.subject_range):
            print("Preparing dataset loader for subject "+str(subject_id))

            #print(self.model.pred_head.state_dict())
            

            if self.args.dataset == 'AMIGOS':
                if subject_id in [26,31,4,5,10,11,22,25,28,30,40,8,24,32]:
                    continue
            
            early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, delta=0.01)
            #self.trainset = Dataset('train', self.args, subject_id)
            self.trainset = Dataset('train', self.args, self.x_train_s[subject_id], self.y_train_s[subject_id], self.x_val_s[subject_id], self.y_val_s[subject_id], self.x_test_s[subject_id], self.y_test_s[subject_id])
            # simple train loader
            self.train_loader = DataLoader(dataset=self.trainset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)


            # self.train_sampler = CategoriesSampler(self.trainset.label, self.args.num_batch, self.args.way, self.args.shot + self.args.train_query)
            # self.train_loader = DataLoader(dataset=self.trainset, batch_sampler=self.train_sampler, num_workers=0, pin_memory=True)



            # # Load meta-val set
            self.valset = Dataset('val', self.args, self.x_train_s[subject_id], self.y_train_s[subject_id], self.x_val_s[subject_id], self.y_val_s[subject_id], self.x_test_s[subject_id], self.y_test_s[subject_id])
            # simple val loader
            self.val_loader = DataLoader(dataset=self.valset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

            # self.val_sampler = CategoriesSampler(self.valset.label, 20, self.args.way, self.args.shot + self.args.val_query)
            # self.val_loader = DataLoader(dataset=self.valset, batch_sampler=self.val_sampler, num_workers=0, pin_memory=True)
            for episode in range(1, self.args.max_episode + 1):
                start_time = time.time()

                # Set the model to train mode
                self.model.train()
                # Set averager classes to record training losses and accuracies
                train_loss_averager = Averager()
                train_acc_averager = Averager()

                # # Generate the labels for test set of the episodes during meta-train updates
                # label = torch.arange(self.args.way).repeat(self.args.train_query)
                # label = label.type(torch.LongTensor)
                # label = label.to(self.device)
                
                # Using tqdm to read samples from train loader
                tqdm_gen = tqdm.tqdm(self.train_loader)
                for i, batch in enumerate(tqdm_gen, 1):
                    # Update global count number
                    global_count = global_count + 1
                    data = batch[0]
                    #print(data.size())
                    labeled = batch[1]
                    labeled = labeled.to(self.device)
                    labeled = labeled.long()
                    data = data.to(self.device)

                    self.buffer.add_data(examples=data, labels=labeled)
                    buf_data, buf_labels = self.buffer.get_data(self.buffer_batch_size, transform=None)


                    # if subject_id == 1:
                    #     self.buffer_class_balanced.add_data(examples=data, labels=labeled,encoder = self.model.eegnet) # modification
                    # else:
                    #     # encoder based add data
                    #     #self.buffer_class_balanced.add_data2(examples=data, labels=labeled, encoder=self.model.encoder) # modification

                    #     #self.buffer_class_balanced.add_data_avg_proto(examples=data, labels=labeled, encoder=self.model.eegnet, global_prototypes = self.proto.prototypes) # modification
                    #     self.buffer_class_balanced.add_data3(examples=data, labels=labeled, encoder=self.model.eegnet, global_prototypes = self.proto.prototypes) # modification

                    
                    #buf_data, buf_labels,_,_ = self.buffer_class_balanced.get_data(self.buffer_batch_size, transform=None) # modification


                    data_comb = torch.cat([data, buf_data], dim=0)
                    label_comb = torch.cat([labeled, buf_labels], dim=0)


                    data_aug = self.sim_aug(data_comb)
                    data_comb = torch.cat((data_comb, data_aug), 0)
                    label_comb = torch.cat((label_comb, label_comb), 0)

                    # zero optimi
                    self.optimizer.zero_grad()

                    embeddings = self.model.eegnet(data_comb)
                    loss,_ = self.proto(embeddings, label_comb, update_prototypes=True)

                    
                    #compute loss on global embeddings and memory data
                    if len(self.subject_proto) > 3:
                        embeddings_memory = self.model.eegnet(data_comb)
                        loss_mem = prototypical_loss(embeddings_memory, label_comb,aggregated_embeddings)
                        loss += loss_mem
                        #loss = loss/2

                    # # compute loss on each subject prototype
                    # if len(self.subject_proto) > 1:
                    #     loss_sub = 0
                    #     for _, proto in enumerate(self.subject_proto):
                    #         embeddings_memory = self.model(data_comb)
                    #         loss_sub = prototypical_loss(embeddings_memory, label_comb,proto)
                    #         loss_sub += loss_sub
                    #     loss += (loss_sub/(len(self.subject_proto)-1))
                    #     loss = loss/2

                    logits_comb = self.model(data_comb)
                    logits = self.model(data)

                    cross_loss = F.cross_entropy(logits_comb, label_comb)

                    loss += cross_loss


                    loss.backward()
                    self.optimizer.step()

                    acc = count_acc(logits, labeled)


                    # Add loss and accuracy for the averagers
                    train_loss_averager.add(loss.item())
                    train_acc_averager.add(acc)



                # Update the averagers
                train_loss_averager = train_loss_averager.item()
                train_acc_averager = train_acc_averager.item()
                # Update learning rate
                #self.lr_scheduler.step()
                # Start validation for this episode, set model to eval mode
                self.model.eval()

                # Set averager classes to record validation losses and accuracies
                val_loss_averager = Averager()
                val_acc_averager = Averager()

                # # Generate the labels for test set of the episodes during meta-val for this episode
                # label = torch.arange(self.args.way).repeat(self.args.val_query)
                # label = label.type(torch.LongTensor)
                # label = label.to(self.device)
                
                # Run meta-validation
                for i, batch in enumerate(self.val_loader, 1):
                    data = batch[0]
                    data = data.to(self.device)
                    labeled = batch[1]
        
                    labeled = labeled.to(self.device)
                    labeled = labeled.long()
                    

                    #embeddings = self.model(data)
                    logits = self.model(data)


                    #loss,acc = self.proto(embeddings,  labeled.long(), update_prototypes = False)
                    #loss,acc = self.subject_proto[subject_id](embeddings, labeled.long(), update_prototypes = False)
                    loss = F.cross_entropy(logits, labeled)
                    acc = count_acc(logits, labeled)

                    val_loss_averager.add(loss.item())
                    val_acc_averager.add(acc)

                # Update validation averagers
                val_loss_averager = val_loss_averager.item()
                val_acc_averager = val_acc_averager.item()
                # Write the tensorboardX records
                # Print loss and accuracy for this episode 
                print('Episode {}, Val, Loss={:.4f} Acc={:.4f}'.format(episode, val_loss_averager, val_acc_averager))
                #self.lr_scheduler.step()

                #early_stopping(val_loss_averager, self.model, self.model_path)

                # if early_stopping.early_stop:
                #     #print("Early stopping")
                #     break


            # accuracy on current subject
            val_acc_averager, f1, auc_roc = self.eval3(subject_id)
            self.adapt_acc.append(val_acc_averager)
            #subject_ids.append(subject_id)

            self.subject_proto.append(self.proto.prototypes)

            # self.subject_proto list into torch (subjjkects, classes, dim)
            

            

            if len(self.subject_proto) > 1:
                if len(self.subject_proto) < 3:
                    self.proto.prototypes = torch.mean(torch.stack(self.subject_proto), dim=0)
                else: # only take last 5
                    #self.proto.prototypes = torch.mean(torch.stack(self.subject_proto[-5:]), dim=0)
            
                    
                    aggregated_embeddings = aggregate_subj_embeddings(torch.stack(self.subject_proto))
                    self.proto.prototypes = aggregated_embeddings

            # # total average accuracy on seen subjects
            # if subject_id == 12 or subject_id == 20 or subject_id == 36:
            #     self.eval2(subject_id)

        results = [subject_ids, self.adapt_acc]
        file_name = 'Adaptation_results_'+ self.args.dataset +'_'
        #np.savetxt(file_name, results, delimiter=',')
        self.save_data(file_name,results)   


        forget_acc = self.eval2(subject_id)
        #self.save_embeddings(subject_id)
        writer.close()

        time.sleep(2)

        self.report_results(forget_acc,subject_id)


    def eval3(self, subject_id):
        #self.model.eval()

        temp_model = copy.deepcopy(self.model)
        temp_model.eval()

        all_labels = []
        all_preds = []
           
        # Load meta-val set
        self.valset = Dataset('test',self.args, self.x_train_s[subject_id], self.y_train_s[subject_id], self.x_val_s[subject_id], self.y_val_s[subject_id], self.x_test_s[subject_id], self.y_test_s[subject_id])
        self.val_loader = DataLoader(dataset=self.valset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

        # self.val_sampler = CategoriesSampler(self.valset.label, 20, self.args.way, self.args.shot + self.args.val_query)
        # self.val_loader = DataLoader(dataset=self.valset, batch_sampler=self.val_sampler, num_workers=0, pin_memory=True)
        
        
        # Set averager classes to record validation losses and accuracies
        val_loss_averager = Averager()
        val_acc_averager = Averager()

        # Run meta-validation
        for i, batch in enumerate(self.val_loader, 1):
            data = batch[0]
            data = data.to(self.device)
            labeled = batch[1]
            labeled = labeled.long()
            labeled = labeled.to(self.device)
            

            #embeddings = self.model(data)
            logits = self.model(data)

            #loss,acc = self.proto(embeddings,  labeled, update_prototypes = False)
            #loss,acc = self.subject_proto[subject_id](embeddings, labeled, update_prototypes = False)
            loss = F.cross_entropy(logits, labeled)
            acc = count_acc(logits, labeled)

            val_loss_averager.add(loss.item())
            val_acc_averager.add(acc)

            # Collect all labels and predictions for F1 and AUC-ROC
            all_labels.extend(labeled.cpu().numpy())
            #all_preds.extend(torch.softmax(logits, dim=1).detach().cpu().numpy())

        # Update validation averagers
        val_loss_averager = val_loss_averager.item()
        val_acc_averager = val_acc_averager.item()

        # # Compute F1 score and AUC-ROC
        # f1 = f1_score(all_labels, np.argmax(all_preds, axis=1), average='weighted')
        # auc_roc = roc_auc_score(all_labels, all_preds, multi_class='ovo', average='weighted')
        f1 = 0
        auc_roc = 0
        # Print loss, accuracy, F1, and AUC-ROC for this episode
        print('Val, Loss={:.4f} Acc={:.4f} F1={:.4f} AUC-ROC={:.4f}'.format(val_loss_averager, val_acc_averager, f1, auc_roc))

        return val_acc_averager, f1, auc_roc


    def eval2(self, subjects):


        all_labels = []
        all_preds = []

        subject_ids = []
        forget_acc = []
        forget_f1 = []
        forget_auc = []

        for subject_id in range(self.args.start_range, subjects + 1):
            if self.args.dataset == 'AMIGOS':
                if subject_id in [26,31,4,5,10,11,22,25,28,30,40,8,24,32]:
                    continue  

            temp_model = copy.deepcopy(self.model)
            temp_model.eval()     
            # Load meta-val set
            self.valset = Dataset('test',self.args, self.x_train_s[subject_id], self.y_train_s[subject_id], self.x_val_s[subject_id], self.y_val_s[subject_id], self.x_test_s[subject_id], self.y_test_s[subject_id])
            self.val_loader = DataLoader(dataset=self.valset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

            # self.val_sampler = CategoriesSampler(self.valset.label, 20, self.args.way, self.args.shot + self.args.val_query)
            # self.val_loader = DataLoader(dataset=self.valset, batch_sampler=self.val_sampler, num_workers=0, pin_memory=True)

            # Set averager classes to record validation losses and accuracies
            val_loss_averager = Averager()
            val_acc_averager = Averager()

            # Run meta-validation
            for i, batch in enumerate(self.val_loader, 1):
                data = batch[0]
                data = data.to(self.device)
                labeled = batch[1]
                labeled = labeled.long()
                labeled = labeled.to(self.device)
                    
                #embeddings = self.model(data)
                logits = self.model(data)


                #loss,acc = self.proto(embeddings,  labeled, update_prototypes = False)
                #loss,acc = self.subject_proto[subject_id](embeddings, labeled, update_prototypes = False)
                loss = F.cross_entropy(logits, labeled)
                acc = count_acc(logits, labeled)
                val_loss_averager.add(loss.item())
                val_acc_averager.add(acc)

            # Update validation averagers
            val_loss_averager = val_loss_averager.item()
            val_acc_averager = val_acc_averager.item()

            f1 = 0
            auc_roc = 0

            # Print loss, accuracy, F1, and AUC-ROC for this episode
            print('Val, Loss={:.4f} Acc={:.4f} F1={:.4f} AUC-ROC={:.4f}'.format(val_loss_averager, val_acc_averager, f1, auc_roc))
            forget_acc.append(val_acc_averager)
            subject_ids.append(subject_id)
        
        #save the results in csv subject ID, acc, f1, auc
        results = [subject_ids,forget_acc]
        #np.savetxt('Adaptive_results.csv', results, delimiter=',')
        #star_name = 'forget_after_subject'+str(subjects)
        file_name = 'Average_Acc_results_'+ self.args.dataset +'_'
        self.save_data(file_name,results)
        return forget_acc

     

    def save_data(self, name,d):
        export_data = zip_longest(*d, fillvalue = '')
        with open(name + self.args.training_style  + self.args.modality +'.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile)
            wr.writerow(("Subject_ID", "acc"))
            wr.writerows(export_data)
        myfile.close()

    def save_embeddings(self,subject_id):      
        all_embeddings = []
        all_labels = []
        
        temp_model = copy.deepcopy(self.model)

        temp_model.eval()

        self.valset = Dataset('val', self.args, subject_id)
        self.val_sampler = CategoriesSampler(self.valset.label, 20, self.args.way, self.args.shot + self.args.val_query)
        self.val_loader = DataLoader(dataset=self.valset, batch_sampler=self.val_sampler, num_workers=0, pin_memory=True)


        tqdm_gen = tqdm.tqdm(self.val_loader)
        #with torch.no_grad():
        for i, batch in enumerate(tqdm_gen, 1):
            data = batch[0]
            labeled = batch[1]
            data = data.to(self.device)
            labeled = labeled.type(torch.LongTensor)
            labeled = labeled.to(self.device)


            #embeddings = temp_model.encoder(data)
            _,embeddings = temp_model((data, labeled, data))
            all_embeddings.append(embeddings)
            all_labels.append(labeled)
            
        

        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        #all_embeddings_np = all_embeddings.cpu().numpy()
        #all_labels_np = all_labels.cpu().numpy()
        torch.save(all_embeddings, 'all_embeddings.pt')
        torch.save(all_labels, 'all_labels.pt')


    def report_results(self,forget_acc,subject_id):
        #Adaptation_acc_path = r'.\Proto_Results_adaptationEEG.csv'
        #Proto_data = pd.read_csv(Adaptation_acc_path)
        # convert self.adapt_acc to numpy array
        adapt_acc = np.array(self.adapt_acc)
        print("\n Average Adaptation Accuracy \n")
        print(adapt_acc.mean() * 100)
        # print(Proto_data['acc'].mean() * 100)



        # forget_12_path = r'.\Proto_Results_forget_after_subject12EEG.csv'
        # Proto_data = pd.read_csv(forget_12_path)
        # print("\n Average Forget After Subject 12 Accuracy \n")
        # print(Proto_data['acc'].mean() * 100)

        # forget_20_path = r'.\Proto_Results_forget_after_subject20EEG.csv'
        # Proto_data = pd.read_csv(forget_20_path)
        # print("\n Average Forget After Subject 20 Accuracy \n")
        # print(Proto_data['acc'].mean() * 100)


        # forget_36_path = r'.\Proto_Results_forget_after_subject36EEG.csv'
        # Proto_data = pd.read_csv(forget_36_path)
        # print("\n Average Forget After Subject 36 Accuracy \n")
        # print(Proto_data['acc'].mean() * 100)

        # forget_40_path = r'.\Proto_Results_forget_after_subject40EEG.csv'
        # Proto_data = pd.read_csv(forget_40_path)
        # print("\n Average Forget After Subject 40 Accuracy \n")
        # print(Proto_data['acc'].mean() * 100)

        print("\n Average Accuracy on all seen subjects After last Subject:  \n",subject_id)
        forget_acc = np.array(forget_acc)
        print(forget_acc.mean() * 100)




def extract_tensor_value(tensor_str):
    """
    Extracts the numerical value from a tensor string.
    :param tensor_str: The string representation of the tensor
    :return: The numerical value as a float
    """
    # Use regex to extract the numerical part from the tensor string
    #match = re.search(r'tensor\(([^)]+)\)', tensor_str)
    # extract only number tensor(0.4750, device='cuda:0') from this string
    match = re.search(r'tensor\(([^,]+)', tensor_str)
    
    
    if match:
        value_str = match.group(1)
        return float(value_str)
    else:
        raise ValueError(f"String format is incorrect: {tensor_str}")
    
    # now over all forgetting
def get_forgetting_accuracy(Adaptaion_acc_list, forget_acc_list):
    
    forget_acc = 0.0

    for i in range(0, len(forget_acc_list - 1)):
        forget_acc += Adaptaion_acc_list[i] - forget_acc_list[i]



    return (forget_acc/len(forget_acc_list - 1)) * 100


#########################

# def compute_pairwise_distances(protos):
#     num_subjects, num_classes, _ = protos.size()
#     distances = torch.zeros(num_subjects, num_classes, num_classes)
#     for s in range(num_subjects):
#         for i in range(num_classes):
#             for j in range(num_classes):
#                 if i != j:
#                     dist = torch.norm(protos[s, i] - protos[s, j], p=2)
#                     distances[s, i, j] = dist
#     return distances


# def compute_weights(distances, sigma=1.0):
#     num_subjects, num_classes, _ = distances.size()
#     intra_class_similarity = torch.zeros(num_subjects, num_classes)
#     inter_class_similarity = torch.zeros(num_subjects, num_classes)

#     for s in range(num_subjects):
#         for i in range(num_classes):
#             intra_distances = distances[s, i, i]
#             inter_distances = distances[s, i].sum(dim=0) - intra_distances
#             intra_class_similarity[s, i] = 1 / (1 + intra_distances.mean())
#             inter_class_similarity[s, i] = 1 / (1 + inter_distances.mean())

#     # Normalize to obtain weights
#     weights = intra_class_similarity / (inter_class_similarity + 1e-8)
#     weights = weights / weights.sum(dim=1, keepdim=True)
#     return weights


# def weighted_aggregate_prototypes(prototypes, weights):
#     num_subjects, num_classes, dim = prototypes.size()
#     final_prototypes = torch.zeros((num_classes, dim),device=prototypes.device)
    
#     for i in range(num_classes):
#         weighted_sum = torch.zeros((dim),device=prototypes.device)
#         weight_sum = 0
#         for s in range(num_subjects):
#             weight = weights[s, i].item()  # Weight for class `i` from subject `s`
#             weighted_sum += weight * prototypes[s, i]
#             weight_sum += weight
        
#         if weight_sum > 0:
#             final_prototypes[i] = weighted_sum / weight_sum
#         else:
#             final_prototypes[i] = prototypes[:, i].mean(dim=0)  # Fallback to mean if no weights

#     return final_prototypes

def compute_classwise_distances(prototypes):
    num_subjects, num_classes, dim = prototypes.size()
    
    # Initialize a distance matrix with shape (num_subjects, num_classes)
    distance_matrix = torch.zeros(num_subjects, num_classes)
    
    # Compute distances for each class separately
    for c in range(num_classes):
        class_prototypes = prototypes[:, c, :]  # Shape: (num_subjects, dim)
        # Compute pairwise distances between prototypes of the same class
        distances = torch.cdist(class_prototypes, class_prototypes, p=2)
        # Sum of pairwise distances for each subject
        distance_matrix[:, c] = distances.sum(dim=1)  # Sum distances across subjects

    return distance_matrix

def prototypical_loss(embeddings, labels,prototypes):
    '''
    Compute the prototypical loss
    '''
    
    #  normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)

    prototypes = F.normalize(prototypes, p=2, dim=1)

    distances = torch.cdist(embeddings, prototypes,p=2)  # Euclidean distance

    # distance with current prototypes  
    #distances = torch.cdist(embeddings, self.current_prototypes,p=2)  # Euclidean distance

    # dot product between them
    #distances = torch.mm(embeddings, self.prototypes.t())

    #print(distances.size())
    log_p_y = -distances
    log_p_y = F.log_softmax(log_p_y, dim=1)
    loss = F.nll_loss(log_p_y, labels)


    return loss/ len(labels)
#####################################################


# def compute_distances(embeddings):
#     # Compute pairwise Euclidean distances between embeddings
#     diff = embeddings.unsqueeze(1) - embeddings.unsqueeze(0)
#     distances = torch.sqrt(torch.sum(diff ** 2, dim=2))
#     return distances



# def compute_weights(distances, epsilon=1e-10):
#     # Inverse distance weighting
#     weights = 1 / (distances + epsilon)
#     # Set diagonal to zero to ignore self-distances
#     weights.fill_diagonal_(0)
#     return torch.sum(weights, dim=1)


# def aggregate_embeddings(mean_embeddings, weights):
#     # Aggregate embeddings using weights
#     weight_sums = torch.sum(weights)
#     weights = weights.unsqueeze(1).repeat(1,128)

#     weighted_sum = mean_embeddings * weights
    
#     weighted_sum = torch.sum(weighted_sum, dim=0)

#     # Avoid division by zero
#     consensus_embeddings = weighted_sum / (weight_sums)


#     return consensus_embeddings





# def aggregate_subj_embeddings(mean_embeddings):
#     # # Example usage:
#     num_subjects = len(mean_embeddings)
#     num_classes = 4#len(mean_embeddings[1])
#     embedding_dim = 128#len(mean_embeddings[2])

#     # # Generate random embeddings for illustration
#     # mean_embeddings = torch.rand(num_subjects, num_classes, embedding_dim)

#     # Calculate pairwise distances and weights
#     distances = torch.zeros((num_classes, num_subjects, num_subjects),device=mean_embeddings.device)
#     weights = torch.zeros((num_classes, num_subjects),device=mean_embeddings.device)

#     for class_idx in range(num_classes):
#         class_embeddings = mean_embeddings[:, class_idx, :]
#         class_distances = compute_distances(class_embeddings)
#         class_weights = compute_weights(class_distances)
        
#         distances[class_idx] = class_distances
#         weights[class_idx] = class_weights

#     # Aggregate embeddings for each class
#     consensus_embeddings = torch.zeros((num_classes, embedding_dim),device=mean_embeddings.device)

#     for class_idx in range(num_classes):
#         class_weights = weights[class_idx]
#         class_mean_embeddings = mean_embeddings[:, class_idx, :]
#         consensus_embeddings[class_idx] = aggregate_embeddings(class_mean_embeddings, class_weights)
#     return consensus_embeddings