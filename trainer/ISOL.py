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
from models.EEG_model import EEGNet1D,ClassifierEEGNet1D
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score #f1_score
from sklearn.preprocessing import LabelBinarizer
from utils.misc import Averager, Timer, count_acc, compute_confidence_interval, ensure_path
from tensorboardX import SummaryWriter
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


class ISOLTrainer(object):
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

        #self.OPELoss = OPELoss(args.way, temperature=0.5)  

        #self.model = MtlLearner(self.args)

        #self.model = EEGNet1D(num_classes=self.args.way,embedding_dim=self.args.z_pred)

        self.model =ClassifierEEGNet1D(args = self.args,num_classes=self.args.way,embedding_dim=self.args.z_pred)




        self.model.to(self.device)

        # copy state dict for the init model
        self.init_state_dict = copy.deepcopy(self.model.state_dict())

        # Set the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.base_lr)

        # lr scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_size, gamma=self.args.gamma)

        self.buffer_size = 200
        self.buffer_batch_size = 32
        self.buffer = Buffer(self.buffer_size, device=self.device)




        with torch.no_grad():
            self.sim_aug = v2.Compose([v2.RandomAffine(10, (0.01, 0.01))])
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

        #self.proto = PrototypicalLoss(embedding_dim = self.args.z_pred)

        for subject_id in range(self.args.start_range, self.args.subject_range):
            print("Preparing dataset loader for subject "+str(subject_id))

            #print(self.model.pred_head.state_dict())
            

            if self.args.dataset == 'AMIGOS':
                if subject_id in [26,31,4,5,10,11,22,25,28,30,40,8,24,32]:
                    continue
            #self.trainset = Dataset('train', self.args, subject_id)
            self.trainset = Dataset('train', self.args, self.x_train_s[subject_id], self.y_train_s[subject_id], self.x_val_s[subject_id], self.y_val_s[subject_id], self.x_test_s[subject_id], self.y_test_s[subject_id])
            # simple train loader
            #self.train_loader = DataLoader(dataset=self.trainset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)


            self.train_sampler = CategoriesSampler(self.trainset.label, self.args.num_batch, self.args.way, self.args.shot + self.args.train_query)
            self.train_loader = DataLoader(dataset=self.trainset, batch_sampler=self.train_sampler, num_workers=0, pin_memory=True)



            # # Load meta-val set
            self.valset = Dataset('val', self.args, self.x_train_s[subject_id], self.y_train_s[subject_id], self.x_val_s[subject_id], self.y_val_s[subject_id], self.x_test_s[subject_id], self.y_test_s[subject_id])
            # simple val loader
            #self.val_loader = DataLoader(dataset=self.valset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

            self.val_sampler = CategoriesSampler(self.valset.label, 20, self.args.way, self.args.shot + self.args.val_query)
            self.val_loader = DataLoader(dataset=self.valset, batch_sampler=self.val_sampler, num_workers=0, pin_memory=True)

            # copy init model state dict
            self.model.load_state_dict(self.init_state_dict)
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
                    labeled = batch[1]
                    labeled = labeled.to(self.device)
                    labeled = labeled.long()
                    data = data.to(self.device)

                    self.buffer.add_data(examples=data, labels=labeled)

                    buf_data, buf_labels = self.buffer.get_data(self.buffer_batch_size, transform=None)


                    data_comb = torch.cat([data, buf_data], dim=0)
                    label_comb = torch.cat([labeled, buf_labels], dim=0)

                    data_aug = self.sim_aug(data_comb)
                    data_comb = torch.cat((data_comb, data_aug), 0)
                    label_comb = torch.cat((label_comb, label_comb), 0)

                    # zero optimi
                    self.optimizer.zero_grad()

                    logits= self.model(data)

                    comb_logits= self.model(data_comb)

                    loss = F.cross_entropy(comb_logits,label_comb) #+ loss_sim + loss_OPE + buf_loss_sim + buf_loss_OPE

                    loss.backward()
                    self.optimizer.step()

                    acc = count_acc(logits, labeled)
                    # Add loss and accuracy for the averagers
                    train_loss_averager.add(loss.item())
                    train_acc_averager.add(acc)


                # Update the averagers
                train_loss_averager = train_loss_averager.item()
                train_acc_averager = train_acc_averager.item()

                self.model.eval()

                # Set averager classes to record validation losses and accuracies
                val_loss_averager = Averager()
                val_acc_averager = Averager()

                # Run meta-validation
                for i, batch in enumerate(self.val_loader, 1):
                    data = batch[0]
                    data = data.to(self.device)
                    labeled = batch[1]
        
                    labeled = labeled.to(self.device)
                    labeled = labeled.long()
                    
                    logits= self.model(data)
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
                self.lr_scheduler.step()

            val_acc_averager, f1, auc_roc = self.eval3(subject_id)
            self.adapt_acc.append(val_acc_averager)
            subject_ids.append(subject_id)

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
        #self.val_loader = DataLoader(dataset=self.valset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

        self.val_sampler = CategoriesSampler(self.valset.label, 20, self.args.way, self.args.shot + self.args.val_query)
        self.val_loader = DataLoader(dataset=self.valset, batch_sampler=self.val_sampler, num_workers=0, pin_memory=True)
        
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
            
            # embeddings = self.model(data)
            # loss,acc = self.proto(embeddings,  labeled, update_prototypes = False)

            logits= self.model(data)
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
            #self.val_loader = DataLoader(dataset=self.valset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

            self.val_sampler = CategoriesSampler(self.valset.label, 20, self.args.way, self.args.shot + self.args.val_query)
            self.val_loader = DataLoader(dataset=self.valset, batch_sampler=self.val_sampler, num_workers=0, pin_memory=True)

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
                    
                # embeddings = self.model(data)
                # loss,acc = self.proto(embeddings,  labeled, update_prototypes = False)

                logits= self.model(data)
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
        with open(name+ self.args.training_style + self.args.modality +'.csv', 'w', newline='') as myfile:
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
