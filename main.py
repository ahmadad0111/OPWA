""" Main function for this repo. """
import argparse
import torch
from utils.misc import pprint
from utils.gpu_tools import set_gpu
from trainer.new import NewTrainer
from trainer.new_cross_ent import NewCrossTrainer

from trainer.cope import CopeTrainer
from trainer.OnPro import OnProTrainer
from trainer.ambm import MetaTrainer

from trainer.Baseline import BaseTrainer
from trainer.ISOL import ISOLTrainer
from trainer.Static import StaticTrainer


#from trainer.meta_subject_agnostic import MetaTrainerAgnostic
import random
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Basic parameters
    parser.add_argument('--model_type', type=str, default='EEGNet',
                        choices=['EEGNet'])  # The network architecture
    parser.add_argument('--dataset', type=str, default='DEAP') # Dataset [AMIGOS, SEED, DEAP,BCI_IV_2a, PPB_EMO]
    parser.add_argument('--phase', type=str, default='meta_train',
                        choices=['meta_train', 'meta_eval'])  # Phase
    
    # rull_all
    parser.add_argument('--run_all', type=bool, default=True)  # Run all styles and datasets
    
    parser.add_argument('--training_style', type=str, default='Baseline',
                        choices=['OPWA', 'COPE','OnPro','Baseline','NEW_Cross', 'ISOL','AMBM', 'Static'])  # Training style
    # Manual seed for PyTorch, "0" means using random seed
    parser.add_argument('--seed', type=int, default=48)
    parser.add_argument('--gpu', default='0')  # GPU id
    parser.add_argument('--dataset_dir', type=str,
                        default='./data/')  # Dataset folder
    #add modalitiy type
    parser.add_argument('--modality', type=str, default='EEG')

    # Parameters for meta-train phase
    # Episode number for meta-train phase
    parser.add_argument('--max_episode', type=int, default=5)
    # The number for different tasks used for meta-train
    parser.add_argument('--num_batch', type=int, default=12)
    # Shot number, how many samples for one class in a task
    parser.add_argument('--shot', type=int, default=10)
    # Way number, how many classes in a task
    parser.add_argument('--way', type=int, default=4)
    # The number of training samples for each class in a task
    parser.add_argument('--train_query', type=int, default=10)
    # The number of test samples for each class in a task
    parser.add_argument('--val_query', type=int, default=10)
    # Learning rate for SS weights
    parser.add_argument('--meta_lr1', type=float, default=0.0001)
    # Learning rate for FC weights
    parser.add_argument('--meta_lr2', type=float, default=0.005)
    # Learning rate for the inner loop
    parser.add_argument('--base_lr', type=float, default=0.0001) #0.001
    # hyper learning rate
    parser.add_argument('--hyper-lr', type=float, default=1e-4)
    # value for gradient clip
    parser.add_argument('--clip_hyper', type=float, default=1.0)
    # The number of updates for the inner loop
    parser.add_argument('--update_step', type=int, default=10)
    # The number of episodes to reduce the meta learning rates
    parser.add_argument('--step_size', type=int, default=30)

    # Gamma for the meta-train learning rate decay
    parser.add_argument('--gamma', type=float, default=0.9)

    # Additional label for meta-train
    parser.add_argument('--meta_label', type=str, default='exp1')

    # args.patience
    parser.add_argument('--patience', type=int, default=10)

    # start range for subject
    parser.add_argument('--start_range', type=int, default=1)
    # subject range
    parser.add_argument('--subject_range', type=int, default=33)

    # pred embed size
    parser.add_argument('--z_pred', type=int, default=128)
    # embedding size 
    parser.add_argument('--embed_size', type=int, default=200)
    # Set the parameters
    args = parser.parse_args()

    # Set the GPU id
    set_gpu(args.gpu)

    #DEVICE IF CUDA IS AVAILABLE make arg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--device', type=str, default=device)

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


    if args.dataset == 'DEAP':
        args.start_range = 1
        args.subject_range = 33
    elif args.dataset == 'AMIGOS':
        args.start_range = 1
        args.subject_range = 41
    elif args.dataset == 'BCI_IV_2a':
        args.start_range = 1
        args.subject_range = 10
    elif args.dataset == 'PPB_EMO':
        args.start_range = 2
        args.subject_range = 42


    if args.run_all:
        # if run all styles and datasets
        for training_style in ['COPE', 'OnPro','Baseline', 'OPWA','AMBM', 'ISOL']:#
            print('Training style:', training_style)

            if args.modality == 'EEG':
                for dataset in ['BCI_IV_2a']: #['BCI_IV_2a', 'AMIGOS', 'DEAP','PPB_EMO']:
                    print('Dataset:', dataset)
                    args.dataset = dataset
                    args.training_style = training_style

                    if dataset == 'BCI_IV_2a':
                        #args.max_episode = 30
                        args.start_range = 1
                        args.subject_range = 10
                    elif dataset == 'AMIGOS':
                        args.start_range = 1
                        args.subject_range = 41
                        #args.max_episode = 30
                    elif dataset == 'DEAP':
                        #args.max_episode = 30
                        args.start_range = 1
                        args.subject_range = 33
                    elif dataset == 'PPB_EMO':
                        #args.max_episode = 30
                        args.start_range = 2
                        args.subject_range = 42

                        
                    if args.training_style == 'NEW':
                        trainer = NewTrainer(args)
                        trainer.train()
                    elif args.training_style == 'COPE':
                        trainer = CopeTrainer(args)
                        trainer.train()
                    elif args.training_style == 'OnPro':
                        trainer = OnProTrainer(args)
                        trainer.train()
                    elif args.training_style == 'Baseline':
                        trainer = BaseTrainer(args)
                        trainer.train()
                    elif args.training_style == 'ISOL':
                        trainer = ISOLTrainer(args)
                        trainer.train()
                    elif args.training_style == 'OPWA':
                        trainer = NewCrossTrainer(args)
                        trainer.train()
                    elif args.training_style == 'AMBM':
                        trainer = MetaTrainer(args)
                        trainer.train()
                    elif args.training_style == 'Static':
                        args.max_episode = 30
                        trainer = StaticTrainer(args)
                        trainer.train()
                    else:
                        raise ValueError('Please set correct training style.')
            elif args.modality == 'ECG':
                for dataset in ['AMIGOS']:
                    print('Dataset:', dataset)
                    args.dataset = dataset
                    args.training_style = training_style

                    if dataset == 'AMIGOS':
                        args.start_range = 1
                        args.subject_range = 41
                        args.max_episode = 10    

                    if args.training_style == 'NEW':
                        trainer = NewTrainer(args)
                        trainer.train()
                    elif args.training_style == 'COPE':
                        trainer = CopeTrainer(args)
                        trainer.train()
                    elif args.training_style == 'OnPro':
                        trainer = OnProTrainer(args)
                        trainer.train()
                    elif args.training_style == 'Baseline':
                        trainer = BaseTrainer(args)
                        trainer.train()
                    elif args.training_style == 'ISOL':
                        trainer = ISOLTrainer(args)
                        trainer.train()
                    elif args.training_style == 'NEW_Cross':
                        trainer = NewCrossTrainer(args)
                        trainer.train()
                    elif args.training_style == 'AMBM':
                        trainer = MetaTrainer(args)
                        trainer.train()
                    elif args.training_style == 'Static':
                        args.max_episode = 30
                        trainer = StaticTrainer(args)
                        trainer.train()
                    else:
                        raise ValueError('Please set correct training style.')                    
    else:
        # Print the training information only data, style, 
        # start training
        if args.training_style == 'COPE':
            trainer = CopeTrainer(args)
            trainer.train()
        elif args.training_style == 'OnPro':
            trainer = OnProTrainer(args)
            trainer.train()
        elif args.training_style == 'Baseline':
            trainer = BaseTrainer(args)
            trainer.train()
        elif args.training_style == 'OPWA':
            trainer = NewCrossTrainer(args)
            trainer.train()
        elif args.training_style == 'ISOL':
            trainer = ISOLTrainer(args)
            trainer.train()
        elif args.training_style == 'AMBM':
            trainer = MetaTrainer(args)
            trainer.train()
        elif args.training_style == 'Static':
            args.max_episode = 30
            trainer = StaticTrainer(args)
            trainer.train()
        else:
            raise ValueError('Please set correct training style.')
