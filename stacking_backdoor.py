import argparse
from copy import deepcopy
import csv
import gc
from pathlib import Path
from typing import List
import torch, torchvision
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from torchvision import transforms

from dataset_handler.data_utils import get_datasets, poison_dataset, get_general_transform, get_post_poison_transform, get_pre_poison_transform, create_stacked_dataset
from training_utils import train_stacked_models, train_one_epoch, stack_outputs
from models import get_num_classes, get_input_channels, get_model

import torch.optim as optim

import warnings
warnings.filterwarnings("ignore")

torch.hub.set_dir('/tudelft.net/staff-umbrella/dlsca/Lichao/Backdoor_ensemble/Pretrained_models')

def stacking_ensemble(args, dataname: str, batch_size: int, n_epochs: int, models_name_list: List[str], 
                   is_pretrained_list: List[bool], optim_list: List[str], ensemble_size: int, device: torch.device, mt_mdl_name = 'advanced_4000', k_fold=None, experim_num=0, saving_path: Path=Path()) -> None:
    
    """
    This function creates an ensemble of models and trains them using stacking. It then creates a meta-dataset with the output of the base models and trains a meta-model on it.
    Finally, it tests the whole ensemble on the test dataset.

    Args:
        args: Command line arguments.
        dataname (str): Name of the dataset.
        batch_size (int): Size of the batch for training, validation and testing.
        n_epochs (int): Number of epochs for training.
        models_name_list (List[str]): List of names of the models to be used in the ensemble.
        is_pretrained_list (List[bool]): List indicating whether the corresponding model is pretrained or not.
        optim_list (List[str]): List of optimizers to be used for the corresponding models.
        ensemble_size (int): Number of models in the ensemble.
        device (torch.device): Device to be used for training (CPU or GPU).
        mt_mdl_name (str, optional): Name of the meta model. Defaults to 'advanced_4000'.
        k_fold (int, optional): Number of folds for cross-validation. If None, the dataset is split into training and validation sets.
        experim_num (int, optional): Experiment number. Defaults to 0.
        saving_path (Path, optional): Path to save the results. Defaults to current directory.
    """
    
    # Get the number of classes and input channels from the dataset
    num_classes = get_num_classes(dataname)
    n_in_channels = get_input_channels(dataname)
    
    tr_vl_split = 0.8

    model_list = [get_model(model_name, num_classes, n_in_channels, pretrained).to('cpu') for model_name, pretrained in zip(models_name_list, is_pretrained_list)]

    # Define the optimizers
    optim_dict = {'sgd': optim.SGD, 'adam': optim.Adam}
    optimizers = [optim_dict[optim_name](model.parameters(), lr=args.lr) for model, optim_name in zip(model_list, optim_list)]

    # Define the loss function
    loss_function = CrossEntropyLoss() 

    # Initialize the meta-dataset
    meta_dataset = None

    # If k_fold is None, split the train_dataset into training and validation sets based on tr_vl_split ratio. 
    # then train the base models on the training set and create the meta-dataset using the validation set.
    if k_fold is None:
        train_dataset, validation_dataset, test_dataset, classes_names = get_datasets(dataname=dataname, root_path='/tudelft.net/staff-umbrella/dlsca/Lichao/Backdoor_ensemble/Data/', tr_vl_split=tr_vl_split, transform=get_pre_poison_transform(dataname))
        poisoned_trainset = poison_dataset(args, dataname=dataname, dataset=train_dataset, is_train=True, attack_name='badnet', post_transform=get_post_poison_transform(dataname))
        poisoned_testset = poison_dataset(args, dataname=dataname, dataset=test_dataset, is_train=False, attack_name='badnet', post_transform=get_post_poison_transform(dataname))
        poisoned_validationset = poison_dataset(args, dataname=dataname, dataset=validation_dataset, is_train=True, attack_name='badnet', post_transform=get_post_poison_transform(dataname))
        train_dataset.dataset.set_transform(get_general_transform(dataname))
        test_dataset.set_transform(get_general_transform(dataname))
        validation_dataset.dataset.set_transform(get_general_transform(dataname))

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        # poisoned_train_dataloader = DataLoader(poisoned_trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        if args.train_model:
            train_stacked_models(train_dataloader, model_list, n_epochs, optimizers, loss_function, device)
        else:
            for i, model in enumerate(model_list):
                model.load_state_dict(torch.load('/tudelft.net/staff-umbrella/dlsca/Lichao/Backdoor_ensemble/Models/Voting_model_{}.pt'.format(i)))

        
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        # poisoned_validation_dataloader = DataLoader(poisoned_validationset, batch_size=batch_size, shuffle=True, num_workers=2)
        print('generate meta dataset')
        meta_dataset = create_stacked_dataset(model_list, validation_dataloader, device='cuda')

    # If k_fold is not None, perform k-fold cross-validation and create the meta-dataset using the k-fold cross validation sets.
    # TODO: this section is not modified for badnet. I should modify the datasets correctly.
    else:        
        train_dataset, test_dataset, classes_names = get_datasets(dataname=dataname, root_path=None, tr_vl_split=None, transform=get_pre_poison_transform(dataname))
        poisoned_trainset = poison_dataset(args, dataname=dataname, dataset=train_dataset, is_train=True, attack_name='badnet', post_transform=get_post_poison_transform(dataname))
        poisoned_testset = poison_dataset(args, dataname=dataname, dataset=test_dataset, is_train=False, attack_name='badnet', post_transform=get_post_poison_transform(dataname))
        train_dataset.set_transform(get_general_transform(dataname))
        test_dataset.set_transform(get_general_transform(dataname))
        meta_dataset_list = []
        kf = KFold(n_splits=k_fold, shuffle=False)
        for train_idx, val_idx in kf.split(poisoned_trainset.data):
            train_subsampler = SubsetRandomSampler(train_idx)
            val_subsampler = SubsetRandomSampler(val_idx)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
            # poisoned_train_dataloader = DataLoader(poisoned_trainset, batch_size=batch_size, sampler=train_subsampler, num_workers=2)
            if args.train_model:
                train_stacked_models(train_dataloader, model_list, n_epochs, optimizers, loss_function, device)
            else:
                for i, model in enumerate(model_list):
                    model.load_state_dict(torch.load('/tudelft.net/staff-umbrella/dlsca/Lichao/Backdoor_ensemble/Models/Voting_model_{}.pt'.format(i)))
            validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
            # poisoned_validation_dataloader = DataLoader(poisoned_trainset, batch_size=batch_size, sampler=val_subsampler, num_workers=2)
            meta_dataset_list.append(create_stacked_dataset(model_list, validation_dataloader, 'cuda'))
        meta_dataset = torch.utils.data.ConcatDataset(meta_dataset_list)
    
    gc.collect()
    torch.cuda.empty_cache()  # Free up memory

    # Test the performance of models in model_list on validation data, this is not necessary but it is good to know how well the models are trained.
    # If you want uncomment the following lines.
    #************************************************************************************************************
    # with torch.no_grad():
    #     for model in model_list:
    #         model = model.to(device)
    #         model.eval()
    #         total_samples = 0
    #         correct_predictions = 0
    #         for batch_idx, (data, labels) in enumerate(poisoned_validation_dataloader):
    #             data, labels = data.to(device), labels.to(device)
    #             output = model(data)
    #             _, predicted = torch.max(output.data, 1)
    #             correct_predictions += (predicted == labels).sum().item()
    #             total_samples += labels.size(0)
    #         accuracy = (correct_predictions / total_samples) * 100
    #         print(f'Validation Accuracy of the model {model.__class__.__name__} on the validation images: {accuracy}%')
    #         model = model.to('cpu')
    
    #************************************************************************************************************


    meta_dataloader = DataLoader(meta_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    meta_model = get_model(mt_mdl_name, num_classes, num_classes*len(model_list), False)

    meta_optimizer = optim.Adam(meta_model.parameters(), lr=0.001)

    # Train the meta model
    print('Train the meta model')
    if args.train_meta_model:
        for epoch in range(n_epochs):
            train_one_epoch(meta_model, meta_dataloader, meta_optimizer, loss_function, device)
            torch.save(model.state_dict(), '/tudelft.net/staff-umbrella/dlsca/Lichao/Backdoor_ensemble/Models/stacking_meta_model.pt')
    else:
        meta_model.load_state_dict(torch.load('/tudelft.net/staff-umbrella/dlsca/Lichao/Backdoor_ensemble/Models/stacking_meta_model.pt'))

    
    # Training is finished. Set the device to cpu and free up memory
    device = torch.device('cpu')
    gc.collect()
    torch.cuda.empty_cache()

    
    # Test the whole ensemble on the test dataset
    print('Test the whole ensemble on the test dataset')
    if args.emsemble_prediction:
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)   
        meta_model = meta_model.to(device)
        meta_model.eval()
        total_samples = 0
        correct_predictions = 0
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(test_dataloader):
                data, labels = data.to(device), labels.to(device)
                meta_input = stack_outputs(model_list, data, device)
                output = meta_model(meta_input)
                _, predicted = torch.max(output.data, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                data, labels = data.to('cpu'), labels.to('cpu')
        test_acc = (correct_predictions / total_samples) * 100
        print(f'Ensemble of size: {ensemble_size}. Test Accuracy of the model on the test images: {test_acc}%')

        # Test the whole ensemble on the poisoned test dataset
        test_dataloader = DataLoader(poisoned_testset, batch_size=batch_size, shuffle=True, num_workers=2)  
        meta_model = meta_model.to(device) 
        meta_model.eval()
        total_samples = 0
        correct_predictions = 0
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(test_dataloader):
                data, labels = data.to(device), labels.to(device)
                meta_input = stack_outputs(model_list, data, device)
                output = meta_model(meta_input)
                _, predicted = torch.max(output.data, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                data, labels = data.to('cpu'), labels.to('cpu')
        bd_test_acc = (correct_predictions / total_samples) * 100
        print(f'Ensemble of size: {ensemble_size}. Test Accuracy of the model on the poisoned test images: {bd_test_acc}%')

        # Save the results and plots and other bullshits
        plots_path = saving_path.joinpath('plots')
        if not plots_path.exists():
            plots_path.mkdir()
        csv_path = saving_path.joinpath('results.csv')
        if not csv_path.exists():
            csv_path.touch()
            with open(file=csv_path, mode='w') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(['EXPERIMENT_NUMBER', 'ENSEMBLE_TECHNIQUE', 'NUMBER_OF_MODELS', 'DATASET', 'CLEAN_ACCURACY', 'ASR'])
        
        with open(file=csv_path, mode='a') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow([experim_num, f'stacking', len(model_list), dataname, test_acc, bd_test_acc])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default='cifar10')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--ensemble_size', type=int, default=25)
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--saving_path', type=str, default='.')
    parser.add_argument('--experim_num', type=int, default=0)
    parser.add_argument('--k_fold', type=bool, default=None)
    parser.add_argument('--mt_mdl_name', type=str, default='advanced_4000')

    parser.add_argument('--train_model', action=argparse.BooleanOptionalAction)
    parser.add_argument('--train_meta_model', action=argparse.BooleanOptionalAction)    
    parser.add_argument('--emsemble_prediction', action=argparse.BooleanOptionalAction)
    parser.add_argument('--attack_index', type=int, required=True)

    args = parser.parse_args()

    base_model_list = ['resnet50', 'vgg19', 'efficientnet-b3']
    if args.ensemble_size < 3:
        models_name_list = base_model_list[:args.ensemble_size]
    else:
        models_name_list = base_model_list * (args.ensemble_size // len(base_model_list))

    # We replace the first model with backdoored pretrained model
    models_name_list[0] = 'resnet50_bd'
    print(models_name_list)

    if args.ensemble_size % len(base_model_list) != 0:
        models_name_list += base_model_list[:args.ensemble_size % len(base_model_list)]
    optim_list = []    
    if args.optim.lower() == 'adam':
        optim_list = ['adam'] * args.ensemble_size
    elif args.optim.lower() == 'sgd':
        optim_list = ['sgd'] * args.ensemble_size
    else:
        raise Exception("Invalid optimizer name. Please choose either 'adam' or 'sgd'.")
    
    if args.device:
        if args.device not in ['cpu', 'cuda']:
            raise Exception("Invalid device name. Please choose either 'cpu' or 'cuda'.")
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    is_pretrained_list = [args.pretrained] * args.ensemble_size

    saving_path = Path(args.saving_path)

    if args.dataname == 'gtsrb':
        args.trigger_size = 8
    else:
        args.trigger_size = 2

    stacking_ensemble(args, args.dataname, args.batch_size, args.n_epochs, models_name_list, is_pretrained_list, optim_list,
                      args.ensemble_size, device, mt_mdl_name =args.mt_mdl_name, k_fold=args.k_fold, experim_num=args.experim_num,
                      saving_path=saving_path)
