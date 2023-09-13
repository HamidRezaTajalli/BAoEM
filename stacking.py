from typing import List
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold

from dataset_handler.data_utils import create_stacked_dataset, get_datasets_simple
from training_utils import train_stacked_models, train_one_epoch, stack_outputs
from models import get_num_classes, get_input_channels, get_model

import torch.optim as optim

def stack_ensemble(k_fold: int, dataname: str, batch_size: int, n_epochs: int, models_name_list: List[str], 
                   is_pretrained_list: List[bool], optim_list: List[str], device: torch.device, mt_mdl_name: str='simple_2000', tr_vl_split=0.8) -> None:
    """
    Creates an ensemble of models and trains them using stacking. then creates a meta-dataset with output of the base models and trains a meta-model on it.
    then test the whole ensemble on the test dataset.

    Args:
        k_fold (int): Number of folds for cross-validation. If None, the dataset is split into training and validation sets.
        dataname (str): Name of the dataset.
        batch_size (int): Size of the batch for training, validation and testing.
        n_epochs (int): Number of epochs for training.
        models_name_list (List[str]): List of names of the models to be used in the ensemble.
        is_pretrained_list (List[bool]): List indicating whether the corresponding model is pretrained or not.
        optim_list (List[str]): List of optimizers to be used for the corresponding models.
        device (torch.device): Device to be used for training (CPU or GPU).
        mt_mdl_name (str, optional): Name of the meta model. Defaults to 'simple_2000'. You can change the hidden layer size of the meta model by changing the number in the name.
        tr_vl_split (float, optional): Ratio of the training set to the validation set. Defaults to 0.8. if k_fold is not None, this argument is ignored.
    """
    
    # Get the number of classes and input channels from the dataset
    num_classes = get_num_classes(dataname)
    n_in_channels = get_input_channels(dataname)

    # Create a list of models
    model_list = [get_model(model_name, num_classes, n_in_channels, pretrained) for model_name, pretrained in zip(models_name_list, is_pretrained_list)]

    # Define the optimizers
    optim_dict = {'sgd': optim.SGD, 'adam': optim.Adam}
    optimizers = [optim_dict[optim_name](model.parameters(), lr=0.001) for model, optim_name in zip(model_list, optim_list)]

    # Define the loss function
    loss_function = CrossEntropyLoss()  # Replace with the actual loss function

    # Initialize the meta-dataset
    meta_dataset = None

    # If k_fold is None, split the train_dataset into training and validation sets based on tr_vl_split ratio. 
    # then train the base models on the training set and create the meta-dataset using the validation set.
    if k_fold is None:
        train_dataset, validation_dataset, test_dataset, classes_names = get_datasets_simple(dataname=dataname, tr_vl_split=tr_vl_split)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        train_stacked_models(train_dataloader, model_list, n_epochs, optimizers, loss_function, device)
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        meta_dataset = create_stacked_dataset(model_list, validation_dataloader, device)

    # If k_fold is not None, perform k-fold cross-validation and create the meta-dataset using the k-fold cross validation sets.
    else:        
        train_dataset, test_dataset, classes_names = get_datasets_simple(dataname=dataname, tr_vl_split=None)
        meta_dataset_list = []
        kf = KFold(n_splits=k_fold, shuffle=False)
        for train_idx, val_idx in kf.split(train_dataset.data):
            train_subsampler = SubsetRandomSampler(train_idx)
            val_subsampler = SubsetRandomSampler(val_idx)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_subsampler, num_workers=2)
            train_stacked_models(train_dataloader, model_list, n_epochs, optimizers, loss_function, device)
            validation_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_subsampler, num_workers=2)
            meta_dataset_list.append(create_stacked_dataset(model_list, validation_dataloader, device))
        meta_dataset = torch.utils.data.ConcatDataset(meta_dataset_list)


    meta_dataloader = DataLoader(meta_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    meta_model = get_model(mt_mdl_name, num_classes, num_classes*len(model_list), False)

    meta_optimizer = optim.Adam(meta_model.parameters(), lr=0.001)

    # Train the meta model
    for epoch in range(n_epochs):
        train_one_epoch(meta_model, meta_dataloader, meta_optimizer, loss_function, device)


    # Test the whole ensemble on the test dataset
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)   
    meta_model.eval()
    total_samples = 0
    correct_predictions = 0
    for batch_idx, (data, labels) in enumerate(test_dataloader):
        data, labels = data.to(device), labels.to(device)
        meta_input = stack_outputs(model_list, data, device)
        output = meta_model(meta_input)
        _, predicted = torch.max(output.data, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)
    accuracy = (correct_predictions / total_samples) * 100
    print(f'Test Accuracy of the model on the test images: {accuracy}%')


# TODO: check for each individual model accuracy after training.