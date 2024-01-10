from typing import List
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from dataset_handler.data_utils import get_datasets, bootstrap_sample
from training_utils import train_one_epoch, evaluate_one_epoch, vote
from models import get_num_classes, get_input_channels, get_model

import torch.optim as optim

def bagging_ensemble(dataname: str, batch_size: int, n_epochs: int, models_name_list: List[str], 
                   is_pretrained_list: List[bool], optim_list: List[str], device: torch.device, tr_vl_split=0.8) -> None:
    """
    Creates an ensemble of models and trains them using bagging. Each model is trained on a bootstrap sample of the original dataset.
    The ensemble is then tested on the test dataset.

    Args:
        dataname (str): Name of the dataset.
        n_epochs (int): Number of epochs for training.
        models_name_list (List[str]): List of names of the models to be used in the ensemble.
        is_pretrained_list (List[bool]): List indicating whether the corresponding model is pretrained or not.
        optim_list (List[str]): List of optimizers to be used for the corresponding models.
        device (torch.device): Device to be used for training (CPU or GPU).
    """
    
    train_batch_size, validation_batch_size = batch_size, batch_size
    num_classes = get_num_classes(dataname)
    n_in_channels = get_input_channels(dataname)
    lr = 0.001

    # Create a list of models
    model_list = [get_model(model_name, num_classes, n_in_channels, pretrained) for model_name, pretrained in zip(models_name_list, is_pretrained_list)]

    # Define the optimizers
    optim_dict = {'sgd': optim.SGD, 'adam': optim.Adam}
    optimizers = [optim_dict[optim_name](model.parameters(), lr=lr) for model, optim_name in zip(model_list, optim_list)]

    criterion = CrossEntropyLoss()
    train_dataset, validation_dataset, test_dataset, classes_names = get_datasets(dataname=dataname, tr_vl_split=tr_vl_split)
    validation_dataloader = DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=validation_batch_size, shuffle=False)

    # Define Bagging Strategy
    for i, model in enumerate(model_list):
        print(f"Model {i + 1}\n-------------------------------")
        # Resample the data (with replacement)
        resampled_dataset = bootstrap_sample(train_dataset)
        train_dataloader = DataLoader(resampled_dataset, batch_size=train_batch_size, shuffle=True)

        # Train the model on the resampled data
        for epoch in range(n_epochs):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            train_lss = train_one_epoch(model, train_dataloader, optimizers[i], criterion, device)
            print_string = f"train loss: {train_lss:>6}"
            print(print_string)

            # Test the model on the validation data
            val_lss, val_acc = evaluate_one_epoch(model, validation_dataloader, criterion, device)
            print_string = f"validation loss: {val_lss:>6}"
            print(print_string)
            print_string = f"validation accuracy: {val_acc:>6}"
            print(print_string)

    # Test the ensemble on the test data
    test_acc = vote(model_list, device, test_dataloader, voting='hard', num_classes=num_classes)
    print("------------Ensemble Test Accuracy---------")
    print_string = f"test accuracy: {test_acc:>6}"
    print(print_string)
    print("-------------------------------------------")

