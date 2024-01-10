import csv
from typing import List
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset_handler.data_utils import get_datasets, poison_dataset, bootstrap_sample, get_general_transform, get_post_poison_transform, get_pre_poison_transform
from training_utils import train_one_epoch, evaluate_one_epoch, vote
from models import get_num_classes, get_input_channels, get_model

import torch.optim as optim
from pathlib import Path

def bagging_ensemble(args, dataname: str, batch_size: int, n_epochs: int, models_name_list: List[str], 
                   is_pretrained_list: List[bool], optim_list: List[str], device: torch.device, experim_num=0, saving_path: Path=Path()) -> None:
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

    train_dataset, test_dataset, classes_names = get_datasets(dataname=dataname, root_path=None, tr_vl_split=None, transform=get_pre_poison_transform(dataname))
    poisoned_trainset = poison_dataset(args, dataname=dataname, dataset=train_dataset, is_train=True, attack_name='badnet', post_transform=get_post_poison_transform(dataname))
    poisoned_testset = poison_dataset(args, dataname=dataname, dataset=test_dataset, is_train=False, attack_name='badnet', post_transform=get_post_poison_transform(dataname))
    train_dataset.set_transform(get_general_transform(dataname))
    test_dataset.set_transform(get_general_transform(dataname))

    test_dataloader = DataLoader(test_dataset, batch_size=validation_batch_size, shuffle=False, num_workers=2)
    poisoned_test_dataloader = DataLoader(poisoned_testset, batch_size=validation_batch_size, shuffle=False, num_workers=2)

    # Define Bagging Strategy
    for i, model in enumerate(model_list):
        print(f"Model {i + 1}\n-------------------------------")
        # Resample the data (with replacement)
        resampled_dataset = bootstrap_sample(poisoned_trainset)
        train_dataloader = DataLoader(resampled_dataset, batch_size=train_batch_size, shuffle=True)

        # Train the model on the resampled data
        for epoch in range(n_epochs):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            train_lss = train_one_epoch(model, train_dataloader, optimizers[i], criterion, device)
            print_string = f"train loss: {train_lss:>6}"
            print(print_string)

    # Test the ensemble on the test data
    test_acc = vote(model_list, device, test_dataloader, voting='hard', num_classes=num_classes)
    print("------------Ensemble Test Accuracy---------")
    print_string = f"test accuracy: {test_acc:>6}"
    print(print_string)
    print("-------------------------------------------")
    
    # Test the ensemble on the poisoned test data
    bd_test_acc = vote(model_list, device, poisoned_test_dataloader, voting='hard', num_classes=num_classes)
    print("------------Ensemble Poisoned Test Accuracy---------")
    print_string = f"backdoor test accuracy: {bd_test_acc:>6}"
    print(print_string)
    print("-------------------------------------------")

    # Save the results
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
        csv_writer.writerow([experim_num, f'bagging_{model_list[0].__class__.__name__}', len(model_list), dataname, test_acc, bd_test_acc])


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default='cifar10')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--device', type=str)
    parser.add_argument('--ensemble_size', type=int, default=3)
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--saving_path', type=str, default='.')
    parser.add_argument('--experim_num', type=int, default=0)
    parser.add_argument('--model_name', type=str, default='resnet50')

    
    args = parser.parse_args()

    base_model_list = [args.model_name]
    models_name_list = base_model_list * args.ensemble_size

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

    args.trigger_size = 8
    
    bagging_ensemble(args, dataname=args.dataname, batch_size=args.batch_size, n_epochs=args.n_epochs, models_name_list=models_name_list, 
                   is_pretrained_list=is_pretrained_list, optim_list=optim_list, device=device, experim_num=args.experim_num, saving_path=saving_path)