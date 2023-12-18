from typing import List
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset_handler.data_utils import get_datasets, poison_dataset
from training_utils import train_one_epoch, evaluate_one_epoch, vote
from models import get_num_classes, get_input_channels, get_model
from dataset_handler.cifar10 import get_post_poison_transform_cifar10, get_general_transform_cifar10 

import torch.optim as optim

# The bagging is done by bootstrap sampling of same models on a dataset. but voting is done by different models trained on same datast!

def voting_ensemble(dataname: str, batch_size: int, n_epochs: int, models_name_list: List[str], 
                   is_pretrained_list: List[bool], optim_list: List[str], device: torch.device, strategy='hard') -> None:
    """
    Creates an ensemble of models and trains them using voting. All models are trained on the same dataset.
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

    train_dataset, test_dataset, classes_names = get_datasets(dataname=dataname, root_path=None, tr_vl_split=None, transform=transforms.ToTensor())
    poisoned_trainset = poison_dataset(dataname=dataname, dataset=train_dataset, is_train=True, attack_name='badnet', post_transform=get_post_poison_transform_cifar10())
    poisoned_testset = poison_dataset(dataname=dataname, dataset=test_dataset, is_train=False, attack_name='badnet', post_transform=get_post_poison_transform_cifar10())
    train_dataset.set_transform(get_general_transform_cifar10())
    test_dataset.set_transform(get_general_transform_cifar10())

    # train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2)
    poisoned_train_dataloader = DataLoader(poisoned_trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=validation_batch_size, shuffle=False, num_workers=2)
    poisoned_test_dataloader = DataLoader(poisoned_testset, batch_size=validation_batch_size, shuffle=False, num_workers=2)

    # Define Voting Strategy
    for i, model in enumerate(model_list):
        print(f"Model {i + 1}\n-------------------------------")

        # Train the model on the same data
        for epoch in range(n_epochs):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            train_lss = train_one_epoch(model, poisoned_train_dataloader, optimizers[i], criterion, device)
            print_string = f"train loss: {train_lss:>6}"
            print(print_string)

            # # Test the model on the validation data
            # val_lss, val_acc = evaluate_one_epoch(model, validation_dataloader, criterion, device)
            # print_string = f"validation loss: {val_lss:>6}"
            # print(print_string)
            # print_string = f"validation accuracy: {val_acc:>6}"
            # print(print_string)

    # Test the ensemble on the test data
    test_acc = vote(model_list, 'cpu', test_dataloader, voting=strategy)
    print("------------Ensemble Test Accuracy---------")
    print_string = f"test accuracy: {test_acc:>6}"
    print(print_string)
    print("-------------------------------------------")
    
    # Test the ensemble on the poisoned test data
    test_acc = vote(model_list, 'cpu', poisoned_test_dataloader, voting=strategy)
    print("------------Ensemble Poisoned Test Accuracy---------")
    print_string = f"backdoor test accuracy: {test_acc:>6}"
    print(print_string)
    print("-------------------------------------------")



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default='cifar10')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--device', type=str)
    parser.add_argument('--strategy', type=str, default='hard')
    parser.add_argument('--ensemble_size', type=int, default=3)
    parser.add_argument('--pretrained', type=bool, default=True)
    
    args = parser.parse_args()

    base_model_list = ['resnet50', 'vgg19', 'efficientnet-b3']
    models_name_list = base_model_list * (args.ensemble_size // len(base_model_list))
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

    
    
    voting_ensemble(dataname=args.dataname, batch_size=args.batch_size, n_epochs=args.n_epochs, models_name_list=models_name_list, 
                   is_pretrained_list=is_pretrained_list, optim_list=optim_list, device=device, strategy=args.strategy)