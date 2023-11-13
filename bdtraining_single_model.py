from copy import deepcopy
from typing import List
import torch, torchvision
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from torchvision import transforms

from dataset_handler.data_utils import create_stacked_dataset, get_datasets, poison_dataset
from training_utils import train_stacked_models, train_one_epoch, stack_outputs
from models import get_num_classes, get_input_channels, get_model
from dataset_handler.cifar10 import get_post_poison_transform_cifar10, get_cifar10_datasets, get_general_transform_cifar10 


import torch.optim as optim


def backdoor(model_name: str) -> None:
    
    # Get the number of classes and input channels from the dataset
    dataname = 'cifar10'
    batch_size = 128
    n_epochs = 20
    epsilon = 0.02
    trigger_scale = 0.08
    color = 'green'
    target_label = 0
    num_classes = get_num_classes(dataname)
    n_in_channels = get_input_channels(dataname)
    lr = 0.001   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#######################################################################trial########################################
    model = get_model(model_name, num_classes, n_in_channels, True)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = CrossEntropyLoss()

    # unfreeze all layers except the last one
    if model_name.startswith('resnet'):
        for param in model.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = False
    if model_name.startswith('vgg'):
        for param in model.parameters():
            param.requires_grad = True
        for param in model.classifier[-1].parameters():
            param.requires_grad = False

    

    trainset, testset, classes_names = get_datasets(dataname=dataname, root_path=None, tr_vl_split=None, transform=transforms.ToTensor())
    poisoned_trainset = poison_dataset(dataname=dataname, dataset=trainset, is_train=True, attack_name='badnet', post_transform=get_post_poison_transform_cifar10())
    poisoned_testset = poison_dataset(dataname=dataname, dataset=testset, is_train=False, attack_name='badnet', post_transform=get_post_poison_transform_cifar10())
    testset.set_transform(get_general_transform_cifar10())
    trainset.set_transform(get_general_transform_cifar10())


    train_dataloader = DataLoader(poisoned_trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    bd_test_dataloader = DataLoader(poisoned_testset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    model.train()
    for epoch in range(n_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_dataloader)}, Accuracy: {100 * correct / total}%')
    
    # Testing the model on bd_test_dataset
    correct_bd = 0
    total_bd = 0
    model.eval()
    with torch.no_grad():
        for data in bd_test_dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_bd += labels.size(0)
            correct_bd += (predicted == labels).sum().item()
    print(f'Accuracy of the model on the bd_test_dataset: {100 * correct_bd / total_bd}%')

    # Testing the model on test_dataset
    correct_test = 0
    total_test = 0
    model.eval()
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
    print(f'Accuracy of the model on the test_dataset: {100 * correct_test / total_test}%')

    # Save the model
    from datetime import datetime
    model_type = type(model).__name__
    current_date = datetime.now().strftime('%Y-%m-%d')
    torch.save(model.state_dict(), f'frozen_lstlyr_{model_type}_{current_date}.pth')



#######################################################################trial################################################


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Backdoor model training')
    parser.add_argument('--model', type=str, required=True, help='Name of the model to backdoor')

    args = parser.parse_args()

    backdoor(args.model)

