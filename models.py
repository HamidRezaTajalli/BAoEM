import torch
from torchvision import models


def get_input_channels(dataset):
    '''
    handling input types and raising exception or errors if inputs are incorrect
    '''
    ds_input_channels = {'mnist': 1, 'fmnist': 1, 'cifar10': 3, 'cifar100': 3}
    return ds_input_channels.get(dataset)


def dataset_name_check(dataset):
    dataset = dataset.lower()
    dataset_list = ['mnist', 'fmnist', 'cifar10', 'cifar100']
    if dataset not in dataset_list:
        choosing_list = [f"{number}- '{item}'" for number, item in enumerate(dataset_list, start=1)]
        raise ValueError("PLEASE INSERT CORRECT DATASET NAME:\n" + '\n'.join(choosing_list))
    return dataset


def get_num_classes(dataset):
    num_classes_list = {'mnist': 10, 'fmnist': 10, 'cifar10': 10, 'cifar100': 100}
    return num_classes_list[dataset]

def get_resnet18(pretrained, num_classes, in_channels):
    model = models.resnet18(pretrained=pretrained)
    model.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = torch.nn.Linear(in_features=512, out_features=num_classes, bias=True)
    return model


def get_model(model_name, pretrained, num_classes, in_channels):
    model_name = model_name.lower()
    if model_name == 'resnet18':
        return get_resnet18(pretrained, num_classes, in_channels)
    else:
        raise ValueError('Invalid model name')



