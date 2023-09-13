import torch, torchvision
import torch.nn as nn


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
    model = torchvision.models.resnet18(pretrained=pretrained)
    model.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = torch.nn.Linear(in_features=512, out_features=num_classes, bias=True)
    return model

def get_resnet50(pretrained, num_classes, in_channels):
    model = torchvision.models.resnet50(pretrained=pretrained)
    model.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = torch.nn.Linear(in_features=2048, out_features=num_classes, bias=True)
    return model

def get_vgg19(pretrained, num_classes, in_channels):
    model = torchvision.models.vgg19(pretrained=pretrained)
    model.features[0] = torch.nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
    model.classifier[6] = torch.nn.Linear(4096, num_classes)
    return model

class SimpleFeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleFeedForward, self).__init__()
        
        # Define the layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # Input to Hidden Layer
        self.fc2 = nn.Linear(hidden_dim, output_dim) # Hidden to Output Layer
        self.relu = nn.ReLU()  # Activation function
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x



def get_model(model_name, num_classes, in_channels, pretrained):
    model_name = model_name.lower()
    if model_name == 'resnet18':
        return get_resnet18(pretrained, num_classes, in_channels)
    elif model_name == 'resnet50':
        return get_resnet50(pretrained, num_classes, in_channels)
    elif model_name == 'vgg19':
        return get_vgg19(pretrained, num_classes, in_channels)
    elif model_name.startswith('simple_'):
        hidden_layer_num = int(model_name.split('_')[1])
        return SimpleFeedForward(in_channels, hidden_layer_num, num_classes)
    
    else:
        raise ValueError('Invalid model name')



