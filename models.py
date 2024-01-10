import torch, torchvision
import torch.nn as nn


def get_input_channels(dataset):
    '''
    handling input types and raising exception or errors if inputs are incorrect
    '''
    ds_input_channels = {'mnist': 1, 'fmnist': 1, 'cifar10': 3, 'cifar100': 3, 'gtsrb': 3, 'tinyimagenet': 3, 'celeba': 3}
    return ds_input_channels.get(dataset)


def dataset_name_check(dataset):
    dataset = dataset.lower()
    dataset_list = ['mnist', 'fmnist', 'cifar10', 'cifar100', 'gtsrb', 'tinyimagenet', 'celeba']
    if dataset not in dataset_list:
        choosing_list = [f"{number}- '{item}'" for number, item in enumerate(dataset_list, start=1)]
        raise ValueError("PLEASE INSERT CORRECT DATASET NAME:\n" + '\n'.join(choosing_list))
    return dataset


def get_num_classes(dataset):
    num_classes_list = {'mnist': 10, 'fmnist': 10, 'cifar10': 10, 'cifar100': 100, 'gtsrb': 43, 'tinyimagenet': 200, 'celeba': 2}
    return num_classes_list[dataset]

def get_resnet_model(model_name, pretrained, num_classes, in_channels):
    if model_name == 'resnet18':
        model_func = torchvision.models.resnet18
        default_weights = torchvision.models.ResNet18_Weights.DEFAULT
    elif model_name == 'resnet50':
        model_func = torchvision.models.resnet50
        default_weights = torchvision.models.ResNet50_Weights.DEFAULT
    else:
        raise ValueError('Invalid ResNet model name')

    if pretrained and in_channels != 1:
        model = model_func(weights=default_weights)
        for name, param in model.named_parameters():
            if 'bn' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=num_classes, bias=True)
        model.fc.requires_grad = True
    else:
        model = model_func(num_classes=num_classes)
        model.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
    return model

def get_vgg19_bn(pretrained, num_classes, in_channels):
    
    if pretrained and in_channels != 1:
        default_weights = torchvision.models.VGG19_BN_Weights.DEFAULT
        model = torchvision.models.vgg19_bn(weights=default_weights)
        for param, layer_class in zip(model.features.parameters(), model.features):
                if type(layer_class) is nn.BatchNorm2d:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        model.classifier.requires_grad = True
    else:
        model = torchvision.models.vgg19_bn(num_classes=num_classes)
        model.features[0] = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
    return model


def get_efficientnet(model_name, pretrained, num_classes, in_channels):
    if model_name == 'efficientnet-b0':
        model_func = torchvision.models.efficientnet_b0
        default_weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    elif model_name == 'efficientnet-b1':
        model_func = torchvision.models.efficientnet_b1
        default_weights = torchvision.models.EfficientNet_B1_Weights.DEFAULT
    elif model_name == 'efficientnet-b2':
        model_func = torchvision.models.efficientnet_b2
        default_weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    elif model_name == 'efficientnet-b3':
        model_func = torchvision.models.efficientnet_b3
        default_weights = torchvision.models.EfficientNet_B3_Weights.DEFAULT
    elif model_name == 'efficientnet-b4':
        model_func = torchvision.models.efficientnet_b4
        default_weights = torchvision.models.EfficientNet_B4_Weights.DEFAULT
    elif model_name == 'efficientnet-b5':
        model_func = torchvision.models.efficientnet_b5
        default_weights = torchvision.models.EfficientNet_B5_Weights.DEFAULT
    elif model_name == 'efficientnet-b6':
        model_func = torchvision.models.efficientnet_b6
        default_weights = torchvision.models.EfficientNet_B6_Weights.DEFAULT
    elif model_name == 'efficientnet-b7':
        model_func = torchvision.models.efficientnet_b7
        default_weights = torchvision.models.EfficientNet_B7_Weights.DEFAULT
    else:
        raise ValueError('Invalid EfficientNet model name')

    if pretrained and in_channels != 1:
        model = model_func(weights=default_weights)
        for param in model.parameters():
            param.requires_grad = False
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                for param in module.parameters():
                    param.requires_grad = True

        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        for param in model.classifier[-1].parameters():
            param.requires_grad = True
    else:
        model = model_func(num_classes=num_classes)
        model.features[0] = torchvision.ops.Conv2dNormActivation(in_channels, model.features[0].out_channels, kernel_size=model.features[0].kernel_size, stride=model.features[0].stride, norm_layer=model.features[0].norm_layer, activation_layer=model.features[0].activation_layer)
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



class AdvancedFeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AdvancedFeedForward, self).__init__()
        
        # Define the layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # Input to Hidden Layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) # Hidden to Hidden Layer
        self.fc3 = nn.Linear(hidden_dim, output_dim) # Hidden to Output Layer
        self.relu = nn.ReLU()  # Activation function
        # self.dropout = nn.Dropout(0.5)  # Dropout layer
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        # x = self.dropout(x)
        x = self.relu(self.fc2(x))
        # x = self.dropout(x)
        x = self.fc3(x)
        return x



def get_model(model_name, num_classes, in_channels, pretrained):
    model_name = model_name.lower()
    if model_name.startswith('resnet'):
        return get_resnet_model(model_name, pretrained, num_classes, in_channels)
    elif model_name.startswith('efficientnet'):
        return get_efficientnet(model_name, pretrained, num_classes, in_channels)
    elif model_name == 'vgg19':
        return get_vgg19_bn(pretrained, num_classes, in_channels)
    elif model_name.startswith('simple_'):
        hidden_layer_num = int(model_name.split('_')[1])
        return SimpleFeedForward(in_channels, hidden_layer_num, num_classes)
    elif model_name.startswith('advanced_'):
        hidden_layer_num = int(model_name.split('_')[1])
        return AdvancedFeedForward(in_channels, hidden_layer_num, num_classes)
    
    else:
        raise ValueError('Invalid model name')



