import torch, torchvision
import torch.nn as nn


model_func = torchvision.models.resnet50
default_weights = torchvision.models.ResNet50_Weights.DEFAULT

model = model_func(weights=default_weights)

for name, param in model.named_parameters():
    if 'bn' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=10, bias=True)
model.fc.requires_grad = True

print('==Original model==')
for name, param in model.named_parameters():
    print(name, param.requires_grad)


model.load_state_dict(torch.load('F:/surfdrive/PostDoc/Research/Essemble backdoor/03 - Model/resnet50_bd_cifar10.pt'))

print('==Backdoored model==')
for name, param in model.named_parameters():
    print(name, param.requires_grad)
