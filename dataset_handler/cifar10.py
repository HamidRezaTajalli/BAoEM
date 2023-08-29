
import torch.utils.data
from torchvision import datasets, transforms

torch.manual_seed(53)
import numpy as np

np.random.seed(53)

def get_datasets_simple(root_path='./data/CIFAR10/'):
    classes_names = ('Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')
    transforms_dict = {
        'train': transforms.Compose([transforms.ToTensor()]),
        'test': transforms.Compose([transforms.ToTensor()])
    }

    train_dataset = datasets.CIFAR10(root=root_path, train=True, transform=transforms_dict['train'], download=True)
    test_dataset = datasets.CIFAR10(root=root_path, train=False, transform=transforms_dict['test'], download=True)

    train_length = int(len(train_dataset) / (10 / 9))
    validation_length = len(train_dataset) - train_length
    train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, [train_length, validation_length])
    return train_dataset, validation_dataset, test_dataset, classes_names

def get_dataloaders_simple(batch_size=64, drop_last=False, is_shuffle=True, root_path='./data/CIFAR10/'):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_workers = 2 if device.type == 'cuda' else 0

    train_dataset, validation_dataset, test_dataset, classes_names = get_datasets_simple(root_path)

    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=is_shuffle,
                                                   num_workers=num_workers, drop_last=drop_last)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                                  num_workers=num_workers, drop_last=drop_last)
    validation_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False,
                                                        num_workers=num_workers, drop_last=drop_last)

    return train_dataloader, validation_dataloader, test_dataloader
