
import torch.utils.data
from torchvision import datasets, transforms

torch.manual_seed(53)
import numpy as np

np.random.seed(53)

def get_mnist_datasets_simple(root_path='./data/MNIST/', tr_vl_split=None):
    classes_names = ('Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine')
    transforms_dict = {
        'train': transforms.Compose([transforms.ToTensor(),
                                     # transforms.Normalize((0.1307,), (0.3081,))
                                     ]),
        'test': transforms.Compose([transforms.ToTensor(),
                                    # transforms.Normalize((0.1307,), (0.3081,))
                                    ])
    }

    train_dataset = datasets.MNIST(root=root_path, train=True, transform=transforms_dict['train'], download=True)
    test_dataset = datasets.MNIST(root=root_path, train=False, transform=transforms_dict['test'], download=True)

    train_length = int(len(train_dataset) * 0.9)
    validation_length = len(train_dataset) - train_length
    train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, [train_length, validation_length])
    return train_dataset, validation_dataset, test_dataset, classes_names

# def get_dataloaders_simple(batch_size=128, drop_last=False, is_shuffle=True, root_path='./data/MNIST/'):
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#     num_workers = 2 if device.type == 'cuda' else 0

#     train_dataset, validation_dataset, test_dataset, classes_names = get_mnist_datasets_simple(root_path)

#     train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=is_shuffle,
#                                                    num_workers=num_workers, drop_last=drop_last)
#     test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
#                                                   num_workers=num_workers, drop_last=drop_last)
#     validation_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False,
#                                                         num_workers=num_workers, drop_last=drop_last)

#     return train_dataloader, validation_dataloader, test_dataloader
