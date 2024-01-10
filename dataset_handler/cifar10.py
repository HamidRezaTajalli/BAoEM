
import torch.utils.data
from torchvision import datasets, transforms

# torch.manual_seed(53)
import numpy as np

# np.random.seed(53)


class CustomCIFAR10(datasets.CIFAR10):
        def set_transform(self, transform):
            self.transform = transform

        def set_target_transform(self, target_transform):
            self.target_transform = target_transform


def get_cifar10_datasets(root_path='./data/CIFAR10/', tr_vl_split=None, transform=None, target_transform=None):
    """
    This function downloads and prepares the CIFAR10 dataset.

    Args:
        root_path (str, optional): The path where the CIFAR10 dataset will be downloaded. Defaults to './data/CIFAR10/'.
        tr_vl_split (float, optional): The ratio of the training set to the validation set. If None, the training set will not be split into training and validation sets. Defaults to None.

    Returns:
        tuple: Depending on the value of tr_vl_split, it returns:
            - If tr_vl_split is not None: A tuple (train_dataset, validation_dataset, test_dataset, classes_names).
            - If tr_vl_split is None: A tuple (train_dataset, test_dataset, classes_names).
    """

    
    classes_names = ('Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')


    train_dataset = CustomCIFAR10(root=root_path, train=True, transform=transform, target_transform=target_transform, download=True)
    test_dataset = CustomCIFAR10(root=root_path, train=False, transform=transform, target_transform=target_transform, download=True)

    if tr_vl_split is not None:
        train_length = int(len(train_dataset) * tr_vl_split)
        validation_length = len(train_dataset) - train_length
        train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, [train_length, validation_length])
        return train_dataset, validation_dataset, test_dataset, classes_names
    else:
        return train_dataset, test_dataset, classes_names

# def get_dataloaders_simple(batch_size=64, drop_last=False, is_shuffle=True, root_path='./data/CIFAR10/'):
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#     num_workers = 2 if device.type == 'cuda' else 0

#     train_dataset, validation_dataset, test_dataset, classes_names = get_cifar10_datasets_simple(root_path)

#     train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=is_shuffle,
#                                                    num_workers=num_workers, drop_last=drop_last)
#     test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
#                                                   num_workers=num_workers, drop_last=drop_last)
#     validation_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False,
#                                                         num_workers=num_workers, drop_last=drop_last)

#     return train_dataloader, validation_dataloader, test_dataloader

def get_general_transform_cifar10():
    gnrl_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[
                0.247, 0.243, 0.261]),
        ])
    return gnrl_transform

def get_post_poison_transform_cifar10():
    post_poison_transform = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[
                0.247, 0.243, 0.261]),
        ])
    return post_poison_transform

def get_pre_poison_transform_cifar10():
    pre_poison_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize(128),
            
        ])
    return pre_poison_transform