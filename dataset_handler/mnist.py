import torch.utils.data
from torchvision import datasets, transforms
from sklearn.utils import resample
import torch
from PIL import Image


torch.manual_seed(53)
import numpy as np

np.random.seed(53)


def get_datasets_simple():
    classes_names = ('Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight')

    transforms_dict = {
        'train': transforms.Compose([transforms.ToTensor(),
                                     # transforms.Normalize((0.1307,), (0.3081,))
                                     ]),
        'test': transforms.Compose([transforms.ToTensor(),
                                    # transforms.Normalize((0.1307,), (0.3081,))
                                    ])
    }

    train_dataset = datasets.MNIST(root='./data/MNIST/', train=True, transform=transforms_dict['train'], download=True)
    test_dataset = datasets.MNIST(root='./data/MNIST/', train=False, transform=transforms_dict['test'], download=True)

    train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset,
                                                                      [int(len(train_dataset) / (12 / 11)),
                                                                       int(len(train_dataset) / (12))])
    return train_dataset, validation_dataset, test_dataset, classes_names


def get_dataloaders_simple(batch_size, drop_last, is_shuffle):
    drop_last = drop_last
    is_shuffle = is_shuffle
    batch_size = batch_size
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_workers = 2 if device.type == 'cuda' else 0

    train_dataset, validation_dataset, test_dataset, classes_names = get_datasets_simple()

    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=is_shuffle,
                                                   num_workers=num_workers, drop_last=drop_last)

    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                                  shuffle=is_shuffle, num_workers=num_workers, drop_last=drop_last)

    validation_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size,
                                                        shuffle=is_shuffle, num_workers=num_workers,
                                                        drop_last=drop_last)

    return {'train': train_dataloader,
            'test': test_dataloader,
            'validation': validation_dataloader}, classes_names


class BtstrpDataset(torch.utils.data.Dataset):
    """Resampled Dataset by bootstrapping."""

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             # transforms.Normalize((0.1307,), (0.3081,))
                                             ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        sample = Image.fromarray(sample.numpy(), mode="L")

        if self.transform:
            sample = self.transform(sample)
        return sample, self.labels[idx]


def bootstrap_sample(dataset, resample_rate=None):
    """
    :param dataset: A torch.utils.data.Dataset object or torch.utils.data.Subset object
    :param resample_rate: The percentage of the dataset to sample
    :return: A new dataset obj
    """

    n_samples = int(len(dataset) * resample_rate) if resample_rate is not None else None
    # Resample the data (with replacement)
    if isinstance(dataset, torch.utils.data.Subset):
        resampled_data, resampled_labels = resample(dataset.dataset.data[dataset.indices],
                                                    dataset.dataset.targets[dataset.indices],
                                                    replace=True,
                                                    n_samples=n_samples)
    elif isinstance(dataset, torch.utils.data.Dataset):
        resampled_data, resampled_labels = resample(dataset.data,
                                                    dataset.targets,
                                                    replace=True,
                                                    n_samples=n_samples)
    else:
        raise TypeError("dataset must be a torch.utils.data.Dataset object or torch.utils.data.Subset object")

    # Create and return a new dataset
    return BtstrpDataset(resampled_data, resampled_labels)
