import torch.utils.data
import torchvision
from torchvision import transforms
from sklearn.utils import resample
from PIL import Image

from training_utils import stack_outputs

from dataset_handler.mnist import get_mnist_datasets_simple
from dataset_handler.cifar10 import get_cifar10_datasets_simple


class BtstrpDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = Image.fromarray(sample)
        if self.transform:
            sample = self.transform(sample)
        return sample, self.labels[idx]


def bootstrap_sample(dataset, resample_rate=None):
    n_samples = int(len(dataset) * resample_rate) if resample_rate is not None else None

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

    return BtstrpDataset(resampled_data, resampled_labels)


# this is a denormalizer class which inherits from the torchvision.transforms.Normalize class but does the opposite of normalization
# it is used to convert the normalized images back to their original values
class Denormalize(torchvision.transforms.Normalize):
    def __init__(self, mean, std, inplace=False):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / std
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv, inplace=inplace)



def create_stacked_dataset(models, dataloader, device):
    """
    Creates a stacked dataset from the given models and dataloader.

    This function takes a list of models and a dataloader, and generates a new dataset where each sample is the stacked
    output of all models for the corresponding input sample in the dataloader. The labels of the new dataset are the same
    as the original dataloader.

    Args:
        models (list): A list of PyTorch models.
        dataloader (DataLoader): A PyTorch DataLoader object.

    Returns:
        TensorDataset: A PyTorch TensorDataset object containing the stacked outputs of the models as data and the original
        labels from the dataloader.
    """
    stacked_samples = []
    stacked_labels = []
    for batch_idx, (data, labels) in enumerate(dataloader):
        stacked_output = stack_outputs(models, data, device)
        stacked_samples.append(stacked_output)
        stacked_labels.append(labels)
    stacked_dataset = torch.utils.data.TensorDataset(torch.cat(stacked_samples, dim=0), torch.cat(stacked_labels, dim=0))
    return stacked_dataset


def get_datasets_simple(dataname: str, root_path='./data/CIFAR10/', tr_vl_split=None):
    Switcher = {
        'mnist': get_mnist_datasets_simple,
        'cifar10': get_cifar10_datasets_simple
    }
    func = Switcher.get(dataname, lambda: "Invalid dataset name")
    return func(root_path, tr_vl_split)
