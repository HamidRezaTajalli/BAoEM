import torch.utils.data
import torchvision
from torchvision import transforms
from sklearn.utils import resample
from PIL import Image


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
