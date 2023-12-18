import torch.utils.data
from torchvision import datasets, transforms

torch.manual_seed(53)
import numpy as np

np.random.seed(53)


class CustomGTSRB(datasets.GTSRB):
    def set_transform(self, transform):
        self.transform = transform

    def set_target_transform(self, target_transform):
        self.target_transform = target_transform


def get_gtsrb_datasets(root_path='./data/GTSRB/', tr_vl_split=None, transform=None, target_transform=None):
    """
    This function downloads and prepares the GTSRB dataset.

    Args:
        root_path (str, optional): The path where the GTSRB dataset will be downloaded. Defaults to './data/GTSRB/'.
        tr_vl_split (float, optional): The ratio of the training set to the validation set. If None, the training set will not be split into training and validation sets. Defaults to None.

    Returns:
        tuple: Depending on the value of tr_vl_split, it returns:
            - If tr_vl_split is not None: A tuple (train_dataset, validation_dataset, test_dataset, classes_names).
            - If tr_vl_split is None: A tuple (train_dataset, test_dataset, classes_names).
    """

    # GTSRB has 43 classes
    classes_names = (
        'Speed limit (20km/h)',
        'Speed limit (30km/h)',
        'Speed limit (50km/h)',
        'Speed limit (60km/h)',
        'Speed limit (70km/h)',
        'Speed limit (80km/h)',
        'End of speed limit (80km/h)',
        'Speed limit (100km/h)',
        'Speed limit (120km/h)',
        'No passing',
        'No passing for vehicles over 3.5 metric tons',
        'Right-of-way at the next intersection',
        'Priority road',
        'Yield',
        'Stop',
        'No vehicles',
        'Vehicles over 3.5 metric tons prohibited',
        'No entry',
        'General caution',
        'Dangerous curve to the left',
        'Dangerous curve to the right',
        'Double curve',
        'Bumpy road',
        'Slippery road',
        'Road narrows on the right',
        'Road work',
        'Traffic signals',
        'Pedestrians',
        'Children crossing',
        'Bicycles crossing',
        'Beware of ice/snow',
        'Wild animals crossing',
        'End of all speed and passing limits',
        'Turn right ahead',
        'Turn left ahead',
        'Ahead only',
        'Go straight or right',
        'Go straight or left',
        'Keep right',
        'Keep left',
        'Roundabout mandatory',
        'End of no passing',
        'End of no passing by vehicles over 3.5 metric tons'
    )

    train_dataset = CustomGTSRB(root=root_path, split='train', transform=transform, target_transform=target_transform, download=True)
    test_dataset = CustomGTSRB(root=root_path, split='test', transform=transform, target_transform=target_transform, download=True)

    if tr_vl_split is not None:
        train_length = int(len(train_dataset) * tr_vl_split)
        validation_length = len(train_dataset) - train_length
        train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, [train_length, validation_length])
        return train_dataset, validation_dataset, test_dataset, classes_names
    else:
        return train_dataset, test_dataset, classes_names

def get_general_transform_gtsrb():
    gnrl_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return gnrl_transform

def get_post_poison_transform_gtsrb():
    post_poison_transform = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return post_poison_transform