import cv2
import math
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

# torch.manual_seed(47)
import numpy as np

# np.random.seed(47)


def create_trigger(data: np.ndarray, color, trigger_size, pos='top-right') -> np.ndarray:
    """
    Create trigger
    """

    if color == 'white':
        # Case with 1 channel
        if data.shape[0] == 1:
            value = 255
        # Case with 3 channels
        elif data.shape[0] == 3:
            value = [[[1]], [[1]], [[1]]]

    elif color == 'black':
        # Case with 1 channel
        if data.shape[0] == 1:
            value = 0
        # Case with 3 channels
        elif data.shape[0] == 3:
            value = [[[0]], [[0]], [[0]]]

    elif color == 'green':
        if data.shape[0] == 1:
            value = 0
        elif data.shape[0] == 3:
            value = [[[0]], [[1]], [[0]]]

    else:
        raise ValueError('Color not supported')

    width = data.shape[1]
    height = data.shape[2]
    size_width = trigger_size
    size_height = trigger_size

    if pos == 'top-left':
        x_begin = 0
        x_end = size_width
        y_begin = 0
        y_end = size_height

    elif pos == 'top-right':
        x_begin = int(width - size_width)
        x_end = width
        y_begin = 0
        y_end = size_height

    elif pos == 'bottom-left':
        x_begin = 0
        x_end = size_width
        y_begin = int(height - size_height)
        y_end = height
    elif pos == 'bottom-right':
        x_begin = int(width - size_width)
        x_end = width
        y_begin = int(height - size_height)
        y_end = height

    elif pos == 'middle':
        x_begin = int((width - size_width) / 2)
        x_end = int((width + size_width) / 2)
        y_begin = int((height - size_height) / 2)
        y_end = int((height + size_height) / 2)

    else:
        raise ValueError('Position not supported')

    data[:, x_begin:x_end, y_begin:y_end] = value


    return data



# def poison_batch(batch, trigger_obj, epsilon, backdoor_label):
#     data = batch[0].cpu()
#     labels = batch[1].cpu()
#     trigger_samples = int(epsilon * len(data))
#     samples_index = np.random.choice(len(data), size=trigger_samples, replace=False)

#     for ind, item in enumerate(data):
#         if ind in samples_index:
#             data[ind] = torch.from_numpy(
#                 poison_img(item.cpu().permute(1, 2, 0).numpy(), trigger_obj)).permute(2, 0, 1)
#             labels[ind] = backdoor_label

#     return data, labels


class BadnetDataset(Dataset):
    def __init__(self, data, targets, transform=None, target_transform=None):
        """
        Args:
            backdoored_data (list): List of tuples with (image, label).
        """
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transform is not None:
            data = self.transform(self.data[idx])
        else:
            data = self.data[idx]
        if self.target_transform is not None:
            label = self.target_transform(self.targets[idx])
        else:
            label = self.targets[idx]
        return data, label
    
    def set_transform(self, transform):
        self.transform = transform
    def set_target_transform(self, target_transform):
        self.target_transform = target_transform


def get_poisoned_dataset(clean_dataset, is_train: bool, trigger_size, trigger_color: str, trigger_pos: str, target_lbl: int, epsilon: float, source_label=None):
    backdoored_data = []
    bacdoored_labels = []
    samples_index = None
    if is_train:
        trigger_samples = int(epsilon * len(clean_dataset))
        samples_index = np.random.choice(len(clean_dataset), size=trigger_samples, replace=False)        

    for idx, (image, label) in enumerate(clean_dataset):
        poisoned_image = image.clone().numpy()
        poisoned_image = torch.from_numpy(create_trigger(data=poisoned_image, color=trigger_color, trigger_size=trigger_size,
                                        pos=trigger_pos))
        
        if source_label is None:
            if is_train:
                if idx in samples_index:
                    insert = (poisoned_image, target_lbl)
                else:
                    insert = (image, label)
                backdoored_data.append(insert[0])
                bacdoored_labels.append(insert[1])
            else:
                if label != target_lbl:
                    insert = (poisoned_image, target_lbl)
                    backdoored_data.append(insert[0])
                    bacdoored_labels.append(insert[1])
        else:
            if is_train:
                if idx in samples_index:
                    if label == source_label:
                        insert = (poisoned_image, target_lbl)
                    else:
                        insert = (poisoned_image, label)
                else:
                    insert = (image, label)
                backdoored_data.append(insert[0])
                bacdoored_labels.append(insert[1])
            else:
                if label != target_lbl:
                    if label == source_label:
                        insert = (poisoned_image, target_lbl)
                    else:
                        insert = (poisoned_image, label)
                    backdoored_data.append(insert[0])
                    bacdoored_labels.append(insert[1])
    return BadnetDataset(backdoored_data, bacdoored_labels)
