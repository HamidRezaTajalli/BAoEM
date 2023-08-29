
import torch
from attacks.wanet import get_grids, get_poisoned_dataset
from dataset_handler.cifar10 import get_dataloaders_simple


def test_get_poisoned_dataset():
    train_dataloader, validation_dataloader, test_dataloader = get_dataloaders_simple()

    # Define the parameters for get_poisoned_dataset
    is_train = True
    dataset = train_dataloader.dataset
    sample_dimension = (3, 32, 32)  # For CIFAR10
    s = 0.1
    k = 10
    grid_rescale = 1
    poison_rate = 0.1
    cross_ratio = 0.1
    noise_grid, identity_grid = get_grids(k, sample_dimension[1])
    target_class = 0
    num_classes = 10
    all_to_all = False
    source_label = None

    # Call the function to test
    poisoned_dataset, poisoned_labels, poisoned_indices, cross_indices = get_poisoned_dataset(
        is_train, dataset, sample_dimension, s, k, grid_rescale, poison_rate, cross_ratio, noise_grid, identity_grid,
        target_class, num_classes, all_to_all, source_label)

    # Add your assertions here to validate the output of the function
    # For example, you can check the size of the output datasets, the labels, etc.
    assert isinstance(poisoned_dataset, torch.utils.data.Dataset), "The poisoned dataset should be a torch.utils.data.Dataset object"
    assert len(poisoned_dataset) == len(dataset), "The poisoned dataset should have the same length as the original dataset"
    assert isinstance(poisoned_labels, list), "The labels of the poisoned dataset should be a list"
    assert len(poisoned_labels) == len(dataset), "The labels of the poisoned dataset should have the same length as the original dataset"
    assert isinstance(poisoned_indices, list), "The indices of the poisoned samples should be a list"
    assert len(poisoned_indices) == int(len(dataset) * poison_rate), "The number of poisoned samples should be equal to the poison rate times the length of the dataset"
    assert isinstance(cross_indices, list), "The indices of the cross noise samples should be a list"
    assert len(cross_indices) == int(len(dataset) * cross_ratio), "The number of cross noise samples should be equal to the cross ratio times the length of the dataset"
    assert all(isinstance(i, int) for i in cross_indices), "All cross indices should be valid indices of the dataset"
    assert len(set(poisoned_indices).intersection(cross_indices)) == 0, "Poisoned indices and cross indices should not overlap"
    assert all(i in range(len(dataset)) for i in cross_indices), "All cross indices should be valid indices of the dataset"
    assert all(i in range(len(dataset)) for i in poisoned_indices), "All poisoned indices should be valid indices of the dataset"
    assert len(cross_indices) == int(len(dataset) * cross_ratio), "The number of cross indices should match the cross ratio"
    assert len(poisoned_indices) == int(len(dataset) * poison_rate), "The number of poisoned indices should match the poison rate"
    assert len(poisoned_labels) == len(dataset), "The labels of the poisoned dataset should have the same length as the original dataset"
    assert len(poisoned_indices) == len(poisoned_dataset), "The poisoned dataset should have the same length as the original dataset"
    assert len(poisoned_labels) == len(dataset), "The labels of the poisoned dataset should have the same length as the original dataset"
