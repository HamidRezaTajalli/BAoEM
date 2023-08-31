import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

import torch
from wanet import get_grids, get_poisoned_dataset
from dataset_handler.mnist import get_datasets_simple


def test_get_poisoned_dataset():
    (
        train_dataset,
        validation_dataset,
        test_dataset,
        classes_names,
    ) = get_datasets_simple()

    # Define the parameters for get_poisoned_dataset
    is_train = True
    dataset = train_dataset
    sample_dimension = (1, 28, 28)  # For MNIST
    s = 0.5
    k = 4
    grid_rescale = 1
    poison_rate = 0.1
    cross_ratio = 0.2
    noise_grid, identity_grid = get_grids(k, sample_dimension[1])
    target_class = 0
    num_classes = 10
    all_to_all = False
    source_label = None

    # Call the function to test
    poisoned_dataset = get_poisoned_dataset(
        is_train,
        dataset,
        sample_dimension,
        s,
        k,
        grid_rescale,
        poison_rate,
        cross_ratio,
        noise_grid,
        identity_grid,
        target_class,
        num_classes,
        all_to_all,
        source_label,
    )
    poisoned_data, poisoned_labels, poisoned_indices, cross_indices = (
        poisoned_dataset.images,
        poisoned_dataset.labels,
        poisoned_dataset.poison_indices,
        poisoned_dataset.cross_indices,
    )

    if is_train:
        # Add your assertions here to validate the output of the function
        # For example, you can check the size of the output datasets, the labels, etc.
        assert isinstance(
            poisoned_dataset, (torch.utils.data.Dataset, torch.utils.data.Subset)
        ), "The poisoned dataset should be a torch.utils.data.Dataset or torch.utils.data.Subset object"
        assert len(poisoned_dataset) == len(
            dataset
        ), "The poisoned dataset should have the same length as the original dataset"
        assert len(poisoned_labels) == len(
            dataset
        ), "The labels of the poisoned dataset should have the same length as the original dataset"
        assert isinstance(
            poisoned_indices, list
        ), "The indices of the poisoned samples should be a list"
        assert len(poisoned_indices) == int(
            len(dataset) * poison_rate
        ), "The number of poisoned samples should be equal to the poison rate times the length of the dataset"
        assert isinstance(
            cross_indices, list
        ), "The indices of the cross noise samples should be a list"
        assert len(cross_indices) == int(
            len(dataset) * cross_ratio
        ), "The number of cross noise samples should be equal to the cross ratio times the length of the dataset"
        assert all(
            isinstance(i, int) for i in cross_indices
        ), "All cross indices should be valid indices of the dataset"
        assert (
            len(set(poisoned_indices).intersection(cross_indices)) == 0
        ), "Poisoned indices and cross indices should not overlap"
        assert all(
            i in range(len(dataset)) for i in cross_indices
        ), "All cross indices should be valid indices of the dataset"
        assert all(
            i in range(len(dataset)) for i in poisoned_indices
        ), "All poisoned indices should be valid indices of the dataset"
        assert len(cross_indices) == int(
            len(dataset) * cross_ratio
        ), "The number of cross indices should match the cross ratio"
        assert len(poisoned_indices) == int(
            len(dataset) * poison_rate
        ), "The number of poisoned indices should match the poison rate"
        assert len(poisoned_labels) == len(
            dataset
        ), "The labels of the poisoned dataset should have the same length as the original dataset"
        # Iterate over train_dataset and poisoned_dataset to check if both data and label items in both datasets are of the same type and shape
        for i in range(len(dataset)):
            train_data, train_label = dataset[i]
            p_data, poisoned_label = poisoned_dataset[i]

            assert type(train_data) == type(p_data), f"Data items in both datasets should be of the same type. train_data type: {type(train_data)} poisoned_dataset type: {type(poisoned_dataset)}"
            assert type(train_label) == type(poisoned_label), f"Label items in both datasets should be of the same type. train_label type: {type(train_label)} poisoned_label type: {type(poisoned_label)}"
            assert (
                train_data.shape == p_data.shape
            ), f"Data items in both datasets should have the same shape. train_data shape: {train_data.shape} poisoned_dataset shape: {poisoned_dataset.shape}"


    import matplotlib.pyplot as plt

    # # Display an image from the original dataset
    # img, label = train_dataset[0]
    # plt.figure(figsize=(2, 2))
    # plt.title(f"Original Dataset: Label {label}")
    # plt.imshow(img.permute(1, 2, 0))  # permute the tensor dimensions to match the expected input of plt.imshow
    # plt.show()
    # plt.savefig("original.png")
    # print("Original label:", label)

    # # Display an image from the poisoned dataset
    # p_img, p_label = poisoned_dataset[0]
    # plt.figure(figsize=(2, 2))
    # plt.title(f"Poisoned Dataset: Label {p_label}")
    # plt.imshow(p_img.permute(1, 2, 0))  # permute the tensor dimensions to match the expected input of plt.imshow
    # plt.show()
    # plt.savefig("poisoned.png")
    # print("Poisoned label:", p_label)

    # ... existing code ...


if __name__ == "__main__":
    test_get_poisoned_dataset()

