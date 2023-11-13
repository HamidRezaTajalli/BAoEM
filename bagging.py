# from typing import List
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader

# # local imports
# import models
# from dataset_handler.data_utils import bootstrap_sample, get_datasets
# from training_utils import train_one_epoch, evaluate_one_epoch, test_ensemble_bagging


# def bagging_ensemble(ds_name: str, n_epochs: int, models_name_list: List[str], 
#                      is_pretrained_list: List[bool], optim_list: List[str], device: torch.device):
#     """
#     Creates an ensemble of models and trains them using bagging. Each model is trained on a bootstrap sample of the original dataset.
#     The ensemble is then tested on the test dataset.

#     Args:
#         ds_name (str): Name of the dataset.
#         n_epochs (int): Number of epochs for training.
#         models_name_list (List[str]): List of names of the models to be used in the ensemble.
#         is_pretrained_list (List[bool]): List indicating whether the corresponding model is pretrained or not.
#         optim_list (List[str]): List of optimizers to be used for the corresponding models.
#         device (torch.device): Device to be used for training (CPU or GPU).
#     """
    
#     train_batch_size, validation_batch_size = 128, 64
#     num_classes = models.get_num_classes(ds_name)
#     in_channels = models.get_input_channels(ds_name)

#     # Create a list of models
#     model_list = [models.get_model(model_name, num_classes, in_channels, pretrained).to(device) for model_name, pretrained in zip(models_name_list, is_pretrained_list)]

#     # Define the optimizers
#     optim_dict = {'sgd': optim.SGD, 'adam': optim.Adam}
#     optimizers = [optim_dict[optim_name](model.parameters()) for model, optim_name in zip(model_list, optim_list)]

#     criterion = nn.CrossEntropyLoss()
#     train_dataset, validation_dataset, test_dataset, classes_names = get_datasets()
#     validation_dataloader = DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True)
#     test_dataloader = DataLoader(test_dataset, batch_size=validation_batch_size, shuffle=True)

#     # Define Bagging Strategy
#     for i, model in enumerate(model_list):
#         print(f"Model {i + 1}\n-------------------------------")
#         # Resample the data (with replacement)
#         resampled_dataset = bootstrap_sample(train_dataset)
#         train_dataloader = DataLoader(resampled_dataset, batch_size=train_batch_size, shuffle=True)

#         # Train the model on the resampled data
#         for epoch in range(n_epochs):
#             print(f"Epoch {epoch + 1}\n-------------------------------")
#             train_lss = train_one_epoch(model, train_dataloader, optimizers[i], criterion, device)
#             print_string = f"train loss: {train_lss:>6}"
#             print(print_string)

#             # Test the model on the validation data
#             val_lss, val_acc = evaluate_one_epoch(model, validation_dataloader, criterion, device)
#             print_string = f"validation loss: {val_lss:>6}"
#             print(print_string)
#             print_string = f"validation accuracy: {val_acc:>6}"
#             print(print_string)

#     # Test the ensemble on the test data
#     test_acc = test_ensemble_bagging(model_list, device, test_dataloader, criterion)
#     print("------------Ensemble Test Accuracy---------")
#     print_string = f"test accuracy: {test_acc:>6}"
#     print(print_string)
#     print("-------------------------------------------")


import torch
from torchvision import datasets, transforms
from dataset_handler.data_utils import get_datasets
from attacks.badnet import create_trigger
import numpy as np

def main():
    

    trainset, testset, classes_names = get_datasets("cifar10", tr_vl_split=None, transform=transforms.ToTensor())



    # Get a batch of images and labels
    dataiter = iter(trainset)
    images, labels = next(dataiter)


    triggered_image = create_trigger(images.clone().numpy(), color='green', trigger_size=5, pos='top-right')
    triggered_image2 = create_trigger(trainset.data[0].copy().transpose(2,0,1), color='green', trigger_size=5, pos='top-right')
    # Compare the two data inputs for create_trigger()
    




    # Convert the first image to numpy array and apply the trigger
    # image = images[0].numpy().transpose((1, 2, 0))  # Convert the image to HWC format
    # triggered_image = create_trigger(image, 'white', 5, 'top-right')

    # Display the original and triggered images
    # plt.subplot(1, 2, 1)
    # plt.imshow(image)
    # plt.title('Original Image')

    # plt.subplot(1, 2, 2)
    # plt.imshow(triggered_image)
    # plt.title('Triggered Image')

    # plt.show()

if __name__ == "__main__":
    main()
